import logging

# save original method so we don't lose other warnings
_original_warning = logging.Logger.warning

def _filter_fast_tfmr(self, msg, *args, **kwargs):
    """Swallow ONLY the MoLFormer CUDA-kernel fallback line."""
    if "Falling back to (slow) pytorch implementation" in str(msg):
        return
    _original_warning(self, msg, *args, **kwargs)

logging.Logger.warning = _filter_fast_tfmr

import argparse
import tdc
import wandb
import numpy as np
import pandas as pd
import os
import pickle
import math
import nltk
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from scoring_function import get_scores, int_div
from rxnflow.tasks.unidock_vina import VinaReward
from utils import mutate, diff_mask_molformer
from synthesis_eval import calculate_molecular_diversity  # synflownet version evaluation (for sEH)
from chem_metrics import mol2sascore, calc_diversity, compute_diverse_top_k  # rxnflow version evaluation (for vina reward)
from replay_buffer import ReplayBuffer
from gflownet.utils import sascore
from gflownet.utils.conditioning import MultiObjectiveWeightedPreferences
from gflownet.utils.config import ConditionalsConfig

from rxnflow.envs.action import Protocol, RxnAction, RxnActionType
from rxnflow.envs.reaction import BiReaction, Reaction, UniReaction
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree
from pathlib import Path
from numpy.typing import NDArray

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ChemicalFilter:
    def __init__(self, catalog: str, property_rule: str):
        self.property_rule = property_rule
        # Catalog names include:
        #   - Structural alert catalogs: "PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC"
        #   - Property-rule tags: "lipinski", "veber"

        catalog_map = {
            "PAINS_A": FilterCatalogParams.FilterCatalogs.PAINS_A,
            "PAINS_B": FilterCatalogParams.FilterCatalogs.PAINS_B,
            "PAINS_C": FilterCatalogParams.FilterCatalogs.PAINS_C,
            "BRENK": FilterCatalogParams.FilterCatalogs.BRENK,
            "NIH": FilterCatalogParams.FilterCatalogs.NIH,
            "ZINC": FilterCatalogParams.FilterCatalogs.ZINC,
        }

        params = FilterCatalogParams()
        params.AddCatalog(catalog_map[catalog])
        self.catalog = FilterCatalog(params)

    def filter(self, smiles_list: list[str]) -> list[bool]:
        """Return a list of booleans indicating whether each SMILES passes all filters.

        For each SMILES s:
          - Parse to RDKit Mol; invalid SMILES -> False
          - Apply all requested property rules (lipinski / veber / ro5)
          - Apply all structural alert catalogs (PAINS/BRENK/NIH/ZINC)
          - Molecule passes (True) only if it satisfies *all* configured rules
        """
        results: list[bool] = []

        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                results.append(False)
                continue

            # 1) Property rules
            if not self._passes_property_rules(mol):
                results.append(False)
                continue

            # 2) Structural alert catalogs: if it matches, it contains undesirable property -> False (reject)
            if self._has_catalog_match(mol):
                results.append(False)
                continue

            results.append(True)

        return results

    def _passes_property_rules(self, mol: Chem.Mol) -> bool:
        """Check all configured property rules on a single molecule, following the logic in RxnFlow VinaTask.constraint """
        if not self.property_rule:
            # If no property rules configured, treat as pass.
            return True
        
        if self.property_rule in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(mol) > 500:
                return False
            if rdMolDescriptors.CalcNumHBD(mol) > 5:
                return False
            if rdMolDescriptors.CalcNumHBA(mol) > 10:
                return False
            if Crippen.MolLogP(mol) > 5:
                return False
            if self.property_rule == "veber":
                if rdMolDescriptors.CalcTPSA(mol) > 140:
                    return False
                if rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
                    return False
        else:
            raise ValueError(self._property_rules)
        return True

    def _has_catalog_match(self, mol: Chem.Mol) -> bool:
        """Return True if the molecule matches any structural alert catalog."""
        if not self.catalog:
            return False

        mol_with_H = Chem.AddHs(mol)
        
        return self.catalog.HasMatch(mol_with_H)


class SynthesizabilityEvaluator:
    def __init__(self, num_workers: int = 4, invalid: float = 0.0, max_size: int = 50_000, use_retrosynthesis: bool = False, sa_threshold: float = 3.0, env: str = 'stock', max_steps: int = 2):
        if use_retrosynthesis:
            env_dir = Path('../../data/envs/' + env)
            reaction_template_path = env_dir / "template.txt"
            building_block_path = env_dir / "building_block.smi"
            pre_computed_building_block_mask_path = env_dir / "bb_mask.npy"
            pre_computed_building_block_fp_path = env_dir / "bb_fp_2_1024.npy"
            pre_computed_building_block_desc_path = env_dir / "bb_desc.npy"

            # set protocol
            protocols: list[Protocol] = []
            protocols.append(Protocol("stop", RxnActionType.Stop))
            protocols.append(Protocol("firstblock", RxnActionType.FirstBlock))
            with reaction_template_path.open() as file:
                reaction_templates = [ln.strip() for ln in file.readlines()]
            for i, template in enumerate(reaction_templates):
                _rxn = Reaction(template)
                if _rxn.num_reactants == 1:
                    rxn = UniReaction(template)
                    protocols.append(Protocol(f"unirxn{i}", RxnActionType.UniRxn, _rxn))
                elif _rxn.num_reactants == 2:
                    for block_is_first in [True, False]:  # this order is important
                        rxn = BiReaction(template, block_is_first)
                        protocols.append(Protocol(f"birxn{i}_{block_is_first}", RxnActionType.BiRxn, rxn))
            protocol_dict: dict[str, Protocol] = {protocol.name: protocol for protocol in protocols}
            stop_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.Stop]
            firstblock_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.FirstBlock]
            unirxn_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.UniRxn]
            birxn_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.BiRxn]

            # set building blocks
            with building_block_path.open() as file:
                lines = file.readlines()
                building_blocks = [ln.split()[0] for ln in lines]
                building_block_ids = [ln.strip().split()[1] for ln in lines]
            blocks: list[str] = building_blocks

            # set precomputed building block feature
            block_fp = np.load(pre_computed_building_block_fp_path)
            block_prop = np.load(pre_computed_building_block_desc_path)

            # set block mask
            block_mask: NDArray[np.bool_] = np.load(pre_computed_building_block_mask_path)
            birxn_block_indices: dict[str, np.ndarray] = {}
            for i, protocol in enumerate(birxn_list):
                birxn_block_indices[protocol.name] = np.where(block_mask[i])[0]
            num_total_actions = (
                1 + len(unirxn_list) + sum(indices.shape[0] for indices in birxn_block_indices.values())
            )

            self.retrosynthesis_analyzer = MultiRetroSyntheticAnalyzer.create(protocols, blocks, num_workers=num_workers)
        else:
            self.retrosynthesis_analyzer = None
            
        self.sa_threshold = sa_threshold
        self.invalid = invalid
        self._seen  = {}  # cache to avoid recomputing (using canonical SMILES)
        self._max   = max_size
        self._max_steps = max_steps

    def get_synthesis(self, smiles: str) -> RetroSynthesisTree | None:

        if self.retrosynthesis_analyzer:
            try:
                # Canonicalize the SMILES string before scoring
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            except:
                return None
                
            if canonical_smiles in self._seen:
                return self._seen[canonical_smiles]
            self.retrosynthesis_analyzer.submit(0, smiles, self._max_steps, [])
            _, retro_tree = self.retrosynthesis_analyzer.result()[0]
            if len(self._seen) < self._max:   # cheap cap to avoid runaway RAM
                self._seen[canonical_smiles] = retro_tree
            return retro_tree
        else:
            return None

    def score(self, smiles: str) -> float:
        try:
            # Canonicalize the SMILES string before scoring
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return 0.0

        if canonical_smiles in self._seen:
            if self.retrosynthesis_analyzer:
                retro_tree = self._seen[canonical_smiles]
                return 1.0 if retro_tree else 0.0
            else:
                return float(self._seen[canonical_smiles] < self.sa_threshold)

        if self.retrosynthesis_analyzer:
            self.retrosynthesis_analyzer.submit(0, canonical_smiles, self._max_steps, [])
            _, retro_tree = self.retrosynthesis_analyzer.result()[0]
            score = 1.0 if retro_tree else 0.0
        else:
            try:
                sa = sascore.calculateScore(mol)  # sometimes, it raises an error: devided by zero (number of fingerprints is zero)
            except:
                sa = 10.0  #self.invalid
            score = float(sa < self.sa_threshold)

        if len(self._seen) < self._max:   # cheap cap to avoid runaway RAM
            self._seen[canonical_smiles] = retro_tree if self.retrosynthesis_analyzer else sa
        return score
    
    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]


class SAEvaluator:
    """
    Stateless aside from an in-process cache.
    - call .score() on a single SMILES
    - call .score_batch() on a list[str]
    """
    def __init__(self, threshold: float | None = None, max_size: int = 50_000):
        self.thresh = threshold
        self.invalid = 10.0
        self._seen  = {}
        self._max   = max_size

    def score(self, smiles: str) -> float:
        try:
            # Canonicalize the SMILES string before scoring
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return self.invalid

        if smiles in self._seen:
            return self._seen[smiles]

        try:
            sa = sascore.calculateScore(mol)
        except:
            return self.invalid
        
        if len(self._seen) < self._max:   # cheap cap to avoid runaway RAM
            self._seen[smiles] = sa
        return sa

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]



class SynthSmilesTrainer():
    def __init__(self, logger, configs):
        self.oracle = configs.oracle.strip()
        self.num_metric = 3 if self.oracle == "vina" else 1

        pdb_path = f'../../data/LIT-PCBA/{configs.vina_receptor}/protein.pdb'
        center = None
        ref_ligand_path = f'../../data/LIT-PCBA/{configs.vina_receptor}/ligand.mol2'
        self.vina = VinaReward(protein_pdb_path=pdb_path, center=center, ref_ligand_path=ref_ligand_path, search_mode="balance") if configs.oracle == "vina" else None
        self.vina_receptor = configs.vina_receptor
        self.vina_hist = {} if configs.oracle == "vina" else None  # to avoid duplicate computation

        # training parameters
        self.n_steps = configs.n_steps
        self.n_warmup_steps = configs.n_warmup_steps
        self.batch_size = configs.batch_size
        self.init_z = configs.init_z
        self.learning_rate = configs.learning_rate
        self.lr_z = configs.lr_z
        self.max_norm = configs.max_norm
        self.beta = configs.beta
        self.buffer_size = configs.buffer_size
        self.sampling_temp = configs.sampling_temp
        self.eval_sampling_temp = configs.eval_sampling_temp
        self.replay_batch_size = configs.replay_batch_size
        self.eval_every = configs.eval_every

        # constraints
        self.chemical_filter = ChemicalFilter(catalog=configs.catalog, property_rule=configs.property_rule) if configs.property_rule != "none" else None
        
        # logger
        self.wandb = configs.wandb
        self.run_name = configs.run_name + f"-seed{configs.seed}"

        # seed
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)

        # device
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(configs.seed)
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.prior = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(self.device)
        self.log_z = torch.nn.Parameter(torch.tensor([self.init_z]).to(self.device))

        self.max_length = configs.max_length if hasattr(configs, "max_length") else self.model.config.max_position_embeddings

        self.sa_evaluator = SAEvaluator()
        self.sa_threshold = configs.sa_threshold
        self.filter_unsynthesizable = configs.filter_unsynthesizable
        self.synthesizability_evaluator = SynthesizabilityEvaluator(use_retrosynthesis=configs.use_retrosynthesis, env=configs.retro_env, max_steps=configs.max_retro_steps)

        self.reshape_reward = configs.reshape_reward

        self.aux_loss = configs.aux_loss
        self.neg_coefficient = configs.neg_coefficient
        self.without_mutation = configs.without_mutation
        self.store_mutated_samples = configs.store_mutated_samples
        
    def train(self):
        oracle = f"{self.oracle}-{self.vina_receptor}" if self.oracle == "vina" else self.oracle
        if not os.path.exists(f'outputs/{oracle}'):
            os.makedirs(f'outputs/{oracle}')

        model_config = self.model.config

        self.replay = ReplayBuffer(eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 1,
                                   pad_token_id = self.tokenizer.pad_token_id,
                                   max_size=self.buffer_size,
                                   evict_by='reward'
                                   )
        
        self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 
                                                 'lr': self.learning_rate},
                                            {'params': self.log_z, 
                                                 'lr': self.lr_z}])

        if self.n_warmup_steps > 0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_steps
            )

        self.negative_replay = ReplayBuffer(eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 1,
                                pad_token_id = self.tokenizer.pad_token_id,
                                max_size=self.buffer_size,
                                evict_by='oldest'
                                )


        print(f"Starting training (oracle: {self.oracle})")
        self.model.train()

        # training loop
        for step in range(self.n_steps):

            tot_loss = 0.0
            tb_loss = 0.0
            tot_aux_loss = 0.0

            # sample new sequences
            with torch.no_grad():
                seqs = self.model.generate(
                    # input_ids,
                    do_sample=True,
                    max_length=self.max_length,
                    num_return_sequences=self.batch_size,
                    temperature=self.sampling_temp,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # evaluate smiles
            smis = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
            all_scores = torch.tensor(get_scores(smis, mode=self.oracle, vina=self.vina, hist=self.vina_hist)).reshape(-1, self.num_metric).to(self.device)
            reward = all_scores[:, 0]
            if self.vina:
                for s, v, q in zip(smis, all_scores[:, 1], all_scores[:, 2]):
                    try:
                        canonical_s = Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
                        self.vina_hist[canonical_s] = {'vina': v.item(), 'qed': q.item()}
                    except:
                        pass


            valid_indices, valid_smiles = [], []
            for i, s in enumerate(smis):
                try: 
                    mol = Chem.MolFromSmiles(s)
                    if mol:
                        valid_indices.append(i)
                        valid_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=False))  # canonicalize smiles 
                except:
                    pass
            valid_reward = reward[valid_indices]
            avg_valid_reward = valid_reward.mean().item()
            unique_onpolicy_smiles = len(set(valid_smiles))
            
            if len(valid_indices) == 0:
                continue  # skip this step if no nonzero reward samples

            encoded = self.tokenizer.batch_encode_plus(valid_smiles, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
            valid_seqs = encoded["input_ids"].to(self.device)

            sa_scores = torch.tensor(self.sa_evaluator.score_batch(valid_smiles)).to(self.device)
            synthesizability = torch.tensor(self.synthesizability_evaluator.score_batch(valid_smiles)).to(self.device)
            
            after_filtering = torch.tensor(self.chemical_filter.filter(valid_smiles)).to(self.device) if self.chemical_filter else torch.ones(len(valid_smiles)).to(self.device)
            positive = synthesizability * after_filtering.float()

            if self.reshape_reward:
                negative_indices = (positive == 0.0).nonzero(as_tuple=True)[0]
                seqs_negative = valid_seqs[negative_indices]
                smis_negative = [valid_smiles[i] for i in negative_indices.tolist()]
                reward_negative = valid_reward[negative_indices]
                if self.aux_loss != "none":
                    self.negative_replay.add_batch(seqs_negative, smis_negative, reward_negative, synthesizability[negative_indices].tolist(), masks=None, use_reshaped_reward=self.reshape_reward)

            if self.filter_unsynthesizable:
                if self.aux_loss != "none":
                    negative_indices = (positive == 0.0).nonzero(as_tuple=True)[0]
                    seqs_negative = valid_seqs[negative_indices]
                    smis_negative = [valid_smiles[i] for i in negative_indices.tolist()]
                    reward_negative = valid_reward[negative_indices]
                    self.negative_replay.add_batch(seqs_negative, smis_negative, reward_negative, synthesizability[negative_indices].tolist(), masks=None, use_reshaped_reward=self.reshape_reward)

                valid_reward = valid_reward[positive.bool()]
                valid_seqs = valid_seqs[positive.bool()]
                valid_smiles = [smis for flag, smis in zip(positive, valid_smiles) if flag]

                if self.store_mutated_samples:
                    mutated_neg_smiles, mutated_seqs, mutated_neg_reward, mutated_mask = [], [], [], []
                    for s, r in zip(valid_smiles, valid_reward):
                        mutated = mutate(s, self.synthesizability_evaluator)
                        if mutated:
                            try:
                                mutated_info = diff_mask_molformer(s, mutated, self.tokenizer)
                            except:
                                continue
                            mutated_neg_smiles.append(mutated)
                            mutated_neg_reward.append(r)  # not used anyway
                            mutated_seqs.append(torch.tensor(mutated_info['input_ids']))
                            mutated_mask.append(torch.tensor(mutated_info['mask']))
                    # mutated_neg_seqs = self.tokenizer.batch_encode_plus(mutated_neg_smiles, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')["input_ids"].to(self.device)
                    if len(mutated_seqs) > 0:
                        mutated_neg_seqs = pad_sequence(mutated_seqs, batch_first=True, padding_value=2).to(self.device)
                        mutated_neg_reward = torch.tensor(mutated_neg_reward).to(self.device)
                        self.negative_replay.add_batch(mutated_neg_seqs.to('cpu'), mutated_neg_smiles, mutated_neg_reward.to('cpu'), [0] * len(mutated_seqs), masks=mutated_mask)

            self.replay.add_batch(valid_seqs, valid_smiles, valid_reward, synthesizability, None, use_reshaped_reward=self.reshape_reward)

            self.model.train()
            ####### on-policy training with valid samples #######
            outputs = self.model(
                input_ids=valid_seqs[:, :-1],
                attention_mask=(valid_seqs[:, :-1] != self.tokenizer.pad_token_id).long(),
                labels=valid_seqs[:, 1:],
            )

            # Fix shape mismatch for torch.gather by aligning shift_logits and shift_labels
            shift_labels = valid_seqs[:, 1:]
            logits = outputs.logits  # (batch, seq_len, vocab)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            seq_token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            seq_token_logprobs = seq_token_logprobs * (shift_labels != self.tokenizer.pad_token_id)
            seq_logprobs = seq_token_logprobs.sum(dim=1)

            # prior likelihood
            with torch.no_grad():
                prior_logits = self.prior(input_ids=valid_seqs[:, :-1], attention_mask=(valid_seqs[:, :-1] != self.tokenizer.pad_token_id).long(), labels=valid_seqs[:, 1:]).logits
                prior_log_probs = torch.nn.functional.log_softmax(prior_logits, dim=-1)

                prior_seq_token_logprobs = torch.gather(prior_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                prior_seq_token_logprobs = prior_seq_token_logprobs * (shift_labels != self.tokenizer.pad_token_id)
                prior_seq_logprobs = prior_seq_token_logprobs.sum(dim=1).detach()


            if self.reshape_reward:
                synth = torch.tensor(self.synthesizability_evaluator.score_batch(valid_smiles)).to(self.device)  # won't be slow (chached)
                valid_reward = valid_reward * synth

            forward_flow = seq_logprobs + self.log_z
            backward_flow = prior_seq_logprobs + self.beta * valid_reward
            loss = torch.pow(forward_flow - backward_flow, 2).mean()
            online_tb_loss = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            # Buffer statistics
            sorted_rb = sorted(self.replay.heap, key=lambda t: t.reward * t.synthesizability.item(), reverse=True)
            top10 = sorted_rb[:10]
            top10_reward = sum(t.reward for t in top10) / 10
            top10_sa = sum(self.sa_evaluator.score(t.smiles) for t in top10) / 10
            top10_synthesizability = sum(t.synthesizability.item() for t in top10) / 10
            try: 
                top10_diversity = calculate_molecular_diversity([Chem.MolFromSmiles(t.smiles) for t in top10])[0]
            except:
                top10_diversity = -1.0

            if len(self.replay.heap) >= 100:
                top100 = sorted_rb[:100]
                top100_reward = sum(t.reward for t in top100) / 100
                top100_sa = sum(self.sa_evaluator.score(t.smiles) for t in top100) / 100
                top100_synthesizability = sum(t.synthesizability.item() for t in top100) / 100
                try: 
                    top100_diversity = calculate_molecular_diversity([Chem.MolFromSmiles(t.smiles) for t in top100])[0]
                except:
                    top100_diversity = -1.0
            else:
                top100_reward = -1.0
                top100_sa = -1.0
                top100_diversity = -1.0
                top100_synthesizability = -1.0

            log_dict = {
                "sampled_max_reward": valid_reward.max().item() if valid_reward.numel() > 0 else 0.0,
                "sampled_avg_reward": avg_valid_reward,
                "sampled_filtered_avg_reward": valid_reward.mean().item() if valid_reward.numel() > 0 else 0.0,
                "sampled_unique_onpolicy_smiles": unique_onpolicy_smiles,
                "sampled_avg_sa": sa_scores.mean().item(),
                "sampled_synth_ratio": synthesizability.mean().item(),
                "sampled_filter_ratio": (after_filtering).float().mean().item(),
                "sampled_max_length": (seqs == 1).nonzero()[:, 1].max().item(),
                "num_onpolicy_samples": len(valid_smiles),
                "buffer_top10_avg_reward": top10_reward,
                "buffer_top10_avg_sa": top10_sa,
                "buffer_top10_diversity": top10_diversity,
                "buffer_top10_synthesizability": top10_synthesizability,
                "buffer_top100_avg_reward": top100_reward,
                "buffer_top100_avg_sa": top100_sa,
                "buffer_top100_diversity": top100_diversity,
                "buffer_top100_synthesizability": top100_synthesizability,
                "buffer_size": len(self.replay.heap),
                "neg_replay_size": len(self.negative_replay.heap) if self.negative_replay else 0,
            }

            if self.oracle == "vina":
                log_dict["sampled_avg_vina"] = float(all_scores[:, 1].mean().item())
                log_dict["sampled_avg_qed"] = float(all_scores[:, 2].mean().item())

            ######## Replay training #######
            replay_tb_loss, replay_aux_loss = 0.0, 0.0
            if len(self.replay.heap) >= self.replay_batch_size:
                buf_inputs, buf_reward = self.replay.sample(self.replay_batch_size, self.device, reward_prioritized=True, replace=True)
                buf_seqs = buf_inputs["input_ids"]
                buf_smis = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in buf_seqs]

                outputs = self.model(
                    input_ids=buf_seqs[:, :-1],
                    attention_mask=(buf_seqs[:, :-1] != self.tokenizer.pad_token_id).long(),
                    labels=buf_seqs[:, 1:],
                )

                # Fix shape mismatch for torch.gather by aligning shift_logits and shift_labels
                shift_labels = buf_seqs[:, 1:]
                logits = outputs.logits  # (batch, seq_len, vocab)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                seq_token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                seq_token_logprobs = seq_token_logprobs * (shift_labels != self.tokenizer.pad_token_id)
                seq_logprobs = seq_token_logprobs.sum(dim=1)

                # for logging
                avg_pos_logq = seq_logprobs.mean().item()

                # prior likelihood
                with torch.no_grad():
                    prior_logits = self.prior(input_ids=buf_seqs[:, :-1], attention_mask=(buf_seqs[:, :-1] != self.tokenizer.pad_token_id).long(), labels=buf_seqs[:, 1:]).logits
                    prior_log_probs = torch.nn.functional.log_softmax(prior_logits, dim=-1)

                    prior_seq_token_logprobs = torch.gather(prior_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    prior_seq_token_logprobs = prior_seq_token_logprobs * (shift_labels != self.tokenizer.pad_token_id)
                    prior_seq_logprobs = prior_seq_token_logprobs.sum(dim=1).detach()

                if self.reshape_reward:
                    buf_synth = torch.tensor(self.synthesizability_evaluator.score_batch(buf_smis)).to(self.device)  # won't be slow (chached)
                    buf_reward = buf_reward * buf_synth

                forward_flow = seq_logprobs + self.log_z
                backward_flow = prior_seq_logprobs + self.beta * buf_reward
                loss = torch.pow(forward_flow - backward_flow, 2).mean()
                replay_tb_loss = loss.item()

                if self.aux_loss != "none" and len(self.negative_replay.heap) >= self.replay_batch_size:
                    
                    neg_inputs, _ = self.negative_replay.sample(self.replay_batch_size, self.device)
                    neg_seqs = neg_inputs["input_ids"]

                    neg_outputs = self.model(
                        input_ids=neg_seqs[:, :-1],
                        attention_mask=(neg_seqs[:, :-1] != self.tokenizer.pad_token_id).long(),
                        labels=neg_seqs[:, 1:],
                    )

                    # Fix shape mismatch for torch.gather by aligning shift_logits and shift_labels
                    shift_labels = neg_seqs[:, 1:]
                    logits = neg_outputs.logits  # (batch, seq_len, vocab)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    seq_token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    seq_token_logprobs = seq_token_logprobs * (shift_labels != self.tokenizer.pad_token_id)

                    neg_seq_logprobs = seq_token_logprobs.sum(dim=1)
                    avg_neg_logp = neg_seq_logprobs.mean().item()

                    if self.aux_loss == "relative_logp":
                        pos_seq_logprobs = seq_logprobs

                        neg_log_sum = torch.logsumexp(neg_seq_logprobs, dim=0) - math.log(max(neg_seq_logprobs.numel(), 1.0))
                        aux_loss = -(pos_seq_logprobs - torch.logaddexp(pos_seq_logprobs, neg_log_sum)).mean()

                        if not self.without_mutation:
                            mutated_neg_smiles = []
                            paired = []
                            for s in buf_smis:
                                mutated = mutate(s, self.synthesizability_evaluator, n_try=1)
                                if mutated:
                                    paired.append(True)
                                    mutated_neg_smiles.append(mutated)
                                else:
                                    paired.append(False)
                            if len(mutated_neg_smiles) > 0:
                                mutated_encoded = self.tokenizer.batch_encode_plus(mutated_neg_smiles, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
                                mutated_neg_seqs = mutated_encoded["input_ids"].to(self.device)
                            
                                paired = torch.tensor(paired).to(self.device)
                                mutated_outputs = self.model(
                                    input_ids=mutated_neg_seqs[:, :-1],
                                    attention_mask=(mutated_neg_seqs[:, :-1] != self.tokenizer.pad_token_id).long(),
                                    labels=mutated_neg_seqs[:, 1:],
                                )

                                # Fix shape mismatch for torch.gather by aligning shift_logits and shift_labels
                                mutated_shift_labels = mutated_neg_seqs[:, 1:]
                                mutated_logits = mutated_outputs.logits  # (batch, seq_len, vocab)
                                mutated_log_probs = torch.nn.functional.log_softmax(mutated_logits, dim=-1)
                                mutated_seq_token_logprobs = torch.gather(mutated_log_probs, 2, mutated_shift_labels.unsqueeze(-1)).squeeze(-1)
                                mutated_seq_token_logprobs = mutated_seq_token_logprobs * (mutated_shift_labels != self.tokenizer.pad_token_id)
                                mutated_seq_logprobs = mutated_seq_token_logprobs.sum(dim=1)

                                mutated_log_sum = torch.logsumexp(mutated_seq_logprobs, dim=0) - math.log(max(mutated_seq_logprobs.numel(), 1.0))
                                aux_loss += -(pos_seq_logprobs[paired] - torch.logaddexp(pos_seq_logprobs[paired], mutated_log_sum)).mean()

                    else:
                        raise ValueError(f"Invalid auxiliary loss: {self.aux_loss}")

                    replay_aux_loss = aux_loss.item()

                    loss = loss + (self.neg_coefficient * aux_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
            else:
                neg_seq_logprobs = torch.tensor(0.0)

            self.scheduler.step()

            log_dict["loss"] = loss.item()
            log_dict["online_tb_loss"] = online_tb_loss
            log_dict["tb_loss"] = replay_tb_loss
            log_dict["log_z"] = self.log_z.item()
            try:
                log_dict["avg_pos_lop"] = seq_logprobs.mean().item()
                log_dict["avg_neg_lop"] = neg_seq_logprobs.mean().item()
            except:
                pass
            log_dict["lr"] = self.scheduler.get_last_lr()[0] if self.n_warmup_steps > 0 else self.learning_rate
            log_dict["lr_logz"] = self.scheduler.get_last_lr()[1] if self.n_warmup_steps > 0 else self.lr_z
            log_dict["aux_loss"] = replay_aux_loss

            if self.wandb != 'disabled':
                wandb.log(log_dict, step=step)
            else:
                print(step, log_dict)

            if step % args.eval_every == 0:
                self.evaluate(num_samples=args.eval_samples, step=step)
                # Save the model and optimizer states
                save_dir = f"outputs/{self.oracle}-{self.vina_receptor}" if self.oracle == "vina" else f"outputs/{self.oracle}"
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"{self.run_name}_model.pt")
                optimizer_path = os.path.join(save_dir, f"{self.run_name}_optimizer.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'log_z': self.log_z,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                    'step': step,
                }, model_path)
                if hasattr(self, "neg_optimizer") and self.neg_optimizer is not None:
                    neg_optimizer_path = os.path.join(save_dir, f"{self.run_name}_neg_optimizer.pt")
                    torch.save(self.neg_optimizer.state_dict(), neg_optimizer_path)
                    
        if self.vina:
            try:
                self.evaluate(num_samples=args.eval_samples, step=self.n_steps-1, final=True, select_diverse_topk=True)
            except:
                self.evaluate(num_samples=args.eval_samples, step=self.n_steps-1, final=True)
        else:
            self.evaluate(num_samples=args.eval_samples, step=self.n_steps-1, final=True)

        # Save the model and optimizer states
        save_dir = f"outputs/{self.oracle}-{self.vina_receptor}" if self.oracle == "vina" else f"outputs/{self.oracle}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{self.run_name}_model.pt")
        optimizer_path = os.path.join(save_dir, f"{self.run_name}_optimizer.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'log_z': self.log_z,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            'step': self.n_steps-1,
        }, model_path)
        if hasattr(self, "neg_optimizer") and self.neg_optimizer is not None:
            neg_optimizer_path = os.path.join(save_dir, f"{self.run_name}_neg_optimizer.pt")
            torch.save(self.neg_optimizer.state_dict(), neg_optimizer_path)


    def evaluate(self, num_samples: int = 1000, step: int = 0, final: bool = False, select_diverse_topk: bool = False):
        """Sample num_samples molecules and report average reward, SA score, and diversity.

        Uses the same generation settings as in train(): stochastic sampling with
        temperature = self.sampling_temp and max_length = self.max_length.
        """
        self.model.eval()
        oracle = f"{self.oracle}-{self.vina_receptor}" if self.oracle == "vina" else self.oracle

        samples: list[str] = []

        if select_diverse_topk:
            remaining = 64000  #self.batch_size * 1000
        elif final:
            remaining = 6400 # self.batch_size * 100
        else:
            remaining = num_samples

        # Use a reasonable per-batch sample size
        per_batch = 64 if "+" in self.oracle else 128

        with torch.no_grad():
            while remaining > 0:
                cur = min(per_batch, remaining)
                seqs = self.model.generate(
                    do_sample=True,
                    max_length=self.max_length,
                    num_return_sequences=cur,
                    temperature=self.eval_sampling_temp,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                smis = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                samples.extend(smis)
                remaining -= cur

        if select_diverse_topk:
            all_scores = torch.tensor(get_scores(samples, mode=self.oracle, vina=self.vina, hist=self.vina_hist)).reshape(-1, self.num_metric)
            reward = all_scores[:, 0]
            synthesizability = torch.tensor(self.synthesizability_evaluator.score_batch(samples))
            synth_ratio = synthesizability.mean().item()
            valid_indices = (synthesizability == 1.0) & (all_scores[:, 2] > 0.5)
            samples = [samples[i] for i in valid_indices.nonzero().squeeze().tolist()]
            reward = reward[valid_indices]
            all_scores = all_scores[valid_indices]
            idx = compute_diverse_top_k(samples, reward, k=100)
            samples = [samples[i] for i in idx]
            reward = reward[idx]
            all_scores = all_scores[idx]
            df = pd.DataFrame(zip(samples, reward.tolist(), all_scores[:, 1].tolist(), all_scores[:, 2].tolist()), columns=["smiles", "reward", "vina", "qed"])
            df.to_csv(f"outputs/{oracle}/{self.run_name}_final_diverse_topk.csv", index=False)
            reward_computed = True
        elif final:
            df = pd.DataFrame(samples, columns=["smiles"])
            df.to_csv(f"outputs/{oracle}/{self.run_name}_final_all.csv", index=False)
            idx = torch.randperm(len(samples))[:num_samples]  # following synflownet
            samples = [samples[i] for i in idx.tolist()]
            reward_computed = False
        else:
            samples = samples[:num_samples]
            reward_computed = False

        # Compute rewards, SA, and diversity
        if not reward_computed:
            all_scores = torch.tensor(get_scores(samples, mode=self.oracle, vina=self.vina, hist=self.vina_hist)).reshape(-1, self.num_metric)
            reward = all_scores[:, 0]
            if self.vina:
                for s, v, q in zip(smis, all_scores[:, 1], all_scores[:, 2]):
                    try:
                        canonical_s = Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
                        self.vina_hist[canonical_s] = {'vina': v.item(), 'qed': q.item()}
                    except:
                        pass
            # synth_ratio = (sa_scores < self.sa_eval_threshold).float().mean().item()
            synthesizability = torch.tensor(self.synthesizability_evaluator.score_batch(samples))
            synth_ratio = synthesizability.mean().item()
        sa_scores = torch.tensor(self.sa_evaluator.score_batch(samples))

        mode = 'final' if final else 'eval'
        # save samples (smis), rewards, synthesizability to csv
        after_filtering = torch.ones(len(samples))
        df = pd.DataFrame(zip(samples, reward.tolist(), synthesizability.tolist(), after_filtering.tolist()), columns=["smiles", "reward", "synthesizability", "chemical_filter"])
        df.to_csv(f"outputs/{oracle}/{self.run_name}_{mode}.csv", index=False)

        with open(f"outputs/{oracle}/{self.run_name}_positive_replay.pkl", "wb") as f:
            pickle.dump(self.replay.heap, f)
        with open(f"outputs/{oracle}/{self.run_name}_negative_replay.pkl", "wb") as f:
            pickle.dump(self.negative_replay.heap, f)

        molecules = []
        unique_indices, unique_smiles, unique_scores, unique_molecules, unique_retrosynthesis = [], [], [], [], []
        for idx, smiles in enumerate(samples):
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2000)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
                if canonical_smiles not in unique_smiles:
                    unique_indices.append(idx)
                    unique_smiles.append(canonical_smiles)
                    unique_scores.append(all_scores[idx])
                    unique_molecules.append(mol)
                    unique_retrosynthesis.append(self.synthesizability_evaluator.score(smiles))
            except:
                mol = None
            if mol:
                molecules.append(mol)
        diversity, _ = calculate_molecular_diversity(molecules)

        if final:
            with open(f"outputs/{oracle}/{self.run_name}_final_unique.pkl", "wb") as f:
                pickle.dump({
                    "unique_smiles": unique_smiles,
                    "unique_scores": unique_scores,
                    "unique_retrosynthesis": unique_retrosynthesis
                }, f)

        if len(unique_smiles) >= 100:
            unique_reward = reward[unique_indices]
            top100_indices = torch.topk(unique_reward, 100).indices
            top100_reward = reward[[unique_indices[i] for i in top100_indices]].mean().item()
            top100_sa = sa_scores[[unique_indices[i] for i in top100_indices]].mean().item()
            top100_synthesizability = synthesizability[[unique_indices[i] for i in top100_indices]].mean().item()
            top100_molecules = [unique_molecules[i] for i in top100_indices]
            top100_diversity, _ = calculate_molecular_diversity(top100_molecules)

            unique_synthesizability = synthesizability[unique_indices]
            synth_top100_indices = torch.topk(unique_reward * unique_synthesizability, 100).indices
            synth_top100_reward = reward[[unique_indices[i] for i in synth_top100_indices]].mean().item()
            synth_top100_sa = sa_scores[[unique_indices[i] for i in synth_top100_indices]].mean().item()
            synth_top100_synthesizability = synthesizability[[unique_indices[i] for i in synth_top100_indices]].mean().item()
            synth_top100_molecules = [unique_molecules[i] for i in synth_top100_indices]
            synth_top100_diversity, _ = calculate_molecular_diversity(synth_top100_molecules)
            synth_top100_all_scores = all_scores[[unique_indices[i] for i in synth_top100_indices]]

        else:
            top100_reward = 0.0
            top100_sa = 0.0
            top100_synthesizability = 0.0
            top100_diversity = 0.0
            synth_top100_reward = 0.0
            synth_top100_sa = 0.0
            synth_top100_synthesizability = 0.0
            synth_top100_diversity = 0.0
            synth_top100_all_scores = torch.zeros_like(all_scores)
            
        mode = 'final' if final else 'eval'
        eval_log = {
            f"{mode}/avg_reward": float(reward.mean().item()),
            f"{mode}/avg_sa_score": float(sa_scores.mean().item()),
            f"{mode}/synth_ratio": float(synth_ratio),
            f"{mode}/diversity": float(diversity),
            f"{mode}/num_unique": len(unique_smiles),
            f"{mode}/top100_reward": top100_reward,
            f"{mode}/top100_sa": top100_sa,
            f"{mode}/top100_synthesizability": top100_synthesizability,
            f"{mode}/top100_diversity": float(top100_diversity),
            f"{mode}/synth_top100_reward": synth_top100_reward,
            f"{mode}/synth_top100_sa": synth_top100_sa,
            f"{mode}/synth_top100_synthesizability": synth_top100_synthesizability,
            f"{mode}/synth_top100_diversity": float(synth_top100_diversity),
        }

        if self.oracle == "vina":
            eval_log[f"{mode}/avg_vina"] = float(all_scores[:, 1].mean().item())
            eval_log[f"{mode}/avg_qed"] = float(all_scores[:, 2].mean().item())
            eval_log[f"{mode}/synth_top100_vina"] = float(synth_top100_all_scores[:, 1].mean().item())
            eval_log[f"{mode}/synth_top100_qed"] = float(synth_top100_all_scores[:, 2].mean().item())

        torch.cuda.empty_cache()
        self.model.train()

        if self.wandb != 'disabled':
            wandb.log(eval_log, step=step)
        else:
            print("Evaluation:", eval_log)
        return eval_log



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=str, default="SEH", choices=["QED", "SEH", "JNK3", "vina"])
    parser.add_argument("--vina_receptor", type=str, default="tmp", choices=["ADRB2", "ALDH1", "ESR_antago", "ESR_ago", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG","VDR", "TP53"])
    parser.add_argument("--max_length", type=int, default=140)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--n_warmup_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_batch_size", type=int, default=64)
    parser.add_argument("--init_z", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # 1e-4
    parser.add_argument("--lr_z", type=float, default=0.001)  # 0.1
    parser.add_argument("--max_norm", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer_size", type=int, default=6400)  # according to RxnFlow
    parser.add_argument("--sampling_temp", type=float, default=1.0)
    parser.add_argument("--eval_sampling_temp", type=float, default=1.0)

    parser.add_argument("--reshape_reward", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)

    parser.add_argument("--sa_threshold", type=float, default=4.0)
    parser.add_argument("--filter_unsynthesizable", action="store_true")
    parser.add_argument("--use_retrosynthesis", action="store_true")
    parser.add_argument("--retro_env", type=str, default="stock_hb", choices=["stock", "stock_curated", "stock_hb"])
    parser.add_argument("--max_retro_steps", type=int, default=2)

    parser.add_argument("--aux_loss", type=str, default="none", choices=["none", "relative_logp", "relative_logp_pairwise_mutated"])
    parser.add_argument("--neg_coefficient", type=float, default=0.0001)
    parser.add_argument("--without_mutation", action="store_true")
    parser.add_argument("--store_mutated_samples", action="store_true")

    parser.add_argument("--catalog", choices=["PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC"], default="")
    parser.add_argument("--property_rule", choices=["lipinski", "veber", "none"], default="none")

    args = parser.parse_args()

    if args.oracle == "vina":
        project = "synth-smiles-sbdd"
        group = args.vina_receptor
    else:
        project = "synth-smiles-seh"
        group = args.oracle

    if args.wandb == "online":

        wandb.init(project=project, name=args.run_name, config=args, group=group)
    elif args.wandb == "offline":
        wandb.init(project=project, name=args.run_name, config=args, group=group, mode="offline")
    else:
        wandb = None

    trainer = SynthSmilesTrainer(logger=wandb, configs=args)
    trainer.train()
