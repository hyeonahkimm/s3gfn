from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path


_original_warning = logging.Logger.warning


def _filter_fast_tfmr(self, msg, *args, **kwargs):
    """Swallow ONLY the MoLFormer CUDA-kernel fallback line."""
    if "Falling back to (slow) pytorch implementation" in str(msg):
        return
    _original_warning(self, msg, *args, **kwargs)


logging.Logger.warning = _filter_fast_tfmr

os.environ["TOKENIZERS_PARALLELISM"] = "false"


VINA_RECEPTORS = (
    "ADRB2",
    "ALDH1",
    "ESR_antago",
    "ESR_ago",
    "FEN1",
    "GBA",
    "IDH1",
    "KAT2A",
    "MAPK1",
    "MTORC1",
    "OPRK1",
    "PKM2",
    "PPARG",
    "VDR",
    "TP53",
)

MAX_LENGTH = 140
INITIAL_LOG_Z = 0.0
MAX_GRAD_NORM = 10.0


@dataclass(frozen=True)
class TaskSpec:
    scorer_mode: str
    receptor: str | None
    output_label: str
    num_metrics: int
    wandb_project: str
    wandb_group: str

    @property
    def is_docking(self) -> bool:
        return self.receptor is not None


def parse_task(value: str) -> TaskSpec:
    normalized = value.strip()
    if normalized.lower() == "seh":
        return TaskSpec("SEH", None, "SEH", 1, "synth-smiles-seh", "SEH")

    prefix, separator, receptor_input = normalized.partition(":")
    if prefix.lower() != "vina" or not separator or not receptor_input:
        raise argparse.ArgumentTypeError("task must be 'seh' or 'vina:<receptor>'")

    receptors_by_lowercase = {receptor.lower(): receptor for receptor in VINA_RECEPTORS}
    receptor = receptors_by_lowercase.get(receptor_input.lower())
    if receptor is None:
        allowed = ", ".join(VINA_RECEPTORS)
        raise argparse.ArgumentTypeError(f"unsupported Vina receptor '{receptor_input}'; choose from: {allowed}")

    return TaskSpec("vina", receptor, f"vina-{receptor}", 3, "synth-smiles-sbdd", receptor)


def docking_input_paths(task: TaskSpec) -> tuple[Path, Path]:
    if not task.is_docking:
        raise ValueError(f"Task '{task.output_label}' does not use docking inputs")
    receptor_dir = Path(__file__).resolve().parents[2] / "data" / "LIT-PCBA" / task.receptor
    return receptor_dir / "protein.pdb", receptor_dir / "ligand.mol2"


def validate_task_inputs(task: TaskSpec):
    if not task.is_docking:
        return
    missing_paths = [path for path in docking_input_paths(task) if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing docking input file(s) for task '{task.output_label}': {missing}")


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _load_runtime_dependencies():
    global np, pd, math, torch, Chem, Crippen, rdMolDescriptors
    global FilterCatalog, FilterCatalogParams
    global AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    global VinaReward, get_scores
    global calculate_molecular_diversity, compute_diverse_top_k, ReplayBuffer
    global sascore, SynthesizabilityEvaluator, wandb

    import math
    import numpy as np
    import pandas as pd
    import torch
    import wandb
    from gflownet.utils import sascore
    from rdkit import Chem
    from rdkit.Chem import Crippen, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rxnflow.tasks.unidock_vina import VinaReward
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    if __package__:
        from .chem_metrics import compute_diverse_top_k
        from .replay_buffer import ReplayBuffer
        from .scoring_function import get_scores
        from .synthesis_eval import calculate_molecular_diversity
        from .synthesizability import SynthesizabilityEvaluator
    else:
        from chem_metrics import compute_diverse_top_k
        from replay_buffer import ReplayBuffer
        from scoring_function import get_scores
        from synthesis_eval import calculate_molecular_diversity
        from synthesizability import SynthesizabilityEvaluator


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
        self.task = configs.task
        self.oracle = self.task.scorer_mode
        self.num_metric = self.task.num_metrics

        self.vina = None
        if self.task.is_docking:
            validate_task_inputs(self.task)
            pdb_path, ref_ligand_path = docking_input_paths(self.task)
            self.vina = VinaReward(protein_pdb_path=str(pdb_path), center=None, ref_ligand_path=str(ref_ligand_path), search_mode="balance")
        self.vina_hist = {} if self.task.is_docking else None  # to avoid duplicate computation

        # training parameters
        self.num_training_steps = configs.num_training_steps
        self.num_warmup_steps = configs.num_warmup_steps
        self.batch_size = configs.batch_size
        self.learning_rate = configs.learning_rate
        self.log_z_learning_rate = configs.log_z_learning_rate
        self.beta = configs.beta
        self.buffer_size = configs.buffer_size
        self.sampling_temperature = configs.sampling_temperature
        self.eval_sampling_temperature = configs.eval_sampling_temperature
        self.replay_batch_size = configs.replay_batch_size
        self.eval_every = configs.eval_every
        self.eval_samples = configs.eval_samples
        self.output_dir = Path(configs.output_dir).expanduser() / self.task.output_label
        self.save_periodic_every = configs.save_periodic_every

        # constraints
        self.chemical_filter = ChemicalFilter(catalog=configs.catalog, property_rule=configs.property_rule) if configs.property_rule != "none" else None
        
        # logger
        self.wandb_mode = configs.wandb_mode
        self.run_name = configs.run_name + f"-seed{configs.seed}"

        # seed
        self.seed = configs.seed
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
        self.log_z = torch.nn.Parameter(torch.tensor([INITIAL_LOG_Z]).to(self.device))

        self.max_length = MAX_LENGTH

        self.sa_evaluator = SAEvaluator()
        self.sa_threshold = configs.sa_threshold
        self.synthesizability_evaluator = SynthesizabilityEvaluator(use_retrosynthesis=configs.use_retrosynthesis, env=configs.retro_env, max_steps=configs.retro_steps)

        self.training_mode = configs.training_mode
        self.aux_coefficient = configs.aux_coefficient
        
    def train(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        model_config = self.model.config

        self.replay = ReplayBuffer(pad_token_id = self.tokenizer.pad_token_id,
                                   max_size=self.buffer_size,
                                   policy='reward',
                                   seed=self.seed,
                                   )
        
        self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), 
                                                 'lr': self.learning_rate},
                                            {'params': self.log_z, 
                                                 'lr': self.log_z_learning_rate}])

        if self.num_warmup_steps > 0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps
            )

        self.negative_replay = ReplayBuffer(pad_token_id = self.tokenizer.pad_token_id,
                                max_size=self.buffer_size,
                                policy='fifo',
                                seed=self.seed,
                                )


        print(f"Starting training (task: {self.task.output_label})")
        self.model.train()

        # training loop
        for step in range(self.num_training_steps):

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
                    temperature=self.sampling_temperature,
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
            replay_synthesizability = synthesizability
            
            after_filtering = torch.tensor(self.chemical_filter.filter(valid_smiles)).to(self.device) if self.chemical_filter else torch.ones(len(valid_smiles)).to(self.device)
            positive = synthesizability * after_filtering.float()

            if self.training_mode == "s3gfn":
                negative_indices = (positive == 0.0).nonzero(as_tuple=True)[0]
                seqs_negative = valid_seqs[negative_indices]
                smis_negative = [valid_smiles[i] for i in negative_indices.tolist()]
                reward_negative = valid_reward[negative_indices]
                self.negative_replay.add_batch(seqs_negative, smis_negative, reward_negative, synthesizability[negative_indices].tolist())

            if self.training_mode == "s3gfn":
                valid_reward = valid_reward[positive.bool()]
                valid_seqs = valid_seqs[positive.bool()]
                valid_smiles = [smis for flag, smis in zip(positive, valid_smiles) if flag]
                replay_synthesizability = synthesizability[positive.bool()]

            if self.training_mode == "reward_shaping":
                valid_reward = valid_reward * positive

            self.replay.add_batch(valid_seqs, valid_smiles, valid_reward, replay_synthesizability)

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


            forward_flow = seq_logprobs + self.log_z
            backward_flow = prior_seq_logprobs + self.beta * valid_reward
            loss = torch.pow(forward_flow - backward_flow, 2).mean()
            online_tb_loss = loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
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
                "training_mode": self.training_mode,
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

            if self.task.is_docking:
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

                forward_flow = seq_logprobs + self.log_z
                backward_flow = prior_seq_logprobs + self.beta * buf_reward
                loss = torch.pow(forward_flow - backward_flow, 2).mean()
                replay_tb_loss = loss.item()

                if self.training_mode == "s3gfn" and len(self.negative_replay.heap) >= self.replay_batch_size:
                    
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

                    pos_seq_logprobs = seq_logprobs
                    neg_log_sum = torch.logsumexp(neg_seq_logprobs, dim=0) - math.log(max(neg_seq_logprobs.numel(), 1.0))
                    aux_loss = -(pos_seq_logprobs - torch.logaddexp(pos_seq_logprobs, neg_log_sum)).mean()

                    replay_aux_loss = aux_loss.item()

                    loss = loss + (self.aux_coefficient * aux_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
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
            log_dict["lr"] = self.scheduler.get_last_lr()[0] if self.num_warmup_steps > 0 else self.learning_rate
            log_dict["lr_logz"] = self.scheduler.get_last_lr()[1] if self.num_warmup_steps > 0 else self.log_z_learning_rate
            log_dict["aux_loss"] = replay_aux_loss

            if self.wandb_mode != 'disabled':
                wandb.log(log_dict, step=step)
            else:
                print(step, log_dict)

            if step % self.eval_every == 0:
                self.evaluate(num_samples=self.eval_samples, step=step)
            if self.save_periodic_every > 0 and (step + 1) % self.save_periodic_every == 0:
                self._save_checkpoint(step, periodic=True)
                    
        if self.vina:
            try:
                self.evaluate(num_samples=self.eval_samples, step=self.num_training_steps-1, final=True, select_diverse_topk=True)
            except:
                self.evaluate(num_samples=self.eval_samples, step=self.num_training_steps-1, final=True)
        else:
            self.evaluate(num_samples=self.eval_samples, step=self.num_training_steps-1, final=True)
        self._save_checkpoint(self.num_training_steps - 1)

    def _checkpoint_path(self, step: int | None = None) -> Path:
        suffix = f"_step{step + 1}_model.pt" if step is not None else "_model.pt"
        return self.output_dir / f"{self.run_name}{suffix}"

    def _save_checkpoint(self, step: int, periodic: bool = False):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._checkpoint_path(step if periodic else None)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'log_z': self.log_z,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
            'step': step,
        }, model_path)


    def evaluate(self, num_samples: int = 1000, step: int = 0, final: bool = False, select_diverse_topk: bool = False):
        """Sample num_samples molecules and report average reward, SA score, and diversity.

        Uses the same generation settings as in train(): stochastic sampling with
        temperature = self.sampling_temperature and max_length = self.max_length.
        """
        self.model.eval()
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
                    temperature=self.eval_sampling_temperature,
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
            df.to_csv(self.output_dir / f"{self.run_name}_final_diverse_topk.csv", index=False)
            reward_computed = True
        elif final:
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
        after_filtering = torch.ones(len(samples))
        df = pd.DataFrame(zip(samples, reward.tolist(), synthesizability.tolist(), after_filtering.tolist()), columns=["smiles", "reward", "synthesizability", "chemical_filter"])
        if final:
            df.to_csv(self.output_dir / f"{self.run_name}_final.csv", index=False)

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

        if self.task.is_docking:
            eval_log[f"{mode}/avg_vina"] = float(all_scores[:, 1].mean().item())
            eval_log[f"{mode}/avg_qed"] = float(all_scores[:, 2].mean().item())
            eval_log[f"{mode}/synth_top100_vina"] = float(synth_top100_all_scores[:, 1].mean().item())
            eval_log[f"{mode}/synth_top100_qed"] = float(synth_top100_all_scores[:, 2].mean().item())

        torch.cuda.empty_cache()
        self.model.train()

        if self.wandb_mode != 'disabled':
            wandb.log(eval_log, step=step)
        else:
            print("Evaluation:", eval_log)
        return eval_log



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--task", type=parse_task, default="seh", metavar="seh|vina:<receptor>", help="optimization task")
    parser.add_argument("--num_training_steps", type=int, default=5000, help="number of training iterations")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="learning-rate warmup iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="on-policy samples per iteration")
    parser.add_argument("--replay_batch_size", type=int, default=64, help="positive and negative replay batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="model learning rate")
    parser.add_argument("--log_z_learning_rate", type=float, default=0.001, help="log-Z learning rate")
    parser.add_argument("--beta", type=float, default=50.0, help="reward coefficient in the RTB objective")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled", help="Weights & Biases logging mode")
    parser.add_argument("--run_name", type=str, default="default", help="run-name prefix")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--buffer_size", type=int, default=6400, help="maximum size of each replay buffer")
    parser.add_argument("--sampling_temperature", type=float, default=1.0, help="training sampling temperature")
    parser.add_argument("--eval_sampling_temperature", type=float, default=1.0, help="evaluation sampling temperature")

    parser.add_argument("--training_mode", choices=["s3gfn", "reward_shaping", "rtb"], default="s3gfn", help="training objective")
    parser.add_argument("--eval_samples", type=int, default=1000, help="samples used for periodic evaluation")
    parser.add_argument("--eval_every", type=int, default=100, help="periodic evaluation interval")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output root directory")
    parser.add_argument("--save_periodic_every", type=nonnegative_int, default=0, help="periodic checkpoint interval; 0 disables periodic checkpoints")

    parser.add_argument("--sa_threshold", type=float, default=4.0, help="synthetic-accessibility threshold")
    parser.add_argument("--use_retrosynthesis", action="store_true", help="check synthesizability with retrosynthesis")
    parser.add_argument("--retro_env", type=str, default="stock_hb", choices=["stock", "stock_curated", "stock_hb"], help="retrosynthesis environment")
    parser.add_argument("--retro_steps", type=int, default=2, help="maximum retrosynthesis steps")

    parser.add_argument("--aux_coefficient", type=float, default=0.0001, help="contrastive auxiliary-loss coefficient")

    parser.add_argument("--catalog", choices=["PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC"], default="", help="optional chemical-filter catalog")
    parser.add_argument("--property_rule", choices=["lipinski", "veber", "none"], default="none", help="optional molecular-property rule")

    args = parser.parse_args()
    validate_task_inputs(args.task)
    _load_runtime_dependencies()

    project = args.task.wandb_project
    group = args.task.wandb_group

    if args.wandb_mode == "online":

        wandb.init(project=project, name=args.run_name, config=args, group=group)
    elif args.wandb_mode == "offline":
        wandb.init(project=project, name=args.run_name, config=args, group=group, mode="offline")
    else:
        wandb = None

    trainer = SynthSmilesTrainer(logger=wandb, configs=args)
    trainer.train()
