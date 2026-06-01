import logging

# save original method so we don't lose other warnings
_original_warning = logging.Logger.warning

def _filter_fast_tfmr(self, msg, *args, **kwargs):
    """Swallow ONLY the MoLFormer CUDA-kernel fallback line."""
    if "Falling back to (slow) pytorch implementation" in str(msg):
        return
    _original_warning(self, msg, *args, **kwargs)

logging.Logger.warning = _filter_fast_tfmr

import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import unique
import math
import torch
from rdkit import Chem
import wandb

from synth_utils import mutate
from replay_buffer import ReplayBuffer

from transformers import AutoModelForCausalLM, AutoTokenizer


from gflownet.utils import sascore
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.envs.reaction import BiReaction, Reaction, UniReaction
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATA_DIR = Path(__file__).resolve().parents[4] / "data"
SA_THRESHOLD = 3.0
MAX_GRAD_NORM = 10.0
REPLAY_SIZE = 1024
ONPOLICY_INTERVAL = 2
GA_RANK_COEFFICIENT = 0.01


class SynthesizabilityEvaluator:
    def __init__(self, num_workers: int = 4, invalid: float = 0.0, max_size: int = 50_000, use_retrosynthesis: bool = False, env: str = 'stock', max_steps: int = 2):
        if use_retrosynthesis:
            env_dir = DATA_DIR / "envs" / env
            reaction_template_path = env_dir / "template.txt"
            building_block_path = env_dir / "building_block.smi"
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
            with building_block_path.open() as file:
                blocks = [line.split()[0] for line in file]

            self.retrosynthesis_analyzer = MultiRetroSyntheticAnalyzer.create(protocols, blocks, num_workers=num_workers)
        else:
            self.retrosynthesis_analyzer = None
            
        self.sa_threshold = SA_THRESHOLD
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
                sa = 10.0
            score = float(sa < self.sa_threshold)

        if len(self._seen) < self._max:   # cheap cap to avoid runaway RAM
            self._seen[canonical_smiles] = retro_tree if self.retrosynthesis_analyzer else sa
        return score
    
    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]



class S3GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "s3gfn"

    def _optimize(self, oracle, config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.oracle.assign_evaluator(oracle)
        self.oracle.assign_synth_evaluator(SynthesizabilityEvaluator(use_retrosynthesis=config['use_retrosynthesis'], 
                                                                     env=config['retro_env'], 
                                                                     max_steps=config['max_retro_steps']))
        use_aux_loss = config["use_aux_loss"]

        tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
        prior = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(device)
        model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(device)
        prior.eval()
        

        log_z = torch.nn.Parameter(torch.tensor([0.], device=device))
        optimizer = torch.optim.Adam([{'params': model.parameters(), 
                                       'lr': config['learning_rate']},
                                      {'params': log_z, 
                                       'lr': config['lr_z']}])

        replay = ReplayBuffer(eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1,
                                   pad_token_id = tokenizer.pad_token_id,
                                   max_size=REPLAY_SIZE,
                                   evict_by='reward'
                                   )
        negative_replay = ReplayBuffer(eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1,
                                pad_token_id = tokenizer.pad_token_id,
                                max_size=REPLAY_SIZE,
                                evict_by='oldest'
                                )

        if config['use_ga']:
            from ga_expert import GeneticOperatorHandler
            ga_handler = GeneticOperatorHandler(
                mutation_rate=config["ga_mutation_rate"],
                population_size=config["ga_population_size"],
            )
        
        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        synth_history = []

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            if not use_aux_loss or step % ONPOLICY_INTERVAL == 0 or len(replay.heap) < config['batch_size']:
                training_mode = 'onpolicy'
                with torch.no_grad():
                    seqs = model.generate(
                        do_sample=True,
                        max_length=config['max_length'],
                        num_return_sequences=config['batch_size'],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                unique_idxs = unique(seqs)
                seqs = seqs[unique_idxs]
                smiles = tokenizer.batch_decode(seqs, skip_special_tokens=True)

                synthesizability = torch.tensor(self.oracle.synth_evaluator.score_batch(smiles)).to(device)
                synth_history.append(synthesizability.mean().item())

                if use_aux_loss:
                    positive_indices = (synthesizability == 1).nonzero(as_tuple=True)[0]
                    positive_smiles = [smiles[i] for i in positive_indices.tolist()]
                    positive_scores = torch.tensor(self.oracle(positive_smiles)).to(device)
                    replay.add_batch(seqs[positive_indices], positive_smiles, positive_scores, [1] * len(positive_smiles))
                    
                    # not to count oracle calls for negative samples
                    negative_indices = (synthesizability == 0).nonzero(as_tuple=True)[0]
                    negative_smiles = [smiles[i] for i in negative_indices.tolist()]
                    negative_scores = torch.zeros(len(negative_smiles)).to(device)
                    negative_seqs = seqs[negative_indices]

                    negative_replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles))
                    valid_seqs = seqs[positive_indices]
                    valid_smiles = positive_smiles
                    valid_scores = positive_scores
                    valid_synth = synthesizability[positive_indices]
                else:
                    scores = torch.tensor(self.oracle(smiles)).to(device)
                    replay.add_batch(seqs, smiles, scores, synthesizability.tolist())
                    valid_seqs = seqs
                    valid_smiles = smiles
                    valid_scores = scores
                    valid_synth = synthesizability

                if self.finish:
                    print('max oracle hit')
                    break
                
                if config['use_ga'] and len(self.oracle) >= config["ga_population_size"]:
                    self.oracle.sort_buffer()
                    pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                    mating_pool = (pop_smis[:REPLAY_SIZE], pop_scores[:REPLAY_SIZE])
                    for g in range(config["ga_generations"]):
                        child_smis, _, pop_smis, pop_scores = ga_handler.query(
                            query_size=config["ga_query_size"],
                            mating_pool=mating_pool,
                            pool=None,
                            rank_coefficient=GA_RANK_COEFFICIENT,
                        )
                        if not child_smis:
                            continue
                        child_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(child_smis)).to(device)
                        child_seqs = tokenizer.batch_encode_plus(child_smis, add_special_tokens=True, padding=True, max_length=config['max_length'], return_tensors='pt')["input_ids"].to(device)
                        child_scores= torch.zeros(len(child_smis)).to(device)

                        if child_synth.sum() > 0:
                            child_pos_indices = (child_synth == 1).nonzero(as_tuple=True)[0]
                            child_pos_smiles = [child_smis[i] for i in child_pos_indices.tolist()]
                            child_pos_scores = torch.tensor(self.oracle(child_pos_smiles)).to(device)
                            child_pos_seqs = child_seqs[child_pos_indices]
                            child_scores[child_pos_indices] = child_pos_scores
                            replay.add_batch(child_pos_seqs, child_pos_smiles, child_pos_scores, [1] * len(child_pos_smiles))
                        else:
                            continue

                        negative_indices = (child_synth == 0).nonzero(as_tuple=True)[0]
                        negative_smiles = [child_smis[i] for i in negative_indices.tolist()]
                        negative_seqs = child_seqs[negative_indices]
                        negative_scores = torch.zeros(len(negative_smiles)).to(device)
                        if use_aux_loss:
                            negative_replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles))

                        if child_synth.sum() > 0:
                            mating_pool = (pop_smis+child_pos_smiles, pop_scores+child_pos_scores.tolist())

            else:  # replay training
                training_mode = 'replay'
                valid_inputs, valid_scores = replay.sample(config['batch_size'], device, reward_prioritized=True)
                valid_seqs = valid_inputs["input_ids"]
                valid_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in valid_seqs]
                valid_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(valid_smiles)).to(device)  # won't be slow (cached)

                if use_aux_loss:
                    if len(negative_replay.heap) < config['batch_size']:
                        step += 1
                        continue
                    neg_inputs, negative_scores = negative_replay.sample(config['batch_size'], device)
                    negative_seqs = neg_inputs["input_ids"]
                    negative_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in negative_seqs]

            if self.finish:
                print('max oracle hit')
                break

            
            too_few_negatives = use_aux_loss and negative_seqs.shape[0] < 4
            if use_aux_loss and (valid_synth.sum() < 4 or too_few_negatives):
                step += 1
                continue
            else:
                aux_loss = torch.zeros((), device=device)

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience * ONPOLICY_INTERVAL:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0
            
            # early stopping2
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10 * ONPOLICY_INTERVAL:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)

            # onpolicy training
            model.train()

            outputs = model(
                input_ids=valid_seqs[:, :-1],
                attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                labels=valid_seqs[:, 1:],
            )

            shift_labels = valid_seqs[:, 1:]
            logits = outputs.logits  # (batch, seq_len, vocab)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            seq_token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            seq_token_logprobs = seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
            seq_logprobs = seq_token_logprobs.sum(dim=1)

            with torch.no_grad():
                prior_logits = prior(
                    input_ids=valid_seqs[:, :-1],
                    attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                    labels=valid_seqs[:, 1:],
                ).logits
                prior_log_probs = torch.nn.functional.log_softmax(prior_logits, dim=-1)
                prior_seq_token_logprobs = torch.gather(prior_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                prior_seq_token_logprobs = prior_seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
                prior_seq_logprobs = prior_seq_token_logprobs.sum(dim=1).detach()

            forward_flow = seq_logprobs + log_z
            backward_flow = prior_seq_logprobs + config['beta'] * valid_scores
            loss = torch.pow(forward_flow - backward_flow, 2).mean()

            if training_mode == 'onpolicy':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                
            if use_aux_loss and len(valid_smiles) > 0 and training_mode == 'replay':
                if not config['with_mutation']:
                    aux_loss = torch.zeros((), device=device)
                    pos_seq_logprobs = seq_logprobs
                else:
                    mutated_neg_smiles = []
                    paired = []
                    for s, f in zip(valid_smiles, valid_synth):
                        if not f:
                            continue
                        mutated = mutate(s, self.oracle.synth_evaluator)
                        if mutated:
                            paired.append(True)
                            mutated_neg_smiles.append(mutated)
                        else:
                            paired.append(False)
                    if mutated_neg_smiles:
                        mutated_neg_seqs = tokenizer.batch_encode_plus(
                            mutated_neg_smiles,
                            add_special_tokens=True,
                            padding=True,
                            max_length=config["max_length"],
                            return_tensors="pt",
                        )["input_ids"].to(device)

                        pos_logits = model(
                            input_ids=valid_seqs[:, :-1],
                            attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                            labels=valid_seqs[:, 1:],
                        ).logits

                        pos_log_probs = torch.nn.functional.log_softmax(pos_logits, dim=-1)
                        pos_seq_token_logprobs = torch.gather(pos_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                        pos_seq_token_logprobs = pos_seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
                        pos_seq_logprobs = pos_seq_token_logprobs.sum(dim=1)
                        mut_logits = model(
                            input_ids=mutated_neg_seqs[:, :-1],
                            attention_mask=(mutated_neg_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                            labels=mutated_neg_seqs[:, 1:],
                        ).logits

                        mut_shift_labels = mutated_neg_seqs[:, 1:]
                        mut_log_probs = torch.nn.functional.log_softmax(mut_logits, dim=-1)
                        mut_seq_token_logprobs = torch.gather(mut_log_probs, 2, mut_shift_labels.unsqueeze(-1)).squeeze(-1)
                        mut_seq_token_logprobs = mut_seq_token_logprobs * (mut_shift_labels != tokenizer.pad_token_id)
                        mut_seq_logprobs = mut_seq_token_logprobs.sum(dim=1)
                        
                        paired_mask = torch.tensor(paired).to(device)
                        
                        mutated_log_sum = torch.logsumexp(mut_seq_logprobs, dim=0) - math.log(max(mut_seq_logprobs.numel(), 1.0))
                        aux_loss = -(pos_seq_logprobs[paired_mask] - torch.logaddexp(pos_seq_logprobs[paired_mask], mutated_log_sum)).mean()

                    else:
                        aux_loss = torch.zeros((), device=device)
                neg_logits = model(
                    input_ids=negative_seqs[:, :-1],
                    attention_mask=(negative_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                    labels=negative_seqs[:, 1:],
                ).logits

                neg_shift_labels = negative_seqs[:, 1:]
                neg_log_probs = torch.nn.functional.log_softmax(neg_logits, dim=-1)
                neg_seq_token_logprobs = torch.gather(neg_log_probs, 2, neg_shift_labels.unsqueeze(-1)).squeeze(-1)
                neg_seq_token_logprobs = neg_seq_token_logprobs * (neg_shift_labels != tokenizer.pad_token_id)
                neg_seq_logprobs = neg_seq_token_logprobs.sum(dim=1)

                neg_log_sum = torch.logsumexp(neg_seq_logprobs, dim=0) - math.log(max(neg_seq_logprobs.numel(), 1.0))
                aux_loss += -(pos_seq_logprobs - torch.logaddexp(pos_seq_logprobs, neg_log_sum)).mean()

                optimizer.zero_grad()
                (loss + config['aux_coefficient'] * aux_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()

            step += 1

        try:
            wandb.log({
                    'final/synth_history_last5_mean': np.mean(synth_history[-5:]),
                    'final/synth_history_mean': np.mean(synth_history),
                    })
        except:
            pass
