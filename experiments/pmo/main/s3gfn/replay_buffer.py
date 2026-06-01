from dataclasses import dataclass, field
import heapq, random
from typing import Optional, List, Literal
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# ---------- RDKit helpers ----------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def ecfp4(mol: Chem.Mol, nBits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)


def tanimoto_bulk(fp, fps: List):
    return DataStructs.BulkTanimotoSimilarity(fp, fps) if fps else []


# ---------- Buffer ----------
@dataclass(order=True)
class Trajectory:
    sort_idx: float = field(init=False, repr=False)
    reward: float = field(compare=False)
    synthesizability: float = field(compare=False)
    smiles: str = field(compare=False)
    ids: torch.Tensor = field(compare=False)
    insertion_order: int = field(compare=False, default=0)
    fp: object = field(compare=False, repr=False, default=None)      # RDKit ExplicitBitVect

    def __post_init__(self):
        self.sort_idx = self.reward  # min-heap on reward


class ReplayBuffer:
    """
    Replay buffer with duplicate suppression and a fixed eviction policy.

    - Near-duplicate: Tanimoto >= (1 - sim)  -> treat as near-dup (replace only if higher reward)
    - Eviction when full:
        * evict_by="reward": positive replay keeps high-reward items
        * evict_by="oldest": negative replay acts as a FIFO queue
    """
    def __init__(
        self,
        eos_token_id: int,
        pad_token_id: int,
        max_size: int = 4096,
        sim: float = 0.25,
        evict_by: Literal["reward", "oldest"] = "reward",
    ):
        self.eos  = eos_token_id
        self.pad  = pad_token_id
        self.max  = max_size
        self.sim  = sim
        self.heap: List[Trajectory] = []   # min-heap by reward
        self.pool = set()                  # SMILES strings
        self.evict_by = evict_by
        self.replaced_cnt = 0
        self._next_insertion_order = 0

    # ---- similarity helpers ----
    def _best_match(self, fp_new):
        if not self.heap:
            return -1, 0.0
        sims = tanimoto_bulk(fp_new, [it.fp for it in self.heap])
        if not sims:
            return -1, 0.0
        j = int(np.argmax(sims))
        return j, float(sims[j])

    def reinitialize(self, trajectories: List[Trajectory]):
        for insertion_order, trajectory in enumerate(trajectories):
            trajectory.insertion_order = insertion_order
        self.heap = trajectories
        self.pool = set([trj.smiles for trj in trajectories])
        heapq.heapify(self.heap)
        self.replaced_cnt = 0
        self._next_insertion_order = len(trajectories)

    # ---- batch add with switchable eviction ----
    def add_batch(
        self,
        ids: torch.Tensor,
        decoded: List[str],
        rewards: torch.Tensor,
        synthesizability: torch.Tensor,
    ):
        ids, rewards = ids.cpu(), rewards.cpu()
        near_dup_sim_thresh = 1.0 - self.sim

        for i, smi in enumerate(decoded):
            if smi in self.pool:
                continue

            mol = mol_from_smiles(smi)
            if mol is None:
                continue
            fp = ecfp4(mol)
            rew_i = float(rewards[i])

            # near-duplicate search
            best_idx, best_sim = self._best_match(fp)
            near_dup = self.heap[best_idx] if (best_idx >= 0 and best_sim >= near_dup_sim_thresh) else None

            # If similar and not better -> skip
            if near_dup and near_dup.reward >= rew_i and near_dup.synthesizability >= synthesizability[i]:
                continue

            ids_i = ids[i].clone()
            ids_i = ids_i[ids_i != self.pad]

            traj = Trajectory(rew_i, synthesizability[i], smi, ids_i, self._next_insertion_order, fp=fp)
            self._next_insertion_order += 1

            # If similar but better -> replace that slot directly
            if near_dup:
                victim_idx = best_idx
                self.pool.discard(self.heap[victim_idx].smiles)
                self.heap[victim_idx] = traj
                heapq.heapify(self.heap)  # maintain min-heap by reward
                self.pool.add(smi)
                continue

            # Not similar: insert or evict depending on capacity & policy
            if len(self.heap) < self.max:
                heapq.heappush(self.heap, traj)
                self.pool.add(smi)
                continue

            # Buffer full -> decide by policy
            if self.evict_by == "reward":
                if rew_i > self.heap[0].reward:
                    worst = heapq.heapreplace(self.heap, traj)
                    self.pool.discard(worst.smiles)
                    self.pool.add(smi)
                    self.replaced_cnt += 1
            elif self.evict_by == "oldest":
                # Replace the oldest trajectory with the new one
                oldest_idx = min(range(len(self.heap)), key=lambda idx: self.heap[idx].insertion_order)
                oldest = self.heap.pop(oldest_idx)
                self.pool.discard(oldest.smiles)
                self.heap.append(traj)
                self.pool.add(smi)
                self.replaced_cnt += 1

    def sample(self, n: int, device: str, reward_prioritized: bool = False):
        n = min(n, len(self.heap))
        if reward_prioritized:
            rewards = torch.tensor([t.reward for t in self.heap], dtype=torch.float32)
            # rewards = torch.tensor([t.reward * t.synthesizability for t in self.heap], dtype=torch.float32)
            # Avoid negative or zero rewards for samplings
            min_reward = rewards.min().item()
            if min_reward <= 0:
                rewards = rewards - min_reward + 1e-6
            probs = rewards / rewards.sum()
            indices = torch.multinomial(probs, n, replacement=True).tolist()
            batch = [self.heap[i] for i in indices]
        else:
            batch = random.sample(self.heap, n)
        ids  = [t.ids for t in batch]
        ids  = pad_sequence(ids,  batch_first=True, padding_value=self.pad).to(device)
        rewards = torch.tensor([t.reward for t in batch], device=device)
        synthesizabilities = torch.tensor([t.synthesizability for t in batch], device=device)
        return {"input_ids": ids, "synthesizability": synthesizabilities, "smiles": [t.smiles for t in batch]}, rewards
