from dataclasses import dataclass, field
import heapq
import random
from typing import List, Literal, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def ecfp4(mol: Chem.Mol, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def tanimoto_bulk(fp, fps: List):
    return DataStructs.BulkTanimotoSimilarity(fp, fps) if fps else []


@dataclass(order=True)
class Trajectory:
    sort_idx: float = field(init=False, repr=False)
    reward: float
    synthesizability: float
    smiles: str
    ids: torch.Tensor
    fp: object = field(compare=False, repr=False, default=None)

    def __post_init__(self):
        self.sort_idx = self.reward


class ReplayBuffer:
    """Bounded replay storage for positive or negative trajectories.

    Reward replay keeps high-reward, nonredundant molecules. FIFO replay keeps
    unique molecules in insertion order and evicts the oldest when full.
    """

    def __init__(
        self,
        pad_token_id: int,
        max_size: int = 4096,
        sim: float = 0.25,
        policy: Literal["reward", "fifo"] = "reward",
        seed: int = 0,
    ):
        self.pad = pad_token_id
        self.max = max_size
        self.sim = sim
        self.heap: List[Trajectory] = []
        self.pool = set()
        self.policy = policy
        self._random = random.Random(seed)
        self._torch_generator = torch.Generator().manual_seed(seed)

    def _best_match(self, fp_new):
        if not self.heap:
            return -1, 0.0
        similarities = tanimoto_bulk(fp_new, [item.fp for item in self.heap])
        if not similarities:
            return -1, 0.0
        best_idx, best_sim = max(enumerate(similarities), key=lambda pair: pair[1])
        return best_idx, float(best_sim)

    def add_batch(
        self,
        ids: torch.Tensor,
        decoded: List[str],
        rewards: torch.Tensor,
        synthesizability: torch.Tensor,
    ):
        ids, rewards = ids.cpu(), rewards.cpu()

        for i, smi in enumerate(decoded):
            if smi in self.pool:
                continue

            reward = float(rewards[i])
            trajectory = Trajectory(reward, synthesizability[i], smi, ids[i].clone())

            if self.policy == "fifo":
                self._add_fifo(trajectory)
                continue

            mol = mol_from_smiles(smi)
            if mol is None:
                continue
            trajectory.fp = ecfp4(mol)
            self._add_reward(trajectory)

    def _add_fifo(self, trajectory: Trajectory):
        if len(self.heap) >= self.max:
            oldest = self.heap.pop(0)
            self.pool.discard(oldest.smiles)
        self.heap.append(trajectory)
        self.pool.add(trajectory.smiles)

    def _add_reward(self, trajectory: Trajectory):
        best_idx, best_sim = self._best_match(trajectory.fp)
        near_duplicate = best_idx >= 0 and best_sim >= 1.0 - self.sim

        if near_duplicate:
            if self.heap[best_idx].reward >= trajectory.reward:
                return
            self.pool.discard(self.heap[best_idx].smiles)
            self.heap[best_idx] = trajectory
            heapq.heapify(self.heap)
            self.pool.add(trajectory.smiles)
            return

        if len(self.heap) < self.max:
            heapq.heappush(self.heap, trajectory)
            self.pool.add(trajectory.smiles)
            return

        if trajectory.reward > self.heap[0].reward:
            lowest_reward = heapq.heapreplace(self.heap, trajectory)
            self.pool.discard(lowest_reward.smiles)
            self.pool.add(trajectory.smiles)

    def sample(
        self,
        n: int,
        device: str,
        reward_prioritized: bool = False,
        return_synth: bool = False,
        replace: bool = True,
    ):
        n = min(n, len(self.heap))
        if reward_prioritized:
            rewards = torch.tensor([trajectory.reward for trajectory in self.heap], dtype=torch.float32)
            min_reward = rewards.min().item()
            if min_reward <= 0:
                rewards = rewards - min_reward + 1e-6
            probabilities = rewards / rewards.sum()
            indices = torch.multinomial(
                probabilities,
                n,
                replacement=replace,
                generator=self._torch_generator,
            ).tolist()
            batch = [self.heap[i] for i in indices]
        else:
            batch = self._random.sample(self.heap, n)

        ids = pad_sequence([trajectory.ids for trajectory in batch], batch_first=True, padding_value=self.pad).to(device)
        rewards = torch.tensor([trajectory.reward for trajectory in batch], device=device)
        if return_synth:
            synthesizabilities = torch.tensor([trajectory.synthesizability for trajectory in batch], device=device)
            return {"input_ids": ids}, rewards, synthesizabilities
        return {"input_ids": ids}, rewards
