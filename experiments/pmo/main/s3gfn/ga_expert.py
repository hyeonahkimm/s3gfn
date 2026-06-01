"""Graph genetic exploration adapted from hyeonahkimm/genetic_gfn."""

import gc
import random
from typing import List

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol

from genetic_operator import crossover, mutate

rdBase.DisableLog("rdApp.error")
MINIMUM = 1e-10


def make_mating_pool(population_mol: List[Mol], population_scores, population_size: int, rank_coefficient=0.01):
    """Sample parents with replacement, prioritizing high-ranked molecules."""
    if rank_coefficient > 0:
        scores = np.asarray(population_scores)
        ranks = np.argsort(np.argsort(-scores))
        weights = 1.0 / (rank_coefficient * len(scores) + ranks)
        indices = list(torch.utils.data.WeightedRandomSampler(weights, population_size, replacement=True))
    else:
        scores = np.asarray(population_scores) + MINIMUM
        indices = np.random.choice(len(population_mol), p=scores / scores.sum(), size=population_size, replace=True)
    return (
        [population_mol[i] for i in indices if population_mol[i] is not None],
        [population_scores[i] for i in indices if population_mol[i] is not None],
    )


def reproduce(mating_pool, mutation_rate):
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    child = crossover.crossover(parent_a, parent_b)
    return mutate.mutate(child, mutation_rate) if child is not None else None


class GeneticOperatorHandler:
    """Generate graph-GA children from a scored SMILES population."""

    def __init__(self, mutation_rate: float = 0.01, population_size: int = 64):
        self.mutation_rate = mutation_rate
        self.population_size = population_size

    def query(self, query_size, mating_pool, pool=None, rank_coefficient=0.01, mutation_rate=None):
        del pool
        mutation_rate = self.mutation_rate if mutation_rate is None else mutation_rate
        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        parents, parent_scores = make_mating_pool(
            population_mol,
            mating_pool[1],
            self.population_size,
            rank_coefficient,
        )
        children = [reproduce(parents, mutation_rate) for _ in range(query_size)]
        child_smiles = []
        for child in children:
            try:
                smiles = Chem.MolToSmiles(child)
            except Exception:
                continue
            if smiles not in child_smiles:
                child_smiles.append(smiles)
        gc.collect()
        parent_smiles = [Chem.MolToSmiles(parent) for parent in parents]
        return child_smiles, None, parent_smiles, parent_scores
