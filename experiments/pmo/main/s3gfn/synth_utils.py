"""Mutation helpers for optional mutated-negative generation."""

from rdkit import Chem

from genetic_operator import mutate as graph_ga_mutate


def mutate(smiles: str, synth_evaluator, mode: str = "graph_ga", n_try: int = 10) -> str | None:
    """Return an unsynthesizable graph mutation, when one can be generated."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    for i in range(n_try):
        if mode != "graph_ga":
            raise ValueError(f"Invalid mode: {mode}")

        mutated = graph_ga_mutate.mutate(mol, 1.0)
        if mutated is None:
            continue

        mutated_smiles = Chem.MolToSmiles(mutated, isomericSmiles=False)
        if synth_evaluator.score(mutated_smiles):
            mol = mutated if (i + 1) % 5 == 0 else Chem.MolFromSmiles(smiles)
        else:
            return mutated_smiles
    return None
