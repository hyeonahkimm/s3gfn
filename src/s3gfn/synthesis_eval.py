"""Molecular diversity metric adapted from SynFlowNet."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist


def calculate_molecular_diversity(molecules: list[Chem.Mol], fingerprint_type: str = "morgan"):
    if fingerprint_type != "morgan":
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
    fingerprints = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2000), dtype=int) for mol in molecules]
    dissimilarities = pdist(np.array(fingerprints), metric="jaccard")
    return np.mean(dissimilarities), dissimilarities
