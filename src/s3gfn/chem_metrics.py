"""
This file is copied from RxnFlow/src/rxnflow/tasks/utils/chem_metrics.py
"""
from rdkit import Chem, DataStructs


def compute_diverse_top_k(
    smiles: list[str],
    rewards: list[float],
    k: int,
    thresh: float = 0.5,
) -> list[int]:
    modes = [(i, smi, float(r)) for i, (r, smi) in enumerate(zip(rewards, smiles, strict=True))]
    modes.sort(key=lambda m: m[2], reverse=True)
    top_modes = [modes[0][0]]

    prev_smis = {modes[0][1]}
    mode_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(modes[0][1]))]
    for i in range(1, len(modes)):
        smi = modes[i][1]
        if smi in prev_smis:
            continue
        prev_smis.add(smi)
        if thresh > 0:
            fp = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
            sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps)
            if max(sim) >= thresh:  # div = 1- sim
                continue
            mode_fps.append(fp)
            top_modes.append(modes[i][0])
        else:
            top_modes.append(modes[i][0])
        if len(top_modes) >= k:
            break
    return top_modes
