from __future__ import annotations

from pathlib import Path

from rxnflow.tasks.unidock_vina import VinaReward

if __package__:
    from .synthesizability import SynthesizabilityEvaluator
else:
    from synthesizability import SynthesizabilityEvaluator


def main():
    smiles_list = [
        "CC1CCc2c(F)cccc2C1NC(=O)c1ccc2c(c1)C(=O)CC2",
        "CN1C(=O)Cc2c(C(=O)Nc3cccc4c3CCCC4=O)cccc21",
        "O=C1CCC(C(=O)NC2C3CCC2Cc2ccccc2C3)c2ccccc21",
    ]

    synth_eval = SynthesizabilityEvaluator(use_retrosynthesis=True, env="stock_hb", max_steps=3)
    synth_score = synth_eval.score_batch(smiles_list)

    vina_receptor = "ALDH1"
    base_dir = Path("../../data/LIT-PCBA") / vina_receptor
    protein_pdb = base_dir / "protein.pdb"
    ref_ligand = base_dir / "ligand.mol2"
    if not protein_pdb.exists() or not ref_ligand.exists():
        raise FileNotFoundError(f"Missing docking inputs under {base_dir}. Expected protein.pdb and ligand.mol2.")

    vina = VinaReward(
        protein_pdb_path=protein_pdb,
        center=None,
        ref_ligand_path=ref_ligand,
        search_mode="balance",
        num_workers=4,
    )
    vina_score = vina.run_smiles(smiles_list, save_path="./test_docking.sdf")[0]

    print("=== Single-sample environment check ===")
    print(f"SMILES           : {smiles_list[0]}")
    print(f"Synthesizability : {synth_score}")
    print(f"Vina score       : {vina_score}")


if __name__ == "__main__":
    main()
