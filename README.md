

# Synthesizable Molecular Generation via Soft-constrained GFlowNets with Rich Chemical Priors (S3-GFN)

Official implementation of the paper [_"Synthesizable SMILES via Soft-constrained GFlowNets with Rich Chemical Priors"_](https://arxiv.org/abs/2602.04119)


**TL;DR -** S3-GFN is a soft-constrained GFlowNet for synthesizable SMILES generation.

**Key features**
- Soft-constrained post-training with contrastive auxiliary loss for synthesizability  
- Separate replay buffers for positive (synthesizable) and negative (unsynthesizable) samples
- Compatible with open-source SMILES language models (e.g., GP-MolFormer)


## Relationship to Prior Codebases

This repository builds upon several excellent open-source projects:

- **[RxnFlow](https://github.com/SeonghwanSeo/RxnFlow)** — synthesis pathway verification and SBDD evaluation  

- **[SynFlowNets](https://github.com/mirunacrt/synflownet)** — adapted for sEH evaluation

- **[PMO](https://github.com/wenhao-gao/mol_opt) / [Genetic-GFN](https://github.com/hyeonahkimm/genetic_gfn)** — sample-efficient evaluation, genetic operators and mutated negative generation
  
  
We thank the original authors for making their implementations publicly available.

Major components introduced in this repository:

- `src/s3gfn`: soft-constrained GFlowNet training for sEH and SBDD
- `experiments/pmo`: sample-efficient molecule generation (adapt the implementation of the PMO benchmark)

For inherited components, see `README_rxnflow.md`.



## Setup

Install the tested environment
- Python 3.10  
- PyTorch 2.5.1 (CUDA 12.1)

```bash
pip install -r requirements.txt
pip install -e .
```

(Optional) Uni-Dock: only required for structure-based drug discovery

```bash
pip install git+https://github.com/dptech-corp/Uni-Dock.git@1.1.2#subdirectory=unidock_tools
```


## Data & Preprocessing

We reuse the datasets and preprocessing pipeline from RxnFlow.

Please follow the dataset download and preprocessing instructions in `data/README.md`.


## S3-GFN Training

Run training from the repository root with package-style execution:

```bash
PYTHONPATH=src python -m s3gfn.train --task seh --training_mode s3gfn --use_retrosynthesis --retro_env stock_hb --retro_steps 3
```

The retained modes are:

| Mode | Behavior |
| --- | --- |
| `s3gfn` | Positive-only RTB with the contrastive auxiliary loss |
| `reward_shaping` | RTB with synthesizability-masked rewards and no auxiliary loss |
| `rtb` | Unconstrained RTB baseline |


Manual constraint controls for synthesizability filters:

```bash
--use_retrosynthesis
--retro_env stock_hb
--retro_steps 3
```

The name of `retro_env` should be matched with the name of directory under `data/envs`.

If you don't use retrosynthesis, synthesizability is defined based on SA scores with a threshold (defalut: 4.0).



## SBDD With Uni-Dock

SBDD requires the optional Uni-Dock installation shown above. Run a docking task with:

```bash
PYTHONPATH=src python -m s3gfn.train --task vina:aldh1 --training_mode s3gfn --use_retrosynthesis --retro_env stock_hb --retro_steps 3
```

Task and receptor names are case-insensitive. Each receptor requires:

```text
data/LIT-PCBA/<RECEPTOR>/protein.pdb
data/LIT-PCBA/<RECEPTOR>/ligand.mol2
```

The supported receptors are `ADRB2`, `ALDH1`, `ESR_antago`, `ESR_ago`, `FEN1`, `GBA`, `IDH1`, `KAT2A`, `MAPK1`, `MTORC1`, `OPRK1`, `PKM2`, `PPARG`, `VDR`, and `TP53`.

