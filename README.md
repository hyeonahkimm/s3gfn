

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
