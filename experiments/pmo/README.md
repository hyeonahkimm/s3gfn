# PMO Experiments

This directory contains the PMO oracle-budget experiment wrapper and the
S3-GFN adapter under `main/s3gfn`.

This codebase is implemented based on the PMO benchmark. For metadata and benchmark details, please refer to the original repository [here](https://github.com/wenhao-gao/mol_opt)

Run an experiment:

```bash
python run.py s3gfn --oracles jnk3 --seed 0
```


The maintained algorithm configuration is
`main/s3gfn/hparams_default.yaml`. Filesystem result persistence is disabled.
Use `--wandb online` or `--wandb offline` when metric logging is needed.
