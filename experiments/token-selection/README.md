# Token-selection experiments

Shared Python package: [`token_selection/`](token_selection/) under `experiments/token-selection/`.

| Arm | Directory | Configs | Reusable code |
|-----|-----------|---------|---------------|
| **RHO-1** (rho excess) | [`rho-1/`](rho-1/) | `rho-1/configs/run_rho_10b.yaml` | `token_selection` CLI with `rho_excess` method |
| **Middle PPL** | — | `token_selection/configs/run_middle_ppl_10b.yaml` | `token_selection` CLI with `middle_ppl` method |
| **BLADE** | [`blade/`](blade/) | — | `blade/aws/train_blade_olmo_370m.py`, `prepare_blade_data.py` |
| **Control** (vanilla CE on RegMix 10B) | [`control/`](control/) | — | `control/aws/train_ce_regmix_olmo_370m.py` |
| **Reference** (HQ 5.5B model) | [`reference/`](reference/) | — | `reference/aws/train_olmo3_370m_refhq.py`, `prepare_refhq_data.py`, `export_refhq_reference.py` |

From repo root, set `PYTHONPATH=experiments/token-selection` (or `cd experiments/token-selection` and `PYTHONPATH=.`). Then launch via `python -m token_selection.scripts.*` and YAML configs. Hardware-specific shell wrappers (B200, L40S, EC2 user-data) have been removed.
