# Skill-DAG experiments

Domain-mixture and Skill-It experiments for prerequisite-ordered data selection.

| Subdirectory | Description |
|--------------|-------------|
| [`mixlaw/`](mixlaw/) | Data mixing laws pilot (24× DataDecide-60M mixtures over olmohq domains). Chinchilla step laws use **in-run `task_loss.jsonl` only** (steps 120–1440); step-1451 final eval is excluded from fits. |
| [`skillit/`](skillit/) | Skill-It probes (7× one-hot DataDecide-60M) + dual-arm OLMo2-370M training. Offline **A_ij** from Chinchilla-extrapolated probes vs RegMix (`mixlaw_fit_chinchilla.json`). |

## Artifact quick links

| What | Path |
|------|------|
| Mixing-law fit + optima | [`mixlaw/mixlaw_fit_chinchilla.json`](mixlaw/mixlaw_fit_chinchilla.json) |
| LightGBM fit + optima | [`mixlaw/mixlaw_fit_lightgbm_chinchilla.json`](mixlaw/mixlaw_fit_lightgbm_chinchilla.json) |
| Chinchilla extrapolation report | [`mixlaw/mixlaw_chinchilla_extrapolated.json`](mixlaw/mixlaw_chinchilla_extrapolated.json) |
| 370M validation recipe (domain weights) | [`mixlaw/validation_mixtures_10b.json`](mixlaw/validation_mixtures_10b.json) |
| Skill-It offline A (7×6) | [`skillit/artifacts/probes_full/A_offline.json`](skillit/artifacts/probes_full/A_offline.json) |
| Skill-It Chinchilla curves | [`skillit/artifacts/probes_full/task_loss_chinchilla_by_family.png`](skillit/artifacts/probes_full/task_loss_chinchilla_by_family.png) |

Refresh fits after changing extrapolation policy:

```bash
cd experiments/skill-dag/mixlaw
py -3 extrapolate_chinchilla.py && py -3 fit_chinchilla.py && py -3 loo_chinchilla.py && py -3 fit_lightgbm_chinchilla.py && py -3 generate_readme.py

cd ../skillit
py -3 plot_probe_chinchilla_results.py --runs-dir artifacts/probes_full/runs --logs-dir artifacts/probes_full/logs --out-dir artifacts/probes_full
```
