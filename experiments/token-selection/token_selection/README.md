# Token selection package

Shared spine for RegMix-10B token-selection arms. Train **OLMo-2 370M from scratch**
with online (or offline-filtered) token selection. Checkpoints + train metrics here;
held-out eval is the 20-label OLMo-ladder `task_loss_bpb` suite triggered on each
permanent save.

Set `PYTHONPATH=experiments/token-selection` from the repo root.

## Methods

| Method | Selection | Scorer | Arm dir |
|--------|-----------|--------|---------|
| `full` | all valid tokens | none | `../control/` (standalone) |
| `rel_ema` | top-k `L_hist − L_curr` | EMA history (`exp` or constant α; optional RefHQ seed) | `../rel-ema-exp/`, `../rel-ema-refhq/` |
| `rho_excess` | top-k `L_curr − L_ref` | frozen RefHQ | `../rho-1/` |
| `middle_ppl` | middle-k by `L_curr` | train-forward CE | `../middle-ppl-token/` |
| `attention_topk` | top-k attn-received | last-layer Q/K recompute | `../attention/` |
| `learnability` | top-k `L_early − L_late` | dual frozen RefHQ | `../learnability-token/` |

Online defaults: `k=0.6`, `t0_steps=0` (no masking warmup). LR warmup stays 24 steps.
BLADE uses a separate standalone trainer (`../blade/`) with dynamic reference syncs.

Doc-level filters (offline): `../middle-ppl-doc/`, `../learnability-doc/`.

## Arm isolation

One shared spine (`TokenSelectTrainModule` + scorers). Arms stay separate by
**config identity**, not git branches:

| Arm | Config | Method |
|-----|--------|--------|
| RHO-1 | `../rho-1/configs/run_rho_10b.yaml` | `rho_excess` |
| REL exp | `../rel-ema-exp/configs/run_rel_ema_exp_10b.yaml` | `rel_ema` |
| REL RefHQ | `../rel-ema-refhq/configs/run_rel_ema_refhq_10b.yaml` | `rel_ema` |
| middle_ppl token | `../middle-ppl-token/configs/run_middle_ppl_token_10b.yaml` | `middle_ppl` |
| attention | `../attention/configs/run_attention_10b.yaml` | `attention_topk` |
| learnability token | `../learnability-token/configs/run_learnability_10b.yaml` | `learnability` |

Do not reuse another arm’s `run_id`, `output_dir`, or S3 `prefix`.

## Layout

```
token_selection/
  configs/           # package smoke / legacy RHO YAML only
  olmo_ext/          # EMA, FrozenReference, scorers, TrainModule, ckpt ladder, task_loss
  scripts/           # sync, manifest, freeze_order, validate, train
  tests/
```

## Shared contracts

- Architecture: `olmo2_370M`, GBS `4_194_304`, seq 2048 (RefHQ-matched).
- Permanent ladder: `{0,125,…,2250,2384}` for 2384-step runs (skip 2375).
- Task loss: full 20-label RC 5-shot `task_loss_bpb` on every permanent save.
- Hardware: `torchrun` world size; leave `train.cuda_visible_devices` empty unless pinning intentionally.

## Quick start (YAML arms)

```bash
# Preflight (needs local tokens + order contract + refs where required)
python -m token_selection.scripts.validate_experiment --config <arm.yaml>

# Dry-run plan / derived steps
python -m token_selection.scripts.train_method --config <arm.yaml> --method <method>

# Launch (requires pinned olmo_core)
python -m token_selection.scripts.train_olmo_template \
  --config <arm.yaml> --method <method> --launch
```

RHO / RefHQ-seeded REL / learnability need local reference `.pt` files first
(`../reference/export_refhq_reference.py` or arm-specific export scripts).

See the top-level [`../README.md`](../README.md) for the full arm table.
