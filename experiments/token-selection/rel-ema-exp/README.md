# REL no-init exponential-α (`rel-ema-exp`)

> **Launch sections below are historical.** The scripts they name were removed with the superseded in-repo trainer, which produced none of the reported runs. The code and launch path behind the reported numbers are in `../olmo_core_token_selection/` (`farmshare/` and `runpod/`); see `../olmo_core_token_selection/PROVENANCE.md`.

Online token selection: keep top **60%** by `REL = L_curr − L_hist`.

**Polarity corrected 2026-09-05.** This arm was originally documented and run with
`REL = L_hist − L_curr`, which is inverted relative to the `rho_excess` / BLADE convention
(current minus reference/history) and therefore selected the tokens the live model had already
*mastered* relative to its own history. The scorer computes `current − history`, and the arm was
re-run on 2026-09-05 with the fixed polarity. **The reported arm is W&B
`eduLLM/token-selection/cc52d5537a03ad8e57cc87a025668b2e`**, which ran on 4xL40S; cite only
that run.

| Knob | Value |
|------|--------|
| EMA | Bias-corrected from **zero** (no RefHQ / θ₀ seed) |
| α schedule | `α(t) = 1 − exp(−t/300)` (`tau=300`) |
| `t0` | **0** (selection from step 0) |
| `k` / γ | 0.6 |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Data | `pretrain/regmix-10b` v1 on `s3://edullm-data/` — realized **10,004,807,041** tokens — one epoch → **2360** steps = 9,898,557,440 tokens, no wrap |
| Checkpoints | `{0, 125, …, 2125, 2360}` (skip 2250) |
| Eval | Full 20-label `task_loss_bpb` on every permanent save |
| `run_id` | `rel-ema-exp-10b-scratch-v1` (**not** `rel-ema-10b-scratch-v1`) |
| Artifact durability | Runtime scratch + W&B |

**Ephemeral scratch:** set `RUN_DIR` empty; stage edullm-data each job; durable
export via the shared spine. `--resume` fetches from S3 when local is empty.

Only the EMA seed mode and α schedule
should differ. Shared package: [`../token_selection/`](../token_selection/).
Config: [`configs/run_rel_ema_exp_10b.yaml`](configs/run_rel_ema_exp_10b.yaml).

## α schedule API

```python
from token_selection.olmo_ext.ema import alpha_exp, alpha_at_step, DEFAULT_ALPHA_TAU

alpha_exp(0, tau=300)           # → 0.0
alpha_exp(300, tau=300)         # → 1 - e^{-1} ≈ 0.632
alpha_at_step(t, t0=0, total_steps=2360,
              alpha_start=0.0, alpha_end=1.0,
              schedule="exp", tau=300)
```

YAML (either top-level or under `ema:`):

```yaml
alpha_schedule: exp
alpha_tau: 300
ema:
  schedule: exp
  tau: 300
  seed_mode: zero
```

`linear` remains the default for other REL arms.

## Controlled stack (shared spine)

`z_loss_multiplier=1e-5` and `max_grad_norm=1.0` are declared in YAML and wired
through `train_olmo_template` → `TokenSelectTrainModule` (CE+z over the selection
mask). Same defaults as RefHQ / control / other spine arms.

## Launch

```bash
export EDULLM_ROOT=/path/to/edullm
export OLMO_CORE_DIR=/path/to/OLMo-core   # must match YAML olmo_core.revision
export RUN_DIR=/path/to/empty/scratch     # required; job-local tokens/ckpts
# Optional hardware: NUM_GPUS=4  or  CUDA_VISIBLE_DEVICES=0,1,2,3
# Optional memory:   RANK_MICROBATCH_SIZE=16384
# Optional workers:  NUM_WORKERS=8   (default: keep YAML value)

bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh" prepare
bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh" train
# Resume later from W&B if local scratch is empty:
WANDB_RESUME_ARTIFACT=entity/project/run-checkpoint:latest \
  bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh" train --resume
```

Or call the shared trainer directly:

```bash
cd "$EDULLM_ROOT"
export PYTHONPATH=experiments/token-selection
export CUDA_VISIBLE_DEVICES=0,1   # example; omit under Slurm allocator
export TOKEN_SELECTION_SKIP_IDLE_CHECK=1

NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
torchrun --standalone --nproc_per_node="$NPROC" \
  -m token_selection.scripts.train_olmo_template \
  --config experiments/token-selection/rel-ema-exp/configs/run_rel_ema_exp_10b.yaml \
  --method rel_ema \
  --olmo-root "$OLMO_CORE_DIR" \
  --launch
```

Do **not** submit AWS training from this arm unless explicitly authorized.
With `launch_train.sh`, task-loss JSON lands under
`$RUN_DIR/task_loss_results/rel-ema-exp/step{N}_task_loss.json`.
Direct YAML launches use `task_loss_results/rel-ema-exp/` under
`experiments/token-selection/`.
