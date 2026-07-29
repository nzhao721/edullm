# REL no-init exponential-α (`rel-ema-exp`)

Online token selection: keep top **60%** by `REL = L_hist − L_curr`.

| Knob | Value |
|------|--------|
| EMA | Bias-corrected from **zero** (no RefHQ / θ₀ seed) |
| α schedule | `α(t) = 1 − exp(−t/300)` (`tau=300`) |
| `t0` | **0** (selection from step 0) |
| `k` / γ | 0.6 |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Data | RegMix 10B → ~2384 steps |
| Checkpoints | `{0, 125, …, 2250, 2384}` (skip 2375) |
| Eval | Full 20-label `task_loss_bpb` on every permanent save |
| `run_id` | `rel-ema-exp-10b-scratch-v1` (**not** `rel-ema-10b-scratch-v1`) |
| S3 export | `s3://edullm-checkpoints/token-sel/rel-ema-exp/` |

Near-clone of [`../rel-ema-refhq/`](../rel-ema-refhq/): only EMA seed + α schedule
should differ. Shared package: [`../token_selection/`](../token_selection/).
Config: [`configs/run_rel_ema_exp_10b.yaml`](configs/run_rel_ema_exp_10b.yaml).

## α schedule API

```python
from token_selection.olmo_ext.ema import alpha_exp, alpha_at_step, DEFAULT_ALPHA_TAU

alpha_exp(0, tau=300)           # → 0.0
alpha_exp(300, tau=300)         # → 1 - e^{-1} ≈ 0.632
alpha_at_step(t, t0=0, total_steps=2384,
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

`linear` remains the default for other REL arms; `seed_mode: refhq` is only for `rel-ema-refhq`.

## Controlled stack (shared spine)

`z_loss_multiplier=1e-5` and `max_grad_norm=1.0` are declared in YAML and wired
through `train_olmo_template` → `TokenSelectTrainModule` (CE+z over the selection
mask). Same defaults as RefHQ / control / other spine arms.

## Launch

```bash
export EDULLM_ROOT=/path/to/edullm
export OLMO_CORE_DIR=/path/to/OLMo-core   # must match YAML olmo_core.revision
export RUN_DIR=/path/to/scratch/rel-ema-exp-run
# Optional hardware: NUM_GPUS=4  or  CUDA_VISIBLE_DEVICES=0,1,2,3
# Optional memory:   RANK_MICROBATCH_SIZE=16384
# Optional workers:  NUM_WORKERS=8   (default: keep YAML value)

bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh" prepare
bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-exp/launch_train.sh" train
# Resume later:
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
