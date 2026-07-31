# REL RefHQ-init constant-α (`rel-ema-refhq`)

**Only arm that seeds EMA history from RefHQ.** Trainable weights still start from
scratch on RegMix 10B; the exported RefHQ step1315 `model.pt` initializes the EMA
history buffers so REL can score against a strong history from step 0.

Near-clone of [`../rel-ema-exp/`](../rel-ema-exp/): controlled knobs match; only
**EMA seed** (`refhq` vs `zero`) and **α schedule** (constant `0.9985` vs
`α(t)=1−exp(−t/300)`) differ.

| Knob | Value |
|------|--------|
| Method | `rel_ema` — top 60% `REL = L_hist − L_curr` |
| EMA seed | RefHQ step1315 (via `ema.seed_mode: refhq`) |
| α | constant `0.9985` (~20% half-life on 2360 steps) |
| `t0` | `0` (selection from step 0; no masking warmup) |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Data | RegMix one-epoch (`pretrain/regmix-10b` on `s3://edullm-data/`) → **2360** steps |
| Checkpoints | `{0, 125, …, 2125, 2360}` (skip 2250); permanent |
| Eval | Immediate full 20-label `task_loss_bpb` on each permanent save |
| `run_id` | `rel-ema-refhq-10b-scratch-v1` |
| S3 export | `s3://edullm-checkpoints/token-sel/rel-ema-refhq/` |

**Ephemeral scratch:** set `RUN_DIR` empty; stage edullm-data each job; durable
export via the shared spine. `--resume` fetches from S3 when local is empty.

Contrast: `rel-ema-exp` uses **bias-corrected zero-init** EMA (`ema.seed_mode: zero`)
and an exponential α schedule — it must **not** call `EMAHistory.seed_from_state_dict`.

Shared package: [`../token_selection/`](../token_selection/).
Config: [`configs/run_rel_ema_refhq_10b.yaml`](configs/run_rel_ema_refhq_10b.yaml).

## Seed API

```python
from token_selection.olmo_ext.ema import EMAHistory, alpha_for_half_life
from token_selection.olmo_ext.train_module import load_reference_state_dict

# α ≈ 0.9985 for ~20% of a 2360-step run:
assert abs(alpha_for_half_life(0.2 * 2360) - 0.9985) < 5e-5

weights = load_reference_state_dict("/path/to/refhq_step1315_model.pt")
ema = EMAHistory.from_module_seeded(model, weights, alpha=0.9985)
# or: ema = EMAHistory.from_module(model, alpha=0.9985); ema.seed_from_state_dict(model, weights)
assert ema.has_history and ema.correction == 1.0
```

YAML:

```yaml
ema:
  seed_mode: refhq   # ONLY this arm
reference:
  load_path: /path/to/refhq_step1315_model.pt
alpha_schedule: linear
alpha_start: 0.9985
alpha_end: 0.9985
t0_steps: 0
```

## Export RefHQ seed

```bash
python experiments/token-selection/reference/export_refhq_reference.py \
  --work-dir /path/to/refhq_export_work \
  --output /path/to/refhq_step1315_model.pt
```

Set `REF_PT` (or `reference.load_path` in the YAML) to that `model.pt` before train.
If unset, `--launch` auto-materializes `reference.s3_uri` into a job cache.

## Launch (1..N GPU)

```bash
export EDULLM_ROOT=/path/to/edullm
export OLMO_CORE_DIR=/path/to/OLMo-core   # must match YAML olmo_core.revision
export RUN_DIR=/path/to/empty/scratch     # required; job-local tokens/ckpts
export REF_PT=/path/to/refhq_step1315_model.pt   # optional if YAML s3_uri set
# Optional hardware: NUM_GPUS=4  or  CUDA_VISIBLE_DEVICES=0,1,2,3
# Optional memory:   RANK_MICROBATCH_SIZE=16384

bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-refhq/launch_train.sh" prepare
bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-refhq/launch_train.sh" train
# Resume later (durable S3 hydrate if local empty):
bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-refhq/launch_train.sh" train --resume
```

FarmShare thin wrapper (same launcher; defaults `RUN_DIR` + `RANK_MICROBATCH_SIZE=16384`):

```bash
export REF_PT=/path/to/refhq_step1315_model.pt
export RUN_DIR=/scratch/$USER/rel-ema-refhq-run
bash "$EDULLM_ROOT/experiments/token-selection/rel-ema-refhq/farmshare/run_rel_ema_refhq.sh" train
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
  --config experiments/token-selection/rel-ema-refhq/configs/run_rel_ema_refhq_10b.yaml \
  --method rel_ema \
  --olmo-root "$OLMO_CORE_DIR" \
  --launch
```

Do **not** submit AWS training from this arm unless you explicitly authorize a
concrete AWS workload. Task-loss outputs land under
`task_loss_results/rel-ema-refhq/step{N}_task_loss.json`
(or `$RUN_DIR/...` when `output_dir` is rewritten by the launch script).
