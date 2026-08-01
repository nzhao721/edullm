# Control arm — random 60% keep on RegMix 10B

Random-pruning baseline for the token-selection experiment matrix (Marion et al.
style). Each sequence keeps a **uniform random 60%** of valid next-token
positions; the rest are masked out of the CE loss.

## Architecture (RefHQ-matched)

| Field | Value |
|-------|-------|
| Config | `TransformerConfig.olmo2_370M` (full attn, no SWA) |
| Size | d_model=1024, 16 layers, 16 heads, vocab 100352 |
| Tokenizer | `allenai/dolma2-tokenizer` (`TokenizerConfig.dolma2`) |
| Seq / GBS / microbatch | 2048 / 4_194_304 / 65_536 tokens |
| Optim | SkipStepAdamW + CosWithWarmup |
| LR | peak 4e-4, warmup 24, `alpha_f=0.1` |
| Other | `z_loss_multiplier=1e-5`, `max_grad_norm=1.0`, `compile_model=True`, from scratch |
| DP | HSDP bf16; world size from `torchrun` / `WORLD_SIZE` |

## Independent variables

| Knob | Value |
|------|-------|
| Selection | uniform random keep `k=0.6` (per sequence, seeded by step) |
| Method | `random` via shared `TokenSelectTrainModule` |
| Default `run_id` | **`control-regmix10b-v2`** |

Do **not** reuse or append to the old `edullm-370M-ce-regmix10b` or `control-regmix10b-v1` prefixes.

## Ephemeral scratch + durable W&B

Designed for a clean machine whose scratch starts empty and is wiped after the job:

- **Train data:** published+validated `pretrain/regmix-10b` on `s3://edullm-data/` via `edullm_data.read` (never `s3://edullm-datasets/`).
- **Stage OK:** `--stage-dir` / `STAGE_DIR` fetches `.u32le.bin` shards for the job.
- Checkpoints/progress remain under runtime-scratch `SAVE_FOLDER` / `PROGRESS_DIR`.
- Each permanent step is synchronously evaluated, uploaded to W&B with its schema-v2 fingerprint, then recorded in local `last_durable_step.json`. Production online uploads fail closed.
- Resume with `--wandb-resume-artifact` / `WANDB_RESUME_ARTIFACT`, or an explicit local `--load-path`.

## Data

```bash
# Optional pre-stage (trainer can also stage via STAGE_DIR alone)
python prepare_control_data.py --work /tmp/control-stage
# → work/train_tokenized/paths_train.txt
```

Budget: **9.9B** tokens (`9900000000`) → **2360** optimizer steps (one epoch under ~9.989B published train; no forced 10B wrap).

## Permanent checkpoint ladder

Shared helper: `token_selection.olmo_ext.checkpoint_ladder.permanent_checkpoint_steps`.

For 2360 steps / interval 125:

`{0, 125, 250, …, 2125, 2360}` — **omit 2250** (within 125 of final).

Every save is permanent (`max_checkpoints=None`); no ephemeral rotation of the ladder.
On each save, rank 0 triggers the full **20-label** OLMo-ladder `task_loss_bpb`
eval. Disable with `--no-task-loss-on-save` or `TASK_LOSS_EVAL=0`.

## Launch

Paths are required via env/CLI — no hardcoded GPU indices or host paths.

### Clean ephemeral machine (resolve + stage)

```bash
export STAGE_DIR=/tmp/control-stage/regmix-10b
export SAVE_FOLDER=/tmp/control-ckpts/${NAME:-control-regmix10b-v2}
export PROGRESS_DIR=/tmp/control-progress/${NAME:-control-regmix10b-v2}
export NPROC=1
export FRESH=1
./launch_control.sh
```

### After optional prepare

```bash
python prepare_control_data.py --work /tmp/control-stage
export TRAIN_PATHS_FILE=/tmp/control-stage/train_tokenized/paths_train.txt
export SAVE_FOLDER=/tmp/control-ckpts/${NAME:-control-regmix10b-v2}
export PROGRESS_DIR=/tmp/control-progress/${NAME:-control-regmix10b-v2}
export NPROC=4
./launch_control.sh
```

Or directly:

```bash
PYTHONPATH=experiments/token-selection \
  python experiments/token-selection/control/train_ce_regmix_olmo_370m.py \
  --name control-regmix10b-v2 \
  --stage-dir /tmp/control-stage/regmix-10b \
  --save-folder /tmp/control-ckpts \
  --progress-dir /tmp/control-progress \
  --length-tokens 9900000000 \
  --mask-keep-rate 0.6 \
  --fresh
```

`global_batch_tokens` must be divisible by `world_size * rank_microbatch_tokens`
(fail-fast otherwise). Pin devices only via the caller’s `CUDA_VISIBLE_DEVICES`
if needed — the scripts never hardcode them.

## Files

| File | Role |
|------|------|
| `train_ce_regmix_olmo_370m.py` | Trainer (custom loop + `TokenSelectTrainModule` + S3 upload-before-end) |
| `prepare_control_data.py` | Optional edullm-data stage → path list |
| `launch_control.sh` | 1..N GPU launcher |
| `README.md` | This doc |

This arm does **not** submit AWS training.
