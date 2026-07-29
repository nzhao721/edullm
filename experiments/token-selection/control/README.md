# Control arm — plain CE on RegMix 10B

Vanilla cross-entropy baseline for the token-selection experiment matrix.
**No token masking / selection** — trains on the entire RegMix 10B corpus.

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
| Selection | none (plain CE) |
| Default `run_id` | **`control-regmix10b-v1`** |

Do **not** reuse or append to the old `edullm-370M-ce-regmix10b` prefix.
Old control checkpoints are orphaned for this experiment’s eval grid.

## S3 export

Checkpoints and task-loss results → `s3://edullm-checkpoints/token-sel/control/`
(`s3.prefix` / code constant: `token-sel/control` via `token_selection.olmo_ext.s3_layout`).
Disable with `S3_EXPORT=0`.

Standalone trainer (no YAML spine); S3 routing is `export_arm_checkpoint("control", …)`.

## Data

- Source: `s3://edullm-datasets/regmix/regmix-10b/`
- Tokenized memmaps (typical): `…/regmix-10b/tokenized/<domain>/<domain>.npy`
- Budget: 10B tokens → **2384** optimizer steps

```bash
python prepare_control_data.py \
  --work /path/to/work \
  --train-tokenized-root /path/to/regmix-10b/tokenized
# → work/train_tokenized/paths_train.txt
```

## Permanent checkpoint ladder

Shared helper: `token_selection.olmo_ext.checkpoint_ladder.permanent_checkpoint_steps`.

For 2384 steps / interval 125:

`{0, 125, 250, …, 2250, 2384}` — **omit 2375** (within 125 of final).

Every save is permanent (`max_checkpoints=None`); no ephemeral rotation.
On each save, rank 0 triggers the full **20-label** OLMo-ladder `task_loss_bpb`
eval (`scripts/farmshare/task_loss/eval_task_loss_olmo_core.py`) asynchronously.
Disable with `--no-task-loss-on-save` or `TASK_LOSS_EVAL=0`.

## Launch

Paths are required via env/CLI — no hardcoded GPU indices or host paths.

### 1 GPU

```bash
export TRAIN_PATHS_FILE=/path/to/paths_train.txt
export SAVE_FOLDER=/path/to/ckpts/${NAME:-control-regmix10b-v1}
export PROGRESS_DIR=/path/to/progress/${NAME:-control-regmix10b-v1}
export NPROC=1
./launch_control.sh
```

Or directly:

```bash
PYTHONPATH=experiments/token-selection \
  python experiments/token-selection/control/train_ce_regmix_olmo_370m.py \
  --name control-regmix10b-v1 \
  --train-paths-file /path/to/paths_train.txt \
  --save-folder /path/to/ckpts \
  --progress-dir /path/to/progress \
  --length-tokens 10000000000
```

### Multi-GPU

```bash
export TRAIN_PATHS_FILE=...
export SAVE_FOLDER=...
export PROGRESS_DIR=...
export NPROC=4   # any world size that divides GBS / rank_microbatch
./launch_control.sh
```

Or:

```bash
PYTHONPATH=experiments/token-selection \
  torchrun --standalone --nproc_per_node=4 \
  experiments/token-selection/control/train_ce_regmix_olmo_370m.py \
  --name control-regmix10b-v1 \
  --train-paths-file /path/to/paths_train.txt \
  --save-folder /path/to/ckpts \
  --progress-dir /path/to/progress
```

`global_batch_tokens` must be divisible by `world_size * rank_microbatch_tokens`
(fail-fast otherwise). Pin devices only via the caller’s `CUDA_VISIBLE_DEVICES`
if needed — the scripts never hardcode them.

## Files

| File | Role |
|------|------|
| `train_ce_regmix_olmo_370m.py` | Trainer (custom loop + HSDP) |
| `prepare_control_data.py` | Build memmap path list |
| `launch_control.sh` | 1..N GPU launcher |
| `README.md` | This doc |

This arm does **not** submit AWS training.
