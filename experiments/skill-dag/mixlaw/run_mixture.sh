#!/usr/bin/env bash
# Train + fully evaluate ONE mixture on ONE GPU.
#
# Idempotent: if task_loss_final.json already exists the mixture is skipped, and
# training itself resumes from the newest checkpoint in the save folder.
#
# Usage: run_mixture.sh <mix_id> <gpu_index>
set -euo pipefail

MIX_ID="${1:?usage: run_mixture.sh <mix_id> <gpu_index>}"
GPU="${2:?usage: run_mixture.sh <mix_id> <gpu_index>}"

WORK="${WORK:-/opt/edullm/mixlaw}"
CODE_DIR="${CODE_DIR:-$WORK/code}"
SLICE_DIR="${SLICE_DIR:-$WORK/slices}"
RUN_ROOT="${RUN_ROOT:-$WORK/runs}"
RESULTS_S3="${RESULTS_S3:-}"

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
# On B200 + dolma2 vocab (100352): mbs=96 OOMs CE logits; probe peak is mbs=32 (~209k tok/s).
# Global batch stays 96 via grad_accum = 96/mbs.
DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-120}"
EVAL_SUBSET_BATCHES="${EVAL_SUBSET_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-6198}"
WARMUP_MODE="${WARMUP_MODE:-capped}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

export WANDB_DISABLED=1
export WANDB_MODE=disabled
# Blackwell has no prebuilt flash-attn wheel for most torch releases; PyTorch SDPA
# covers sm_100 and is plenty at seq 2048 / head_dim 32.
export OLMO_FLASH_ATTENTION="${OLMO_FLASH_ATTENTION:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

RUN_NAME="$(printf 'mix%02d' "$MIX_ID")"
MIX_SLICES="$SLICE_DIR/$RUN_NAME"
PATHS_FILE="$MIX_SLICES/paths_train.txt"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
PROGRESS_DIR="$RUN_DIR/progress"
LOG_DIR="$RUN_DIR/logs"
FINAL_JSON="$PROGRESS_DIR/task_loss_final.json"

if [[ ! -f "$PATHS_FILE" ]]; then
  echo "[$RUN_NAME] missing $PATHS_FILE (run prepare_data.sh first)" >&2
  exit 2
fi

mkdir -p "$CKPT_DIR" "$PROGRESS_DIR" "$LOG_DIR"

if [[ -f "$FINAL_JSON" ]]; then
  echo "[$RUN_NAME] already complete ($FINAL_JSON); skipping"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
# Distinct rendezvous port per GPU so concurrent single-GPU runs on one node do
# not collide.
MASTER_PORT=$((29500 + GPU))

echo "[$(date -Is)] [$RUN_NAME] training on GPU $GPU (port $MASTER_PORT)"

# shellcheck disable=SC2086
torchrun \
  --nnodes=1 \
  --nproc-per-node=1 \
  --master_port="$MASTER_PORT" \
  "$CODE_DIR/train_datadecide_60m.py" \
  --name "$RUN_NAME" \
  --paths-file "$PATHS_FILE" \
  --save-folder "$CKPT_DIR" \
  --progress-dir "$PROGRESS_DIR" \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --device-eval-batch-size "$DEVICE_EVAL_BATCH_SIZE" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-subset-batches "$EVAL_SUBSET_BATCHES" \
  --num-workers "$NUM_WORKERS" \
  --warmup-mode "$WARMUP_MODE" \
  --seed "$SEED" \
  $EXTRA_TRAIN_ARGS \
  2>&1 | tee -a "$LOG_DIR/train.log"

# Newest unsharded checkpoint (OLMo writes step<N>-unsharded directories).
LATEST_CKPT="$(find "$CKPT_DIR" -maxdepth 1 -name 'step*-unsharded' -type d \
  | sort -t't' -k3 -n | tail -n 1)"
if [[ -z "$LATEST_CKPT" ]]; then
  echo "[$RUN_NAME] no unsharded checkpoint found under $CKPT_DIR" >&2
  exit 3
fi

echo "[$(date -Is)] [$RUN_NAME] full task-loss eval from $LATEST_CKPT"
torchrun \
  --nnodes=1 \
  --nproc-per-node=1 \
  --master_port="$MASTER_PORT" \
  "$CODE_DIR/eval_task_loss.py" \
  --checkpoint "$LATEST_CKPT" \
  --out "$FINAL_JSON" \
  --run-name "$RUN_NAME" \
  --device-eval-batch-size "$DEVICE_EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  2>&1 | tee -a "$LOG_DIR/eval.log"

if [[ -n "$RESULTS_S3" ]]; then
  aws s3 sync "$PROGRESS_DIR" "$RESULTS_S3/$RUN_NAME/progress" --only-show-errors || true
  aws s3 sync "$LOG_DIR" "$RESULTS_S3/$RUN_NAME/logs" --only-show-errors || true
fi

echo "[$(date -Is)] [$RUN_NAME] done"
