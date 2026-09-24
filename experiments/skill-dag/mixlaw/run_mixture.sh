#!/usr/bin/env bash
# Train + fully evaluate ONE mixture on ONE GPU via DomainMixtureStream
# (edullm-data peak pool; sole supported data path — no per-mix slices).
#
# WORK may start empty aside from a staged pool + recipes.
#
# Idempotent: if task_loss_final.json already exists the mixture is skipped, and
# training itself resumes from the newest local checkpoint.
#
# Required:
#   WORK   job root (pool/, recipe/, runs/)
#
# W&B (optional):
#   Source ${WORK}/wandb-session.env (or RUN_DIR) with WANDB_API_KEY, or export
#   WANDB_API_KEY. When present, defaults WANDB_MODE=online and project mixlaw.
#
# Usage: run_mixture.sh <mix_id> <gpu_index>
set -euo pipefail

MIX_ID="${1:?usage: run_mixture.sh <mix_id> <gpu_index>}"
GPU="${2:?usage: run_mixture.sh <mix_id> <gpu_index>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}}"

: "${WORK:?Set WORK to a job root (pool/, recipe/, runs/)}"

POOL_DIR="${POOL_DIR:-$WORK/pool}"
RECIPE_WORK="${RECIPE_WORK:-$WORK/recipe}"
RUN_ROOT="${RUN_ROOT:-$WORK/runs}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"

DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-120}"
EVAL_SUBSET_BATCHES="${EVAL_SUBSET_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-6198}"
WARMUP_MODE="${WARMUP_MODE:-capped}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

WANDB_PROJECT="${WANDB_PROJECT:-mixlaw}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-60m-pilot}"
# Prefer W&B when session file / API key exists; otherwise train without it.
if [[ -f "${WORK}/wandb-session.env" || -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi

export OLMO_FLASH_ATTENTION="${OLMO_FLASH_ATTENTION:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Do not export WANDB_DISABLED=1 — trainer opts in via --wandb-mode + API key.

RUN_NAME="$(printf 'mix%02d' "$MIX_ID")"
MIX_WEIGHTS_JSON="$RECIPE_WORK/$RUN_NAME/mix_weights.json"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
PROGRESS_DIR="$RUN_DIR/progress"
LOG_DIR="$RUN_DIR/logs"
FINAL_JSON="$PROGRESS_DIR/task_loss_final.json"

if [[ ! -f "$MIX_WEIGHTS_JSON" ]]; then
  echo "[$RUN_NAME] missing $MIX_WEIGHTS_JSON (run prepare_data.sh first)" >&2
  exit 2
fi
if [[ ! -f "$POOL_DIR/edullm_data_source.json" ]]; then
  echo "[$RUN_NAME] missing $POOL_DIR/edullm_data_source.json — refuse orphan pool" >&2
  exit 2
fi
if [[ ! -d "$POOL_DIR" ]]; then
  echo "[$RUN_NAME] missing POOL_DIR=$POOL_DIR" >&2
  exit 2
fi

mkdir -p "$CKPT_DIR" "$PROGRESS_DIR" "$LOG_DIR"

if [[ -f "${WORK}/wandb-session.env" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${WORK}/wandb-session.env"
  set -u
fi
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[$RUN_NAME] WANDB_MODE=online but WANDB_API_KEY unset; falling back to disabled" >&2
  WANDB_MODE=disabled
fi
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_PROJECT WANDB_MODE
mkdir -p "${WANDB_DIR}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"

if [[ -f "$FINAL_JSON" ]]; then
  echo "[$RUN_NAME] already complete ($FINAL_JSON); skipping"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU"
MASTER_PORT=$((29500 + GPU))

WANDB_EXTRA=(
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-run-name "${WANDB_RUN_NAME}"
  --wandb-group "${WANDB_GROUP}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_EXTRA+=(--wandb-entity "${WANDB_ENTITY}")
fi

echo "[$(date -Is)] [$RUN_NAME] training on GPU $GPU (port $MASTER_PORT) dataset_id=$DATASET_ID wandb=${WANDB_MODE}"

# shellcheck disable=SC2086
torchrun \
  --nnodes=1 \
  --nproc-per-node=1 \
  --master_port="$MASTER_PORT" \
  "$CODE_DIR/train_datadecide_60m.py" \
  --name "$RUN_NAME" \
  --pool-dir "$POOL_DIR" \
  --dataset-id "$DATASET_ID" \
  --mix-weights-json "$MIX_WEIGHTS_JSON" \
  --save-folder "$CKPT_DIR" \
  --progress-dir "$PROGRESS_DIR" \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --device-eval-batch-size "$DEVICE_EVAL_BATCH_SIZE" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-subset-batches "$EVAL_SUBSET_BATCHES" \
  --num-workers "$NUM_WORKERS" \
  --warmup-mode "$WARMUP_MODE" \
  --seed "$SEED" \
  "${WANDB_EXTRA[@]}" \
  $EXTRA_TRAIN_ARGS \
  2>&1 | tee -a "$LOG_DIR/train.log"

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

# Mirror final eval onto the training W&B run when enabled (resume via wandb_run_id.txt).
if [[ "${WANDB_MODE}" != "disabled" && -f "$FINAL_JSON" && -n "${WANDB_API_KEY:-}" ]]; then
  echo "[$(date -Is)] [$RUN_NAME] logging final task-loss to W&B project=${WANDB_PROJECT}"
  PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python - <<PY
import json, os
from pathlib import Path
from mixlaw_wandb import init_wandb, wandb_log_eval, finish_wandb
import argparse
args = argparse.Namespace(
    wandb_project=os.environ.get("WANDB_PROJECT", "mixlaw"),
    wandb_entity=os.environ.get("WANDB_ENTITY") or None,
    wandb_run_name=os.environ.get("WANDB_RUN_NAME", "${RUN_NAME}"),
    wandb_group=os.environ.get("WANDB_GROUP", "60m-pilot"),
    wandb_mode=os.environ.get("WANDB_MODE", "online"),
    name="${RUN_NAME}",
)
progress = Path(r"${PROGRESS_DIR}")
payload = json.loads(Path(r"${FINAL_JSON}").read_text(encoding="utf-8"))
step = int(payload.get("step") or payload.get("global_step") or 0)
run = init_wandb(args, {"phase": "final_task_loss", "mix": "${RUN_NAME}"}, id_dir=progress, tags=["mixlaw", "final-eval"])
if run is not None:
    wandb_log_eval(run, payload if "task_loss_bpb" in payload or "labels" in payload else {"task_loss_bpb": payload.get("per_label") or payload}, step=step or 0, eval_path=Path(r"${FINAL_JSON}"))
    finish_wandb(run)
PY
fi

echo "[$(date -Is)] [$RUN_NAME] done"
