#!/usr/bin/env bash
# Enqueue full 20-label task_loss_bpb for one permanent RHO-1 checkpoint.
#
# Invoked by TaskLossEvalCallback via TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT:
#   bash enqueue_task_loss.sh <step> <checkpoint_dir> <out_json> <run_id>
#
# Prefer Slurm when available; otherwise run torchrun locally in the background.
set -euo pipefail

STEP="${1:?step}"
CHECKPOINT_DIR="${2:?checkpoint_dir}"
OUT_JSON="${3:?out_json}"
RUN_ID="${4:-rho-1-regmix10b-v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-$(cd "$TS_ROOT/../.." && pwd)}"
EVAL_PY="${TOKEN_SELECTION_TASK_LOSS_EVAL_PY:-$EDULLM_ROOT/scripts/farmshare/task_loss/eval_task_loss_olmo_core.py}"
MODEL_NAME="${TOKEN_SELECTION_TASK_LOSS_MODEL_NAME:-rho-1}"
DEVICE_BS="${DEVICE_EVAL_BATCH_SIZE:-4}"
VENV="${VENV:-}"

mkdir -p "$(dirname "$OUT_JSON")"

if [[ -f "$OUT_JSON" ]]; then
  echo "SKIP step${STEP} (exists $OUT_JSON)"
  exit 0
fi

run_eval() {
  if [[ -n "$VENV" && -x "$VENV/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  fi
  export WANDB_DISABLED=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
  torchrun --standalone --nproc_per_node=1 \
    "$EVAL_PY" \
    --checkpoint "$CHECKPOINT_DIR" \
    --format auto \
    --out "$OUT_JSON" \
    --run-name "${MODEL_NAME}-step${STEP}" \
    --device-eval-batch-size "$DEVICE_BS"
}

if command -v sbatch >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}${ALLOW_SBATCH_TASK_LOSS:-}" ]]; then
  # Nested Slurm job for eval (training job continues). Requires ALLOW_SBATCH_TASK_LOSS=1
  # when not already inside a Slurm allocation that permits sbatch.
  JOB_NAME="tl-rho1-${STEP}"
  SBATCH_SCRIPT="${TOKEN_SELECTION_TASK_LOSS_SBATCH:-$EDULLM_ROOT/scripts/farmshare/task_loss/task_loss_eval_step.sbatch}"
  if [[ -f "$SBATCH_SCRIPT" ]]; then
    export RUN_DIR="${RUN_DIR:-$(dirname "$(dirname "$CHECKPOINT_DIR")")}"
    export MODEL_NAME STEP CHECKPOINT_DIR
    export CKPT_FORMAT=distcp
    export VENV="${VENV:-${RUN_DIR}/venv}"
    sbatch --exclude=wheat-01 --job-name="$JOB_NAME" --gres=gpu:1 \
      --export=ALL,RUN_DIR,MODEL_NAME,STEP,CHECKPOINT_DIR,CKPT_FORMAT,VENV \
      --time="${TASK_LOSS_TIME:-01:00:00}" \
      "$SBATCH_SCRIPT" || run_eval &
    exit 0
  fi
fi

# Local / fallback: background torchrun so training is not blocked.
run_eval &
echo "enqueued local pid=$! step=${STEP} out=${OUT_JSON}"
