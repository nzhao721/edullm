#!/usr/bin/env bash
# Platform-agnostic Skill-It DataDecide-60M probe launcher.
#
# Required:
#   PROBE_ID       e.g. probe_dclm | probe_arxiv | ...
#   SAVE_FOLDER    checkpoint directory
#   PATHS_FILE     paths_train.txt for this probe (from build_mixture_data)
#
# Optional:
#   PROGRESS_DIR   default: <SAVE_FOLDER>/../progress or $PROGRESS_DIR
# Optional:
#   NPROC          ranks (default 1): 1 → python, >1 → torchrun
#   DEVICE_BATCH_SIZE / NUM_WORKERS / SEED / WARMUP_MODE
#   RESULTS_S3     optional progress/log sync prefix (aws s3 sync; best-effort)
#   PYTHON         python executable
#
# Exact mixlaw pilot eval cadence (do NOT pass --full-task-suite-in-run):
#   eval_interval=120, eval_subset_batches=4, device_eval_batch_size=32
#
# Example:
#   PROBE_ID=probe_dclm PATHS_FILE=$WORK/slices/probe_dclm/paths_train.txt \
#     SAVE_FOLDER=$WORK/runs/probe_dclm/checkpoints \
#     PROGRESS_DIR=$WORK/runs/probe_dclm/progress \
#     NPROC=1 bash launch_probe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLIT_ROOT="${SCRIPT_DIR}"
MIXLAW_ROOT="$(cd "${SKILLIT_ROOT}/../mixlaw" && pwd)"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-1}"

: "${PROBE_ID:?PROBE_ID is required}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required}"
: "${PATHS_FILE:?PATHS_FILE is required}"

PROGRESS_DIR="${PROGRESS_DIR:-$(cd "$(dirname "${SAVE_FOLDER}")" && pwd)/progress}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-32}"
EVAL_INTERVAL=120
EVAL_SUBSET_BATCHES=4
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-6198}"
WARMUP_MODE="${WARMUP_MODE:-capped}"
RESULTS_S3="${RESULTS_S3:-}"
if [[ -n "${EXTRA_TRAIN_ARGS:-}" || "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[launch_probe] probe eval contract does not allow EXTRA_TRAIN_ARGS or SKIP_EVAL" >&2
  exit 2
fi
PROBE_PORT_OFFSET="$(printf '%s' "${PROBE_ID}" | cksum | awk '{print $1 % 1000}')"
MASTER_PORT="${MASTER_PORT:-$((29500 + PROBE_PORT_OFFSET))}"

export WANDB_DISABLED=1
export WANDB_MODE=disabled
export OLMO_FLASH_ATTENTION=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "[${PROBE_ID}] missing PATHS_FILE=${PATHS_FILE}" >&2
  exit 2
fi

mkdir -p "${SAVE_FOLDER}" "${PROGRESS_DIR}"
FINAL_JSON="${PROGRESS_DIR}/task_loss_final.json"
if [[ -f "${FINAL_JSON}" ]]; then
  echo "[${PROBE_ID}] already complete (${FINAL_JSON}); skipping"
  exit 0
fi

# Drop partial checkpoints from a crashed run (resume is not supported for probes).
if compgen -G "${SAVE_FOLDER}/step*" >/dev/null || [[ -f "${SAVE_FOLDER}/config.yaml" ]]; then
  echo "[${PROBE_ID}] clearing partial checkpoints under ${SAVE_FOLDER}"
  rm -rf "${SAVE_FOLDER:?}/"*
fi

TRAIN_SCRIPT="${MIXLAW_ROOT}/train_datadecide_60m.py"
EVAL_SCRIPT="${MIXLAW_ROOT}/eval_task_loss.py"

ARGS=(
  --name "${PROBE_ID}"
  --paths-file "${PATHS_FILE}"
  --save-folder "${SAVE_FOLDER}"
  --progress-dir "${PROGRESS_DIR}"
  --device-batch-size "${DEVICE_BATCH_SIZE}"
  --device-eval-batch-size "${DEVICE_EVAL_BATCH_SIZE}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-subset-batches "${EVAL_SUBSET_BATCHES}"
  --num-workers "${NUM_WORKERS}"
  --warmup-mode "${WARMUP_MODE}"
  --seed "${SEED}"
)

echo "[launch_probe] PROBE_ID=${PROBE_ID} NPROC=${NPROC} mbs=${DEVICE_BATCH_SIZE} eval_mbs=${DEVICE_EVAL_BATCH_SIZE} eval_interval=${EVAL_INTERVAL}"

run_train() {
  if [[ "${NPROC}" -eq 1 ]]; then
    MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
      "${PYTHON}" "${TRAIN_SCRIPT}" "${ARGS[@]}"
  else
    torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
      "${TRAIN_SCRIPT}" "${ARGS[@]}"
  fi
}

run_eval() {
  local ckpt="$1"
  local eval_args=(
    --checkpoint "${ckpt}"
    --out "${FINAL_JSON}"
    --run-name "${PROBE_ID}"
    --device-eval-batch-size "${DEVICE_EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
  )
  MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    "${PYTHON}" "${EVAL_SCRIPT}" "${eval_args[@]}"
}

run_train

# Newest unsharded checkpoint (OLMo writes step*-unsharded directories) or plain step*.
LATEST_CKPT="$(find "${SAVE_FOLDER}" -maxdepth 1 \( -name 'step*-unsharded' -o -name 'step*' \) -type d \
  2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "[${PROBE_ID}] no checkpoint found under ${SAVE_FOLDER}" >&2
  exit 3
fi

echo "[launch_probe] post-eval from ${LATEST_CKPT} (6 curve labels)"
run_eval "${LATEST_CKPT}"

if [[ -n "${RESULTS_S3}" ]]; then
  if command -v aws >/dev/null 2>&1; then
    aws s3 sync "${PROGRESS_DIR}" "${RESULTS_S3}/${PROBE_ID}/progress" --only-show-errors || true
  else
    echo "[launch_probe] aws CLI missing; skip RESULTS_S3 sync" >&2
  fi
fi

echo "[launch_probe] ${PROBE_ID} done"
