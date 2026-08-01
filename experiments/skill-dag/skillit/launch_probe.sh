#!/usr/bin/env bash
# Platform-agnostic Skill-It DataDecide-60M probe launcher (edullm-data stream).
#
# Required:
#   PROBE_ID           e.g. probe_dclm
#   POOL_DIR           working pool pinned to pretrain/olmo-127b/v1
#   MIX_WEIGHTS_JSON   per-probe sidecar from prepare_skillit_probe_data.py
#   SAVE_FOLDER        checkpoint directory on runtime scratch
#
# Optional:
#   PROGRESS_DIR       default: sibling of SAVE_FOLDER
#   NPROC              ranks (default 1)
#   WANDB_PROJECT      default skillit
#   WANDB_MODE         production requires online
#   ALLOW_LOCAL_ONLY   1 permits offline/disabled W&B for local smoke only
#
# Example:
#   PROBE_ID=probe_dclm POOL_DIR=$WORK/pool \
#     MIX_WEIGHTS_JSON=$WORK/recipe/probe_dclm/mix_weights.json \
#     SAVE_FOLDER=$WORK/runs/probe_dclm/checkpoints \
#     bash launch_probe.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLIT_ROOT="${SCRIPT_DIR}"
MIXLAW_ROOT="$(cd "${SKILLIT_ROOT}/../mixlaw" && pwd)"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-1}"

: "${PROBE_ID:?PROBE_ID is required}"
: "${POOL_DIR:?POOL_DIR is required}"
: "${MIX_WEIGHTS_JSON:?MIX_WEIGHTS_JSON is required}"
: "${SAVE_FOLDER:?SAVE_FOLDER is required}"

PROGRESS_DIR="${PROGRESS_DIR:-$(cd "$(dirname "${SAVE_FOLDER}")" && pwd)/progress}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-32}"
EVAL_INTERVAL=120
EVAL_SUBSET_BATCHES=4
NUM_WORKERS="${NUM_WORKERS:-6}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
SEED="${SEED:-6198}"
WARMUP_MODE="${WARMUP_MODE:-capped}"
WANDB_PROJECT="${WANDB_PROJECT:-skillit}"
ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY:-0}"
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi
if [[ "${ALLOW_LOCAL_ONLY}" != "1" && "${WANDB_MODE}" != "online" ]]; then
  echo "[launch_probe] production requires WANDB_MODE=online; set ALLOW_LOCAL_ONLY=1 only for local smoke" >&2
  exit 2
fi
if [[ "${WANDB_MODE}" == "online" ]]; then
  test -n "${WANDB_API_KEY:-}" || {
    echo "[launch_probe] WANDB_API_KEY is required for production online runs" >&2
    exit 2
  }
  "${PYTHON}" -c "import wandb" >/dev/null 2>&1 || {
    echo "[launch_probe] wandb is required for production artifact durability" >&2
    exit 2
  }
fi
if [[ -n "${EXTRA_TRAIN_ARGS:-}" || "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "[launch_probe] probe eval contract does not allow EXTRA_TRAIN_ARGS or SKIP_EVAL" >&2
  exit 2
fi
PROBE_PORT_OFFSET="$(printf '%s' "${PROBE_ID}" | cksum | awk '{print $1 % 1000}')"
MASTER_PORT="${MASTER_PORT:-$((29500 + PROBE_PORT_OFFSET))}"

# Keep mixlaw trainer quiet on W&B; we log final eval from skillit side only.
export WANDB_DISABLED=1
export OLMO_FLASH_ATTENTION=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ ! -f "${MIX_WEIGHTS_JSON}" ]]; then
  echo "[${PROBE_ID}] missing MIX_WEIGHTS_JSON=${MIX_WEIGHTS_JSON}" >&2
  exit 2
fi
"${PYTHON}" "${SKILLIT_ROOT}/prepare_skillit_370m_data.py" \
  --pool-dir "${POOL_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --dataset-version "${DATASET_VERSION}" \
  --pool-layout probe \
  --validate-pool-only
"${PYTHON}" - "${MIX_WEIGHTS_JSON}" "${DATASET_ID}" "${DATASET_VERSION}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
got = (payload.get("dataset_id"), payload.get("dataset_version"))
expected = (sys.argv[2], sys.argv[3])
if got != expected:
    raise SystemExit(f"{sys.argv[1]} source {got!r} != pinned source {expected!r}")
PY

mkdir -p "${SAVE_FOLDER}" "${PROGRESS_DIR}"
FINAL_JSON="${PROGRESS_DIR}/task_loss_final.json"
if [[ -f "${FINAL_JSON}" ]]; then
  echo "[${PROBE_ID}] already complete (${FINAL_JSON}); skipping"
  exit 0
fi

if compgen -G "${SAVE_FOLDER}/step*" >/dev/null || [[ -f "${SAVE_FOLDER}/config.yaml" ]]; then
  echo "[${PROBE_ID}] clearing partial checkpoints under ${SAVE_FOLDER}"
  rm -rf "${SAVE_FOLDER:?}/"*
fi

TRAIN_SCRIPT="${MIXLAW_ROOT}/train_datadecide_60m.py"
EVAL_SCRIPT="${MIXLAW_ROOT}/eval_task_loss.py"

ARGS=(
  --name "${PROBE_ID}"
  --pool-dir "${POOL_DIR}"
  --mix-weights-json "${MIX_WEIGHTS_JSON}"
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

echo "[launch_probe] PROBE_ID=${PROBE_ID} POOL_DIR=${POOL_DIR} NPROC=${NPROC} WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE} ALLOW_LOCAL_ONLY=${ALLOW_LOCAL_ONLY}"

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
  if [[ -n "${LADDER_BASE_CONFIG:-}" ]]; then
    eval_args+=(--base-config "${LADDER_BASE_CONFIG}")
  fi
  MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    "${PYTHON}" "${EVAL_SCRIPT}" "${eval_args[@]}"
}

run_train

LATEST_CKPT="$(find "${SAVE_FOLDER}" -maxdepth 1 \( -name 'step*-unsharded' -o -name 'step*' \) -type d \
  2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "[${PROBE_ID}] no checkpoint found under ${SAVE_FOLDER}" >&2
  exit 3
fi

echo "[launch_probe] post-eval from ${LATEST_CKPT} (6 curve labels)"
run_eval "${LATEST_CKPT}"

# Final probe eval → same W&B project as 370M arms (no mixlaw trainer edits).
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  unset WANDB_DISABLED || true
  export WANDB_MODE
  PYTHONPATH="${SKILLIT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" -c "
from pathlib import Path
from wandb_logging import log_probe_final_eval
url = log_probe_final_eval(
    eval_path=Path(r'''${FINAL_JSON}'''),
    probe_id=r'''${PROBE_ID}''',
    project=r'''${WANDB_PROJECT}''',
    entity=(r'''${WANDB_ENTITY:-}''' or None),
    mode=r'''${WANDB_MODE}''',
    save_folder=Path(r'''${SAVE_FOLDER}'''),
    progress_dir=Path(r'''${PROGRESS_DIR}'''),
)
print(f'[launch_probe] wandb probe eval url={url}', flush=True)
"
fi

echo "[launch_probe] ${PROBE_ID} done"
