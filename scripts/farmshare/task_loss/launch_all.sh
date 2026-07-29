#!/usr/bin/env bash
# Orchestrate: mint AWS session → sync checkpoints → submit 3 single-GPU eval jobs.
# Run on FarmShare login node (via control socket) after scripts are uploaded.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SCRATCH="/scratch/users/${SUNET}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRATCH}/agent-runs/task-loss-blade-ce-relema-${STAMP}}"
CKPT_DEST="${CKPT_DEST:-${SCRATCH}/checkpoints/token-selection-370m}"
EDULLM_ROOT="${EDULLM_ROOT:-${HOME}/edullm}"
LADDER_RUN="${LADDER_RUN:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217}"
VENV="${VENV:-${LADDER_RUN}/venv}"
BASE_CONFIG="${BASE_CONFIG:-${LADDER_RUN}/checkpoints/edullm-370M-30B/step5000-unsharded/config.yaml}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${CKPT_DEST}"
chmod 700 "${RUN_DIR}"

SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -f "${SCRIPT_SRC}/eval_task_loss_olmo_core.py" "${RUN_DIR}/scripts/"
cp -f "${SCRIPT_SRC}/task_loss_eval_model.sbatch" "${RUN_DIR}/scripts/"
cp -f "${SCRIPT_SRC}/sync_ckpts_from_s3.sh" "${RUN_DIR}/scripts/"
cp -f "${SCRIPT_SRC}/mint_aws_session.sh" "${RUN_DIR}/scripts/"
# Ensure unix line endings if copied from Windows.
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.sh "${RUN_DIR}/scripts/"*.sbatch || true
chmod +x "${RUN_DIR}/scripts/"*.sh

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv at ${VENV}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "missing base config ${BASE_CONFIG}" >&2
  exit 1
fi

# Ensure olmo_core is importable in the eval venv.
if ! "${VENV}/bin/python" -c 'import olmo_core, olmo' 2>/dev/null; then
  echo "venv missing olmo_core and/or olmo; attempting pip install ai2-olmo-core into ${VENV}" >&2
  "${VENV}/bin/pip" install -q 'ai2-olmo-core' || {
    echo "Failed to install olmo_core. Point VENV at an env that has both ai2-olmo and olmo_core." >&2
    exit 1
  }
fi

echo "=== mint AWS session ==="
export RUN_DIR EDULLM_ROOT
# shellcheck disable=SC1091
source "${RUN_DIR}/scripts/mint_aws_session.sh"
export AWS_SESSION_ENV

echo "=== sync checkpoints → ${CKPT_DEST} ==="
export DEST_ROOT="${CKPT_DEST}"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
bash "${RUN_DIR}/scripts/sync_ckpts_from_s3.sh"

submit_one() {
  local model="$1" root="$2" fmt="$3" jname="$4"
  local job
  job=$(sbatch --exclude=wheat-01 \
    --job-name="${jname}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",MODEL_NAME="${model}",CKPT_ROOT="${root}",CKPT_FORMAT="${fmt}",VENV="${VENV}",BASE_CONFIG="${BASE_CONFIG}",DEVICE_EVAL_BATCH_SIZE=4,LADDER_BASE_CONFIG="${BASE_CONFIG}" \
    "${RUN_DIR}/scripts/task_loss_eval_model.sbatch")
  echo "${job}"
}

echo "=== submit 3 single-GPU eval jobs ==="
submit_one blade "${CKPT_DEST}/blade/checkpoints" state_pt tl-blade
submit_one ce-regmix "${CKPT_DEST}/ce-regmix/checkpoints" state_pt tl-ce
submit_one rel-ema "${CKPT_DEST}/rel-ema" distcp tl-relema

echo "RUN_DIR=${RUN_DIR}"
echo "CKPT_DEST=${CKPT_DEST}"
squeue -u "${SUNET}" -o '%.18i %.12P %.16j %.8T %.10M %.6D %R' || true
