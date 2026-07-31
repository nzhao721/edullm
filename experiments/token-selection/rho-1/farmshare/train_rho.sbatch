#!/usr/bin/env bash
#SBATCH --job-name=rho-1-v1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=logs/rho1_%j.out
#SBATCH --error=logs/rho1_%j.err
#SBATCH --exclude=wheat-01
#
# RHO-1 train (rho-1-regmix10b-v1). GPU count comes from --gres / env,
# not from a hard-coded device list. Pass e.g. --gres=gpu:4 on sbatch.
#
# Ephemeral empty-scratch (default):
#   FROM_SCRATCH=1 — stage edullm-data; durable export to edullm-checkpoints
# Crash resume of THIS run_id only: RESUME=1 (fetches from S3 if local empty)
# Never resume the discarded rho-excess-10b-scratch-v1
#
# Optional overrides:
#   OFFLINE=1 SKIP_S3_UPLOAD=1 — local smoke only (not durable)
#   VENV — optional; falls back to python3 on PATH
set -euo pipefail

RUN_DIR="${RUN_DIR:-$PWD}"
EDULLM_ROOT="${EDULLM_ROOT:-$RUN_DIR/edullm}"
REPO_ROOT="${REPO_ROOT:-$EDULLM_ROOT/experiments/token-selection}"
FARMSHARE_DIR="${EDULLM_ROOT}/experiments/token-selection/rho-1/farmshare"
VENV="${VENV:-${RUN_DIR}/venv}"

# Discover allocated GPUs (Slurm / CUDA_VISIBLE_DEVICES); default 1.
if [[ -z "${NUM_GPUS:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NUM_GPUS="${SLURM_GPUS_ON_NODE%%(*}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _g <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#_g[@]}"
  else
    NUM_GPUS=1
  fi
fi

export RUN_DIR EDULLM_ROOT REPO_ROOT WORK="$RUN_DIR" VENV
export NUM_GPUS
export RANK_MICROBATCH_SIZE="${RANK_MICROBATCH_SIZE:-16384}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
# Match YAML / RefHQ contract unless explicitly overridden.
export COMPILE_MODEL="${COMPILE_MODEL:-1}"
export RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
  export FROM_SCRATCH=0
else
  export FROM_SCRATCH="${FROM_SCRATCH:-1}"
fi
export TOKEN_SELECTION_SKIP_IDLE_CHECK=1
export OLMO_ATTN_BACKEND="${OLMO_ATTN_BACKEND:-auto}"
# Default online: stage edullm-data + durable S3 export. Opt out explicitly for smoke.
export OFFLINE="${OFFLINE:-0}"
export SKIP_S3_UPLOAD="${SKIP_S3_UPLOAD:-0}"
export SKIP_REF_EXPORT="${SKIP_REF_EXPORT:-0}"
export OLMO_CORE_DIR="${OLMO_CORE_DIR:-$RUN_DIR/OLMo-core}"
export TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT="${TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT:-$FARMSHARE_DIR/enqueue_task_loss.sh}"

mkdir -p "${RUN_DIR}/logs"
cd "${RUN_DIR}"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || module load python/3.13.0 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true

if [[ -x "${VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
elif command -v python3 >/dev/null 2>&1; then
  echo "[$(date -Is)] WARN: no venv at ${VENV}; using $(command -v python3) on PATH" >&2
else
  echo "ERROR: no python3 on PATH and missing venv at ${VENV}" >&2
  echo "  Ephemeral hosts should install deps into a job-local venv or module env." >&2
  exit 2
fi

if [[ "${OFFLINE}" == "1" ]]; then
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE AWS_SESSION_ENV || true
  export AWS_EC2_METADATA_DISABLED=true
fi

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-none} NUM_GPUS=${NUM_GPUS} FROM_SCRATCH=${FROM_SCRATCH} RESUME=${RESUME} OFFLINE=${OFFLINE} SKIP_S3_UPLOAD=${SKIP_S3_UPLOAD}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

bash "${FARMSHARE_DIR}/run_rho_train.sh" prepare
bash "${FARMSHARE_DIR}/run_rho_train.sh" train

echo "[$(date -Is)] sbatch finished"
