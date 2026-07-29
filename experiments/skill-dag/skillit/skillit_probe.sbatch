#!/usr/bin/env bash
#SBATCH --job-name=skillit-probe
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --exclude=wheat-01
#SBATCH --output=logs/probe-%A_%a.out
#SBATCH --error=logs/probe-%A_%a.err
#
# One array task → one Slurm allocation → one GPU → one probe.
# Request only 64G RAM (not --mem=0) so Slurm can place each task as soon as
# a single GPU is free, including sharing a multi-GPU node with other tasks.
# Submit via submit_skillit_probes.sh (sets --array and RUN_DIR).
set -euo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${SKILLIT_ROOT:?SKILLIT_ROOT is required}"
: "${MIXLAW_ROOT:?MIXLAW_ROOT is required}"
: "${LADDER_RUN:?LADDER_RUN is required}"

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
PROBE_ID="$(sed -n "$((IDX + 1))p" "${RUN_DIR}/probe_ids.txt")"
if [[ -z "${PROBE_ID}" ]]; then
  echo "no probe for array index ${IDX} in ${RUN_DIR}/probe_ids.txt" >&2
  exit 2
fi

VENV="${VENV:-${LADDER_RUN}/venv}"
PATHS_FILE="${RUN_DIR}/slices/${PROBE_ID}/paths_train.txt"
SAVE_FOLDER="${RUN_DIR}/runs/${PROBE_ID}/checkpoints"
PROGRESS_DIR="${RUN_DIR}/runs/${PROBE_ID}/progress"

mkdir -p "${RUN_DIR}/logs" "${SAVE_FOLDER}" "${PROGRESS_DIR}"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || module load python/3.13.0 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv at ${VENV}" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ -f "${LADDER_RUN}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${LADDER_RUN}/env.sh"
fi
export OLMO_LADDER_ROOT="${OLMO_LADDER_ROOT:-${LADDER_RUN}/OLMo-ladder}"
export HF_HOME="${HF_HOME:-${LADDER_RUN}/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export WANDB_DISABLED=1
export WANDB_MODE=disabled
# Probe contract: SDPA only (ladder env.sh sets flash=1; broken flash-attn wheel on L40S).
export OLMO_FLASH_ATTENTION=0
# fused_loss requires flash_attn; keep off and use small microbatch on 48G L40S.
export OLMO_FUSED_LOSS=0
export OLMO_ACTIVATION_CHECKPOINTING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${MIXLAW_ROOT}:${PYTHONPATH:-}"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID} array=${IDX} probe=${PROBE_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true

export PROBE_ID SAVE_FOLDER PATHS_FILE PROGRESS_DIR
export PYTHON="${VENV}/bin/python"
export NPROC=1
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
export DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RESULTS_S3="${RESULTS_S3:-}"

bash "${SKILLIT_ROOT}/launch_probe.sh"

echo "[$(date -Is)] ${PROBE_ID} finished"
