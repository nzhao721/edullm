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
#SBATCH --output=logs/probe-%A_%a.out
#SBATCH --error=logs/probe-%A_%a.err
#
# One array task → one Slurm allocation → one GPU → one probe.
# Request only 64G RAM (not --mem=0) so Slurm can place each task as soon as
# a single GPU is free, including sharing a multi-GPU node with other tasks.
# Submit via submit_skillit_probes.sh (sets --array and RUN_DIR).
#
# Requires TRAIN_VENV (or VENV) — a prebuilt GPU env with torch + olmo-core.
# Does not assume a persistent ladder scratch tree.
set -euo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${SKILLIT_ROOT:?SKILLIT_ROOT is required}"
: "${MIXLAW_ROOT:?MIXLAW_ROOT is required}"
: "${POOL_DIR:?POOL_DIR is required}"
: "${RECIPE_WORK:?RECIPE_WORK is required}"

TRAIN_VENV="${TRAIN_VENV:-${VENV:-}}"
if [[ -z "${TRAIN_VENV}" ]]; then
  echo "TRAIN_VENV (or VENV) is required — no hardcoded ladder path" >&2
  exit 2
fi

IDX="${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}"
PROBE_ID="$(sed -n "$((IDX + 1))p" "${RUN_DIR}/probe_ids.txt")"
if [[ -z "${PROBE_ID}" ]]; then
  echo "no probe for array index ${IDX} in ${RUN_DIR}/probe_ids.txt" >&2
  exit 2
fi

MIX_WEIGHTS_JSON="${RECIPE_WORK}/${PROBE_ID}/mix_weights.json"
SAVE_FOLDER="${RUN_DIR}/runs/${PROBE_ID}/checkpoints"
PROGRESS_DIR="${RUN_DIR}/runs/${PROBE_ID}/progress"

mkdir -p "${RUN_DIR}/logs" "${SAVE_FOLDER}" "${PROGRESS_DIR}"

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || module load python/3.13.0 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true

if [[ ! -x "${TRAIN_VENV}/bin/python" ]]; then
  echo "missing venv at ${TRAIN_VENV}" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${TRAIN_VENV}/bin/activate"

# Ephemeral caches under this run (not a shared ladder tree).
export HF_HOME="${HF_HOME:-${RUN_DIR}/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
# Mixlaw trainer stays W&B-quiet; launch_probe logs final eval when enabled.
export WANDB_DISABLED=1
# Probe contract: SDPA only (broken flash-attn wheel on L40S).
export OLMO_FLASH_ATTENTION=0
export OLMO_FUSED_LOSS=0
export OLMO_ACTIVATION_CHECKPOINTING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# mixlaw first so DomainMixtureStream comes from the shared sampler.
export PYTHONPATH="${MIXLAW_ROOT}:${SKILLIT_ROOT}:${PYTHONPATH:-}"

set +u
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/wandb-session.env"
fi
set -u
export WANDB_PROJECT="${WANDB_PROJECT:-skillit}"
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_MODE="online"
  else
    WANDB_MODE="disabled"
  fi
fi
export WANDB_MODE

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID} array=${IDX} probe=${PROBE_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "WANDB_PROJECT=${WANDB_PROJECT} WANDB_MODE=${WANDB_MODE}"
nvidia-smi -L || true

export PROBE_ID POOL_DIR MIX_WEIGHTS_JSON SAVE_FOLDER PROGRESS_DIR
export PYTHON="${TRAIN_VENV}/bin/python"
export NPROC=1
# Match mixlaw / launch_probe.sh defaults (32/32) so probe family losses for A align.
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
export DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ALLOW_LOCAL_ONLY="${ALLOW_LOCAL_ONLY:-0}"

bash "${SKILLIT_ROOT}/launch_probe.sh"

echo "[$(date -Is)] ${PROBE_ID} finished"
