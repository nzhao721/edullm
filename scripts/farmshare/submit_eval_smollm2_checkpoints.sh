#!/bin/bash
#SBATCH --job-name=smollm2-eval-ckpts
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --exclude=wheat-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-10%4
#SBATCH --output=/scratch/users/nzhao2/agent-runs/smollm2-135m-500m-40ep-20260730-092920/logs/eval-ckpts-%A_%a.out
#SBATCH --error=/scratch/users/nzhao2/agent-runs/smollm2-135m-500m-40ep-20260730-092920/logs/eval-ckpts-%A_%a.er

set -euo pipefail

RUN_DIR="${RUN_DIR:-/scratch/users/nzhao2/agent-runs/smollm2-135m-500m-40ep-20260730-092920}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/output/checkpoints}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/output/evals}"
STAGING="${STAGING:-/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging}"
VENV="${VENV:-/scratch/users/nzhao2/agent-runs/venvs/smollm2-train}"
TASKS="${TASKS:-HellaSwag}"
N_SHOT="${N_SHOT:-5}"

source "${VENV}/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/users/nzhao2/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${OUT_DIR}" "${RUN_DIR}/logs"

mapfile -t CKPTS < <(ls -1d "${CKPT_DIR}"/step* | sort)
idx="${SLURM_ARRAY_TASK_ID:-0}"
if (( idx < 0 || idx >= ${#CKPTS[@]} )); then
  echo "array index ${idx} out of range (n=${#CKPTS[@]})" >&2
  exit 1
fi

ckpt="${CKPTS[$idx]}"
step="$(basename "${ckpt}")"
out="${OUT_DIR}/${step}_hellaswag.json"

echo "=== $(date -Is) task ${idx}: ${step} ==="
echo "GPU: $(nvidia-smi -L || true)"
echo "tasks: ${TASKS}"

if [[ -f "${out}" && "${FORCE_REEVAL:-0}" != "1" ]]; then
  echo "skip existing ${out}"
  exit 0
fi

python "${STAGING}/scripts/farmshare/eval_arc_task_loss_smollm.py" \
  --checkpoint "${ckpt}" \
  --tokenizer-id "${TOKENIZER_ID:-HuggingFaceTB/SmolLM2-135M}" \
  --out "${out}" \
  --run-name "smollm2-135m-${step}" \
  --n-shot "${N_SHOT}" \
  --tasks ${TASKS}

echo "=== $(date -Is) done ${step} ==="
