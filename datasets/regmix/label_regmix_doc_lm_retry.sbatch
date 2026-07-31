#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-lm-retry
#SBATCH --output=logs/lm-retry-%A_%a.out
#SBATCH --error=logs/lm-retry-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${WORK_MANIFEST:?}" "${LM_LABELS_ROOT:?}"
: "${EARLY_CHECKPOINT:?}" "${LATE_AVG_CHECKPOINT:?}" "${BASE_CONFIG:?}" "${RETRY_INDICES:?}"

cd "${RUN_DIR}"
source "${VENV}/bin/activate"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=1
export WANDB_MODE=disabled
export HF_HOME="${HF_HOME:-${RUN_DIR}/hf_cache}"
mkdir -p "${RUN_DIR}/logs" "${LM_LABELS_ROOT}" "${HF_HOME}"
export PYTHONPATH="${RUN_DIR}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

CHUNK_INDEX="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${RETRY_INDICES}")"
if [[ -z "${CHUNK_INDEX}" ]]; then
  echo "empty retry index for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

python "${RUN_DIR}/scripts/label_regmix_doc_lm.py" \
  --work-manifest "${WORK_MANIFEST}" \
  --index "${CHUNK_INDEX}" \
  --out-root "${LM_LABELS_ROOT}" \
  --early-checkpoint "${EARLY_CHECKPOINT}" \
  --late-avg-checkpoint "${LATE_AVG_CHECKPOINT}" \
  --base-config "${BASE_CONFIG}" \
  --seq-len "${SEQ_LEN:-2048}" \
  --batch-tokens "${BATCH_TOKENS:-4096}"
