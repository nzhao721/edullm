#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo-dl
#SBATCH --output=logs/download-%A_%a.out
#SBATCH --error=logs/download-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${MANIFEST:?}" "${LOCAL_ROOT:?}"

source "${VENV}/bin/activate"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${RUN_DIR}/hf-cache"
mkdir -p "${HF_HOME}" "${LOCAL_ROOT}"

python "${RUN_DIR}/scripts/download_olmo_shard.py" \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --local-root "${LOCAL_ROOT}"
