#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo-up
#SBATCH --output=logs/upload-%A_%a.out
#SBATCH --error=logs/upload-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${MANIFEST:?}" "${LOCAL_ROOT:?}"
: "${BUCKET:?}" "${PREFIX:?}" "${AWS_SESSION_ENV:?}"

# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
source "${VENV}/bin/activate"

python "${RUN_DIR}/scripts/upload_olmo_shard.py" \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --local-root "${LOCAL_ROOT}" \
  --bucket "${BUCKET}" \
  --prefix "${PREFIX}"
