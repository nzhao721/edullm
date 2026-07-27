#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=regmix-dl
#SBATCH --output=logs/download-%A_%a.out
#SBATCH --error=logs/download-%A_%a.err

set -Eeuo pipefail

: "${RUN_DIR:?}" "${VENV:?}" "${MANIFEST:?}" "${LOCAL_ROOT:?}"
: "${SRC_BUCKET:?}"

# AWS session only required when falling back to S3 downloads.
SHARED_AWS_SESSION="/scratch/users/nzhao2/agent-runs/_shared/aws-session-dolma2-tok.env"
if [[ -f "${SHARED_AWS_SESSION}" ]]; then
  # shellcheck disable=SC1090
  source "${SHARED_AWS_SESSION}"
elif [[ -n "${AWS_SESSION_ENV:-}" && -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

mkdir -p "${LOCAL_ROOT}"

python "${RUN_DIR}/scripts/download_s3_shard.py" \
  --manifest "${MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --local-root "${LOCAL_ROOT}" \
  --src-bucket "${SRC_BUCKET}" \
  --local-mirror "${LOCAL_MIRROR:-}"
