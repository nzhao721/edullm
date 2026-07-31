#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo-s3
#SBATCH --output=logs/finalize-%j.out
#SBATCH --error=logs/finalize-%j.er

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${VENV:?}"
: "${AWS_SESSION_ENV:?}"
: "${BUCKET:?}"
: "${S3_PREFIX:?}"

# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
unset PREFIX
source "${VENV}/bin/activate"
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

python "${RUN_DIR}/scripts/finalize_olmo_upsample_upload.py" \
  --run-dir "${RUN_DIR}" \
  --bucket "${BUCKET}" \
  --prefix "${S3_PREFIX}"
