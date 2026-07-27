#!/usr/bin/env bash
#SBATCH --job-name=olmo-ckpt-s3
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/upload_ckpts_%j.out
#SBATCH --error=logs/upload_ckpts_%j.err
#SBATCH --exclude=wheat-01
set -euo pipefail

: "${RUN_DIR:?}"
: "${BUCKET:=edullm-olmo2-370m-cpt-checkpoints}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
: "${AWS_SESSION_ENV:?}"

export PATH="${HOME}/tools/aws/bin:${HOME}/.local/bin:${PATH}"
set -a
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
set +a
unset AWS_PROFILE

CKPT="${RUN_DIR}/checkpoints/edullm-370M-30B"
PREFIX=edullm-370M-30B

echo "[$(date -Is)] upload start bucket=s3://${BUCKET}/${PREFIX}/"
aws sts get-caller-identity --output json

for step in 5000 10000 15000; do
  src="${CKPT}/step${step}-unsharded"
  dest="s3://${BUCKET}/${PREFIX}/step${step}-unsharded/"
  if [[ ! -d "${src}" ]]; then
    echo "missing ${src}" >&2
    exit 1
  fi
  echo "[$(date -Is)] sync ${src} -> ${dest} ($(du -sh "${src}" | awk '{print $1}'))"
  aws s3 sync "${src}" "${dest}" --region "${AWS_DEFAULT_REGION}" --only-show-errors
  echo "[$(date -Is)] done step ${step}"
done

echo "[$(date -Is)] verify"
aws s3 ls "s3://${BUCKET}/${PREFIX}/" --region "${AWS_DEFAULT_REGION}"
aws s3 ls "s3://${BUCKET}/${PREFIX}/" --recursive --human-readable --summarize --region "${AWS_DEFAULT_REGION}" | tail -5
echo "[$(date -Is)] upload complete"
