#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo127b-edullm-pub
#SBATCH --output=logs/olmo127b_edullm_publish_%j.out
#SBATCH --error=logs/olmo127b_edullm_publish_%j.e

# Skip restage; publish already-complete publish-stage → edullm-landing.
# Uses RefreshingBoto3S3 + laptop-pushed aws-session.env for multi-hour STS.

set -Eeuo pipefail

: "${RUN_DIR:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}"

VENV="${VENV:-${RUN_DIR}/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${PATH}:${HOME}/.local/bin:${HOME}/tools/aws/bin"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

SCRIPTS="${SCRIPTS:-${EDULLM_ROOT}/datasets/olmohq}"
BUCKET="${BUCKET:-edullm-datasets}"
PREFIX="${PREFIX:-olmo100b/olmo-mix-1124-30b}"
TEXT_RUN_DIR="${TEXT_RUN_DIR:-/scratch/users/nzhao2/agent-runs/olmo-mix-upsample-20260723-103547}"
CACHE_DIR="${CACHE_DIR:-${RUN_DIR}/cache}"
MANIFEST_LOCAL="${MANIFEST_LOCAL:-${CACHE_DIR}/plan/tokenized_manifest.json}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
PURPOSE="${PURPOSE:-Default reservoir corpus mirroring olmo-mix-1124 for sampling training mixes}"

mkdir -p "${CACHE_DIR}/plan" "${RUN_DIR}/logs"

echo "[$(date -u +%FT%TZ)] caller=$(aws sts get-caller-identity --query Arn --output text)" >&2
if [[ ! -f "${MANIFEST_LOCAL}" ]]; then
  aws s3 cp "s3://${BUCKET}/${PREFIX}/plan/tokenized_manifest.json" "${MANIFEST_LOCAL}"
fi

if [[ "${SKIP_EDULLM_DATA_INSTALL:-0}" != 1 && ! -d "${EDULLM_DATA_DIR}/src/edullm_data" ]]; then
  rm -rf "${EDULLM_DATA_DIR}"
  git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
fi
if [[ "${SKIP_EDULLM_DATA_INSTALL:-0}" != 1 ]]; then
  pip install -q -e "${EDULLM_DATA_DIR}"
fi

echo "[$(date -u +%FT%TZ)] skip-stage publish from ${STAGE_DIR}" >&2
python "${SCRIPTS}/publish_olmohq_edullm_data.py" \
  --manifest "${MANIFEST_LOCAL}" \
  --s3-bucket "${BUCKET}" \
  --s3-tokenized-prefix "${PREFIX}/tokenized" \
  --text-run-dir "${TEXT_RUN_DIR}" \
  --stage-dir "${STAGE_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --purpose "${PURPOSE}" \
  --shard-bytes 1073741824 \
  --val-fraction 0.0015 \
  --hash-workers 16 \
  --copy-workers 16 \
  --skip-stage

echo "[$(date -u +%FT%TZ)] done" >&2
