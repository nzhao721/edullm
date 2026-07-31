#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo127b-edullm-pub
#SBATCH --output=logs/olmo127b_edullm_publish_%j.out
#SBATCH --error=logs/olmo127b_edullm_publish_%j.e

# FarmShare-only publish of olmohq → pretrain/olmo-127b.
# Streams tokenized shards from s3://edullm-datasets on this compute node
# (no laptop data path). Stages tokens/<source>/ (source labels, domain omitted),
# max 1 GiB shards, val=0.15% per source, then publish() → edullm-landing.

set -Eeuo pipefail

: "${RUN_DIR:?}" "${STAGE_DIR:?}" "${EDULLM_ROOT:?}"

# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV:?}"

export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

SCRIPTS="${SCRIPTS:-${RUN_DIR}/scripts/olmohq}"
BUCKET="${BUCKET:-edullm-datasets}"
PREFIX="${PREFIX:-olmo100b/olmo-mix-1124-30b}"
CACHE_DIR="${CACHE_DIR:-${RUN_DIR}/cache}"
MANIFEST_LOCAL="${MANIFEST_LOCAL:-${CACHE_DIR}/plan/tokenized_manifest.json}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"

mkdir -p "${CACHE_DIR}/plan" "${RUN_DIR}/logs"

echo "[$(date -u +%FT%TZ)] caller=$(aws sts get-caller-identity --query Arn --output text)" >&2
echo "[$(date -u +%FT%TZ)] fetching active tokenized_manifest.json on FarmShare node" >&2
aws s3 cp "s3://${BUCKET}/${PREFIX}/plan/tokenized_manifest.json" "${MANIFEST_LOCAL}"

rm -rf "${EDULLM_DATA_DIR}"
git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
pip install -q -e "${EDULLM_DATA_DIR}"

echo "[$(date -u +%FT%TZ)] stream-stage + publish pretrain/olmo-127b from s3://${BUCKET}/${PREFIX}/tokenized/" >&2
python "${SCRIPTS}/publish_olmohq_edullm_data.py" \
  --manifest "${MANIFEST_LOCAL}" \
  --s3-bucket "${BUCKET}" \
  --s3-tokenized-prefix "${PREFIX}/tokenized" \
  --stage-dir "${STAGE_DIR}" \
  --dataset-id pretrain/olmo-127b \
  --shard-bytes 1073741824 \
  --val-fraction 0.0015 \
  --hash-workers 16 \
  --copy-workers 16 \
  --force

echo "[$(date -u +%FT%TZ)] done" >&2
