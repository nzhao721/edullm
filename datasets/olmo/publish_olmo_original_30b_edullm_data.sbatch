#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=olmo30b-edullm-pub
#SBATCH --output=logs/olmo30b_edullm_publish_%j.out
#SBATCH --error=logs/olmo30b_edullm_publish_%j.e

# FarmShare-only publish of olmo30b → pretrain/olmo-original-30b.
# Streams from s3://edullm-datasets on this compute node (no laptop data path).
# Layout: tokens/<source>/ (source labels, domain omitted); max 1 GiB; val=0.15%/source.

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
PREFIX="${PREFIX:-olmo30b/olmo-mix-1124-30b}"
TEXT_RUN_DIR="${TEXT_RUN_DIR:-/scratch/users/nzhao2/agent-runs/olmo-mix-30b-20260722}"
CACHE_DIR="${CACHE_DIR:-${RUN_DIR}/cache}"
MANIFEST_LOCAL="${MANIFEST_LOCAL:-${CACHE_DIR}/plan/tokenized_manifest.json}"
EDULLM_DATA_DIR="${EDULLM_DATA_DIR:-${RUN_DIR}/edullm-data}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-original-30b}"
PURPOSE="${PURPOSE:-30B dataset with the same domain weights as olmo-mix-1124}"

mkdir -p "${CACHE_DIR}/plan" "${RUN_DIR}/logs"

echo "[$(date -u +%FT%TZ)] caller=$(aws sts get-caller-identity --query Arn --output text)" >&2
echo "[$(date -u +%FT%TZ)] fetching tokenized_manifest.json on FarmShare node" >&2
aws s3 cp "s3://${BUCKET}/${PREFIX}/plan/tokenized_manifest.json" "${MANIFEST_LOCAL}"

rm -rf "${EDULLM_DATA_DIR}"
git clone --depth 1 --branch main https://github.com/edu-llm/edullm-data.git "${EDULLM_DATA_DIR}"
pip install -q -e "${EDULLM_DATA_DIR}"

echo "[$(date -u +%FT%TZ)] stream-stage + publish ${DATASET_ID} from s3://${BUCKET}/${PREFIX}/tokenized/" >&2
python "${SCRIPTS}/publish_olmohq_edullm_data.py" \
  --manifest "${MANIFEST_LOCAL}" \
  --s3-bucket "${BUCKET}" \
  --s3-tokenized-prefix "${PREFIX}/tokenized" \
  --text-run-dir "${TEXT_RUN_DIR:?}" \
  --stage-dir "${STAGE_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --purpose "${PURPOSE}" \
  --about "30B-token dolma2 corpus with the same domain weights as allenai/olmo-mix-1124 across seven sources (dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki). Shards nest as tokens/<source>/ (labels.source only; domain omitted)." \
  --legacy-uri "s3://${BUCKET}/${PREFIX}/" \
  --shard-bytes 1073741824 \
  --val-fraction 0.0015 \
  --hash-workers 16 \
  --copy-workers 16 \
  --force

echo "[$(date -u +%FT%TZ)] done" >&2
