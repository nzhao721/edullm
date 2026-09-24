#!/usr/bin/env bash
# Stage olmohq (~127B) on FarmShare and publish to edullm-landing -> edullm-data.
# All S3 I/O happens on the Slurm node (not the laptop).
#
# Layout: tokens/<source>/ only (labels.source only; domain omitted).
# Shards: max 1 GiB. Val: same fraction from every source.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/olmo127b-edullm-publish-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts/olmohq"
cd "${RUN_DIR}"

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
  pip install -U pip wheel boto3
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
fi

cp -a "${STAGING_ROOT}/datasets/olmohq/publish_olmohq_edullm_data.py" "${RUN_DIR}/scripts/olmohq/"
cp -a "${STAGING_ROOT}/datasets/olmohq/publish_olmohq_edullm_data.sbatch" "${RUN_DIR}/scripts/olmohq/"
sed -i 's/\r$//' \
  "${RUN_DIR}/scripts/olmohq/"*.py \
  "${RUN_DIR}/scripts/olmohq/"*.sbatch \
  2>/dev/null || true

export EDULLM_ROOT="${RUN_DIR}"
export RUN_DIR STAGE_DIR

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",STAGE_DIR="${STAGE_DIR}",EDULLM_ROOT="${EDULLM_ROOT}",SCRIPTS="${RUN_DIR}/scripts/olmohq" \
  "${RUN_DIR}/scripts/olmohq/publish_olmohq_edullm_data.sbatch")
echo "publish_job=${JOB}"
echo "run_dir=${RUN_DIR}"
echo "stage_dir=${STAGE_DIR}"
echo "target_dataset=pretrain/olmo-127b"
echo "constraints: source-only labels; max 1GiB shards; val=0.15% per source"
echo "io: FarmShare Slurm node syncs from s3://edullm-datasets then publishes to landing"
