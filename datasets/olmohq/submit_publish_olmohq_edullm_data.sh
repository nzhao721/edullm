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

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts/olmohq" "${RUN_DIR}/scripts/farmshare"
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
cp -a "${STAGING_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/farmshare/"
cp -a "${STAGING_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/"
sed -i 's/\r$//' \
  "${RUN_DIR}/scripts/olmohq/"*.py \
  "${RUN_DIR}/scripts/olmohq/"*.sbatch \
  "${RUN_DIR}/scripts/farmshare/"*.sh \
  2>/dev/null || true

export EDULLM_ROOT="${RUN_DIR}"
export RUN_DIR STAGE_DIR
# shellcheck disable=SC1091
source "${RUN_DIR}/scripts/farmshare/prepare_aws_session_light.sh" || {
  echo "ERROR: could not mint AWS session for edullm-datasets read / edullm-landing write" >&2
  exit 1
}

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",STAGE_DIR="${STAGE_DIR}",EDULLM_ROOT="${EDULLM_ROOT}",SCRIPTS="${RUN_DIR}/scripts/olmohq",AWS_SESSION_ENV="${AWS_SESSION_ENV}" \
  "${RUN_DIR}/scripts/olmohq/publish_olmohq_edullm_data.sbatch")
echo "publish_job=${JOB}"
echo "run_dir=${RUN_DIR}"
echo "stage_dir=${STAGE_DIR}"
echo "target_dataset=pretrain/olmo-127b"
echo "constraints: source-only labels; max 1GiB shards; val=0.15% per source"
echo "io: FarmShare Slurm node syncs from s3://edullm-datasets then publishes to landing"
