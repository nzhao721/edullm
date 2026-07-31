#!/usr/bin/env bash
# Stage RegMix 10B from FarmShare scratch and publish to edullm-landing -> edullm-data.
#
# Layout: tokens/<source>/ only (labels.source only; domain omitted).
# Shards: max 1 GiB. Val: same fraction from every source (weights match full mix).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
REGMIX_ROOT="${REGMIX_ROOT:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-edullm-publish-$(date -u +%Y%m%dT%H%M%SZ)}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts/regmix" "${RUN_DIR}/scripts/farmshare"
cd "${RUN_DIR}"

if [[ -d "${REGMIX_ROOT}/venv" && ! -e "${RUN_DIR}/venv" ]]; then
  ln -s "${REGMIX_ROOT}/venv" "${RUN_DIR}/venv"
fi
if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
  pip install -U pip wheel boto3
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
fi

cp -a "${STAGING_ROOT}/datasets/regmix/publish_regmix_edullm_data.py" "${RUN_DIR}/scripts/regmix/"
cp -a "${STAGING_ROOT}/datasets/regmix/publish_regmix_edullm_data.sbatch" "${RUN_DIR}/scripts/regmix/"
cp -a "${STAGING_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/farmshare/"
cp -a "${STAGING_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/"
sed -i 's/\r$//' \
  "${RUN_DIR}/scripts/regmix/"*.py \
  "${RUN_DIR}/scripts/regmix/"*.sbatch \
  "${RUN_DIR}/scripts/farmshare/"*.sh \
  2>/dev/null || true

export EDULLM_ROOT="${RUN_DIR}"
export RUN_DIR REGMIX_ROOT STAGE_DIR
# shellcheck disable=SC1091
source "${RUN_DIR}/scripts/farmshare/prepare_aws_session_light.sh" || {
  echo "ERROR: could not mint AWS session for edullm-landing writes" >&2
  exit 1
}

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",REGMIX_ROOT="${REGMIX_ROOT}",STAGE_DIR="${STAGE_DIR}",EDULLM_ROOT="${EDULLM_ROOT}",SCRIPTS="${RUN_DIR}/scripts/regmix",TOKENIZED_ROOT="${REGMIX_ROOT}/tokenized",AWS_SESSION_ENV="${AWS_SESSION_ENV}" \
  "${RUN_DIR}/scripts/regmix/publish_regmix_edullm_data.sbatch")
echo "publish_job=${JOB}"
echo "run_dir=${RUN_DIR}"
echo "stage_dir=${STAGE_DIR}"
echo "target_dataset=pretrain/regmix-10b"
echo "constraints: source-only labels; max 1GiB shards; val=0.15% per source"
