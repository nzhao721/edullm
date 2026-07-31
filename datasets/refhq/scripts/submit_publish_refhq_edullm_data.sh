#!/usr/bin/env bash
# Stage RefHQ 5.5B from FarmShare scratch and publish to edullm-landing -> edullm-data.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
REFHQ_ROOT="${REFHQ_ROOT:-/scratch/users/${SUNET}/refhq-regmix-5p5b-v1}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_DIR="${RUN_DIR:-${REUSE_RUN:-/scratch/users/${SUNET}/agent-runs/refhq-edullm-publish-$(date -u +%Y%m%dT%H%M%SZ)}}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"
EDULLM_ROOT="${EDULLM_ROOT:-${STAGING_ROOT}/edullm}"
HQ_SCRIPTS="${HQ_SCRIPTS:-${REFHQ_ROOT}/datasets/refhq/scripts}"

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

mkdir -p "${RUN_DIR}/logs"
cd "${RUN_DIR}"

if [[ -d "${REFHQ_ROOT}/venv" && ! -e "${RUN_DIR}/venv" ]]; then
  ln -s "${REFHQ_ROOT}/venv" "${RUN_DIR}/venv"
fi
if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
  pip install -U pip wheel
  pip install boto3
else
  # shellcheck disable=SC1091
  source "${RUN_DIR}/venv/bin/activate"
fi

dolma_hq_sync_to_run "${STAGING_ROOT}" "${RUN_DIR}"
dolma_hq_export_pythonpath "${RUN_DIR}"
mkdir -p "${RUN_DIR}/scripts"
cp -a "${STAGING_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/"

export EDULLM_ROOT="${RUN_DIR}"
export RUN_DIR REFHQ_ROOT STAGE_DIR HQ_SCRIPTS
# shellcheck disable=SC1091
source "${RUN_DIR}/scripts/farmshare/prepare_aws_session_light.sh" || {
  echo "ERROR: could not mint AWS session for edullm-landing writes" >&2
  exit 1
}

JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",REFHQ_ROOT="${REFHQ_ROOT}",STAGE_DIR="${STAGE_DIR}",EDULLM_ROOT="${EDULLM_ROOT}",HQ_SCRIPTS="${HQ_SCRIPTS}",AWS_SESSION_ENV="${AWS_SESSION_ENV}",REUSE_RUN="${REUSE_RUN:-}" \
  "${HQ_SCRIPTS}/publish_refhq_edullm_data.sbatch")
echo "publish_job=${JOB}"
echo "run_dir=${RUN_DIR}"
echo "stage_dir=${STAGE_DIR}"
echo "target_dataset=pretrain/refhq-regmix-5p5b"
