#!/usr/bin/env bash
# Upload text corpus (if needed), dolma2-tokenize all domains, upload tokenized/ to S3.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/refhq-regmix-5p5b-v1}"
STAGING_ROOT="${STAGING_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
DOMAIN_LIST="${DOMAIN_LIST:-dclm starcoder pes2o arxiv open-web-math algebraic-stack wiki}"
PLAN="${PLAN:-${RUN_DIR}/manifests/plan.json}"
VENV="${VENV:-${RUN_DIR}/venv}"
HQ_SCRIPTS="${RUN_DIR}/datasets/refhq/scripts"

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tokenized"
cd "${RUN_DIR}"

dolma_hq_stage_shared_utils "${STAGING_ROOT}" "${RUN_DIR}"
if [[ ! -d "${RUN_DIR}/datasets/refhq" ]]; then
  mkdir -p "${RUN_DIR}/datasets"
  cp -a "${STAGING_ROOT}/datasets/refhq" "${RUN_DIR}/datasets/"
fi
sed -i 's/\r$//' "${RUN_DIR}/datasets/refhq/scripts/"*.sh "${RUN_DIR}/datasets/refhq/scripts/"*.sbatch 2>/dev/null || true

# shellcheck disable=SC1091
source "${VENV}/bin/activate" 2>/dev/null || true

export RUN_DIR VENV PLAN DOMAIN_LIST HQ_SCRIPTS
dolma_hq_export_pythonpath "${RUN_DIR}"

read -r -a DOMAINS <<< "${DOMAIN_LIST}"
N=${#DOMAINS[@]}

UPLOAD_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",PLAN="${PLAN}",HQ_SCRIPTS="${HQ_SCRIPTS}" \
  "${HQ_SCRIPTS}/finalize_hq_reference_upload.sbatch")
echo "text_upload_job=${UPLOAD_JOB}"

TOK_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${UPLOAD_JOB} \
  --array=0-$((N - 1)) \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",PLAN="${PLAN}",DOMAIN_LIST="${DOMAIN_LIST}",HQ_SCRIPTS="${HQ_SCRIPTS}" \
  "${HQ_SCRIPTS}/tokenize_hq_reference_domain.sbatch")
echo "tokenize_job=${TOK_JOB}"

TOK_UP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --dependency=afterok:${TOK_JOB} \
  --chdir="${RUN_DIR}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",VENV="${VENV}",PLAN="${PLAN}",HQ_SCRIPTS="${HQ_SCRIPTS}" \
  "${HQ_SCRIPTS}/finalize_refhq_tokenized_upload.sbatch")
echo "tokenized_upload_job=${TOK_UP_JOB}"

S3_BUCKET=$(python3 -c "import json; print(json.load(open('${PLAN}'))['s3_bucket'])")
S3_PREFIX=$(python3 -c "import json; print(json.load(open('${PLAN}'))['s3_prefix'])")
echo "text s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "tokenized s3://${S3_BUCKET}/${S3_PREFIX}/tokenized/"
