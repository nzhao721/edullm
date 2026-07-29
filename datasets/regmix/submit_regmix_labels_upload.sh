#!/usr/bin/env bash
# Upload RegMix heuristic labels/ + LM lm_labels/ trees to S3 (FarmShare).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-10b-20260725-124810}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
DST_BUCKET="${DST_BUCKET:-edullm-datasets}"
DST_PREFIX="${DST_PREFIX:-regmix/regmix-10b}"
VENV="${VENV:-${RUN_DIR}/venv}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_LABELS="${SKIP_LABELS:-0}"
SKIP_LM_LABELS="${SKIP_LM_LABELS:-0}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs"
cd "${RUN_DIR}"

REGMIX_ROOT="${EDULLM_ROOT}/datasets/regmix"
cp -a "${REGMIX_ROOT}/finalize_regmix_labels_upload.py" "${RUN_DIR}/scripts/"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv at ${VENV}; create one in the RegMix run dir first" >&2
  exit 1
fi

# Mint AWS session env for compute nodes (broker profile on login node).
export EDULLM_ROOT RUN_DIR
# shellcheck disable=SC1091
source "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"

EXTRA_FLAGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_FLAGS+=(--dry-run)
fi
if [[ "${SKIP_LABELS}" == "1" ]]; then
  EXTRA_FLAGS+=(--skip-labels)
fi
if [[ "${SKIP_LM_LABELS}" == "1" ]]; then
  EXTRA_FLAGS+=(--skip-lm-labels)
fi
# shellcheck disable=SC2145
EXTRA_FLAGS_STR="${EXTRA_FLAGS[*]:-}"

cat > "${RUN_DIR}/env_labels_upload.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${VENV}
DST_BUCKET=${DST_BUCKET}
DST_PREFIX=${DST_PREFIX}
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
EOF

# Refresh AWS session just before upload via wrap that re-sources prepare_aws_session.
UPLOAD_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=8 \
  --mem=32G \
  --time=06:00:00 \
  --job-name=regmix-labels-up \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/labels-upload-%j.out" \
  --error="${RUN_DIR}/logs/labels-upload-%j.err" \
  --wrap="set -Eeuo pipefail; source ${RUN_DIR}/env_labels_upload.sh; export EDULLM_ROOT RUN_DIR; source ${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh; source \${AWS_SESSION_ENV}; source \${VENV}/bin/activate; python ${RUN_DIR}/scripts/finalize_regmix_labels_upload.py --run-dir \${RUN_DIR} --dst-bucket \${DST_BUCKET} --dst-prefix \${DST_PREFIX} ${EXTRA_FLAGS_STR}")
echo "upload_job_id=${UPLOAD_JOB}"
echo "${UPLOAD_JOB}" > "${RUN_DIR}/labels_upload_job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "dst=s3://${DST_BUCKET}/${DST_PREFIX}/"
echo "submitted labels upload=${UPLOAD_JOB}"
