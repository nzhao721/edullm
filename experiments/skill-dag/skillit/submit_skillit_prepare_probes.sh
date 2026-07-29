#!/usr/bin/env bash
# Materialize 8 Skill-It probe slices on FarmShare (CPU only; no GPUs).
#
# Reuses the olmohq working pool from the completed mixlaw validation run:
#   /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/pool/tokenized
#
# Usage (on a FarmShare login node):
#   bash experiments/skill-dag/skillit/submit_skillit_prepare_probes.sh
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-skillit-probes-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
MIXLAW_POOL_RUN="${MIXLAW_POOL_RUN:-/scratch/users/${SUNET}/agent-runs/mixlaw-validation-10b-20260728-190236}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${MIXLAW_POOL_RUN}/pool/tokenized}"
BUILD_WORKERS="${BUILD_WORKERS:-8}"
PROBE_ARTIFACTS_S3="${PROBE_ARTIFACTS_S3:-s3://edullm-datasets/skillit/probes}"

SKILLIT_ROOT="${EDULLM_ROOT}/experiments/skill-dag/skillit"
MIXLAW_ROOT="${EDULLM_ROOT}/experiments/skill-dag/mixlaw"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/slices"

if [[ ! -d "${TOKENIZED_DIR}/dclm" ]]; then
  echo "missing tokenized pool at ${TOKENIZED_DIR}" >&2
  exit 2
fi
if [[ ! -f "${SKILLIT_ROOT}/prepare_probes.sh" ]]; then
  echo "missing ${SKILLIT_ROOT}/prepare_probes.sh (sync repo to ${EDULLM_ROOT} first)" >&2
  exit 2
fi

# Reuse the mixlaw validation venv when present; otherwise create a minimal one.
VENV="${VENV:-${MIXLAW_POOL_RUN}/venv}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
  VENV="${RUN_DIR}/venv"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install -U pip wheel
  pip install boto3 tqdm numpy
else
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

AWS_SESSION_ENV="${AWS_SESSION_ENV:-/scratch/users/${SUNET}/agent-runs/aws-session.env}"
if [[ -f "${AWS_SESSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${VENV}
EDULLM_ROOT=${EDULLM_ROOT}
SKILLIT_ROOT=${SKILLIT_ROOT}
MIXLAW_ROOT=${MIXLAW_ROOT}
TOKENIZED_DIR=${TOKENIZED_DIR}
OUT_DIR=${RUN_DIR}/slices
AWS_SESSION_ENV=${AWS_SESSION_ENV}
PROBE_ARTIFACTS_S3=${PROBE_ARTIFACTS_S3}
BUILD_WORKERS=${BUILD_WORKERS}
EOF

_AWS_PATH='export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"'

PREP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task="${BUILD_WORKERS}" \
  --mem=64G \
  --time=04:00:00 \
  --job-name=skillit-prep \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/prep-%j.out" \
  --error="${RUN_DIR}/logs/prep-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; ${_AWS_PATH}; source ${RUN_DIR}/env.sh; [[ -f \${AWS_SESSION_ENV} ]] && source \${AWS_SESSION_ENV}; source \${VENV}/bin/activate; export SKILLIT_PROBE_WORK=${RUN_DIR} TOKENIZED_DIR OUT_DIR BUILD_WORKERS=\${BUILD_WORKERS} PROBE_ARTIFACTS_S3; bash ${SKILLIT_ROOT}/prepare_probes.sh'")

echo "prep_job_id=${PREP_JOB}"
echo "${PREP_JOB}" > "${RUN_DIR}/prep_job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "TOKENIZED_DIR=${TOKENIZED_DIR}"
echo "submitted skillit-prep=${PREP_JOB}"
