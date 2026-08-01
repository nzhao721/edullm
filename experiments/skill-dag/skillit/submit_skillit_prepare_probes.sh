#!/usr/bin/env bash
# Build edullm-data working pool + probe recipe sidecars for Skill-It 60M probes.
#
# Stages from published s3://edullm-data (default pretrain/olmo-127b) unless
# POOL_DIR already has edullm_data_source.json. Never reads s3://edullm-datasets/.
# Bootstraps a CPU venv in RUN_DIR — no persistent ladder/scratch pool assumed.
#
# Usage (FarmShare login node):
#   bash experiments/skill-dag/skillit/submit_skillit_prepare_probes.sh
set -Eeuo pipefail

SUNET="${SUNET:-${USER:?set SUNET or USER}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIXLAW_ROOT="$(cd "${SCRIPT_DIR}/../mixlaw" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-${REPO_ROOT}}"

RUN_NAME="${RUN_NAME:-skillit-probes-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
POOL_DIR="${POOL_DIR:-}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-127b}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
TOKENS_PER_PARAM="${TOKENS_PER_PARAM:-5}"
BUILD_WORKERS="${BUILD_WORKERS:-8}"
EDULLM_DATA_PKG="${EDULLM_DATA_PKG:-edullm-data @ git+https://github.com/edu-llm/edullm-data@main}"

if [[ "${DATASET_ID}" != "pretrain/olmo-127b" || "${DATASET_VERSION}" != "v1" ]]; then
  echo "SkillIt probe source is pinned to pretrain/olmo-127b/v1" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/recipe" "${RUN_DIR}/scripts"

cp -a "${SCRIPT_DIR}/prepare_skillit_probe_data.py" \
  "${SCRIPT_DIR}/prepare_skillit_370m_data.py" \
  "${SCRIPT_DIR}/probes.json" \
  "${SCRIPT_DIR}/prepare_probes.sh" \
  "${RUN_DIR}/scripts/"
cp -a "${MIXLAW_ROOT}/mixlaw_common.py" \
  "${MIXLAW_ROOT}/recipe_data.py" \
  "${MIXLAW_ROOT}/stage_working_pool_from_edullm_data.py" \
  "${RUN_DIR}/scripts/"
if [[ -f "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" ]]; then
  cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
    "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" \
    "${RUN_DIR}/scripts/" 2>/dev/null || true
fi
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,py} 2>/dev/null || true

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm numpy
pip install --upgrade "${EDULLM_DATA_PKG}"

"${RUN_DIR}/venv/bin/python" "${RUN_DIR}/scripts/prepare_skillit_probe_data.py" \
  --work "${RUN_DIR}/recipe" \
  --tokens-per-param "${TOKENS_PER_PARAM}"

if [[ -z "${POOL_DIR}" ]]; then
  POOL_DIR="${RUN_DIR}/pool"
  mkdir -p "${POOL_DIR}"
  export EDULLM_ROOT RUN_DIR
  unset PREFIX || true
  AWS_SESSION_ENV="${AWS_SESSION_ENV:-}"
  if [[ -f "${RUN_DIR}/scripts/prepare_aws_session_light.sh" ]]; then
    # shellcheck disable=SC1091
    source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
    # shellcheck disable=SC1090
    source "${AWS_SESSION_ENV}"
  fi
  BUDGET_TOKENS="$("${RUN_DIR}/venv/bin/python" -c \
    "import sys; sys.path.insert(0, '${RUN_DIR}/scripts'); from mixlaw_common import token_budget; print(token_budget(${TOKENS_PER_PARAM})[2])")"
  VERSION_ARG=""
  if [[ -n "${DATASET_VERSION}" ]]; then
    VERSION_ARG="--dataset-version ${DATASET_VERSION}"
  fi
  cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
AWS_SESSION_ENV=${AWS_SESSION_ENV:-}
POOL_DIR=${POOL_DIR}
RECIPE_WORK=${RUN_DIR}/recipe
TOKENS_PER_PARAM=${TOKENS_PER_PARAM}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION}
BUDGET_TOKENS=${BUDGET_TOKENS}
EOF
  POOL_JOB=$(sbatch --parsable \
    --partition=normal \
    --cpus-per-task="${BUILD_WORKERS}" \
    --mem=128G \
    --time=12:00:00 \
    --job-name=skillit-probe-pool \
    --chdir="${RUN_DIR}" \
    --output="${RUN_DIR}/logs/pool-%j.out" \
    --error="${RUN_DIR}/logs/pool-%j.err" \
    --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\"\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}\"; source ${RUN_DIR}/env.sh; [[ -n \"\${AWS_SESSION_ENV}\" && -f \${AWS_SESSION_ENV} ]] && source \${AWS_SESSION_ENV}; source \${VENV}/bin/activate; cd ${RUN_DIR}/scripts; python stage_working_pool_from_edullm_data.py --dataset-id ${DATASET_ID} ${VERSION_ARG} --out-dir ${POOL_DIR} --mixtures-json ${RUN_DIR}/scripts/probes.json --budget-tokens ${BUDGET_TOKENS}'")
  echo "pool_job_id=${POOL_JOB}"
  echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"
else
  "${RUN_DIR}/venv/bin/python" "${RUN_DIR}/scripts/prepare_skillit_370m_data.py" \
    --pool-dir "${POOL_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-version "${DATASET_VERSION}" \
    --pool-layout probe \
    --validate-pool-only
  cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
POOL_DIR=${POOL_DIR}
RECIPE_WORK=${RUN_DIR}/recipe
TOKENS_PER_PARAM=${TOKENS_PER_PARAM}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION}
EOF
fi

echo "RUN_DIR=${RUN_DIR}"
echo "POOL_DIR=${POOL_DIR}"
echo "RECIPE_WORK=${RUN_DIR}/recipe"
echo "DATASET_ID=${DATASET_ID}"
echo "DATASET_VERSION=${DATASET_VERSION}"
echo "probe recipe sidecars ready"
