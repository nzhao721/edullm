#!/usr/bin/env bash
# Stage a Skill-It 370M working pool from published s3://edullm-data/.
#
# Uses prepare_skillit_370m_data.py → edullm_data.read.dataset_paths / resolve_latest.
# Default dataset: pretrain/olmo-original-30b (validated). Does not read
# s3://edullm-datasets/ and does not assume a pre-existing local/scratch pool.
#
# Bootstraps a CPU venv in RUN_DIR (edullm-data + boto3). No ladder GPU venv.
#
# Usage (FarmShare login node, with AWS session env available):
#   bash experiments/skill-dag/skillit/submit_skillit_train_pool.sh
set -Eeuo pipefail

SUNET="${SUNET:-${USER:?set SUNET or USER}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIXLAW_ROOT="$(cd "${SCRIPT_DIR}/../mixlaw" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EDULLM_ROOT="${EDULLM_ROOT:-${REPO_ROOT}}"

RUN_NAME="${RUN_NAME:-skillit-train-pool-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
DATASET_ID="${DATASET_ID:-pretrain/olmo-original-30b}"
DATASET_VERSION="${DATASET_VERSION:-}"
BUDGET_TOKENS="${BUDGET_TOKENS:-10000000000}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"
EDULLM_DATA_PKG="${EDULLM_DATA_PKG:-edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/pool" "${RUN_DIR}/work" "${RUN_DIR}/plan"
cd "${RUN_DIR}"

cp -a "${MIXLAW_ROOT}/mixlaw_common.py" "${RUN_DIR}/scripts/"
cp -a "${SCRIPT_DIR}/prepare_skillit_370m_data.py" \
  "${SCRIPT_DIR}/skillit_train_recipe.json" \
  "${RUN_DIR}/scripts/"
if [[ -f "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" ]]; then
  cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/" 2>/dev/null || true
  cp -a "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/" 2>/dev/null || true
fi
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,py} 2>/dev/null || true

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm numpy
# shellcheck disable=SC2086
pip install ${EDULLM_DATA_PKG}

export EDULLM_ROOT RUN_DIR
unset PREFIX || true
AWS_SESSION_ENV="${AWS_SESSION_ENV:-}"
if [[ -f "${RUN_DIR}/scripts/prepare_aws_session_light.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
  # shellcheck disable=SC1090
  source "${AWS_SESSION_ENV}"
fi

VERSION_FLAGS=""
if [[ -n "${DATASET_VERSION}" ]]; then
  VERSION_FLAGS="--dataset-version ${DATASET_VERSION}"
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
AWS_SESSION_ENV=${AWS_SESSION_ENV:-}
DATASET_ID=${DATASET_ID}
BUDGET_TOKENS=${BUDGET_TOKENS}
VERSION_FLAGS=${VERSION_FLAGS}
EOF

POOL_JOB=$(sbatch --parsable \
  --partition=normal \
  --cpus-per-task="${BUILD_WORKERS}" \
  --mem=128G \
  --time=12:00:00 \
  --job-name=skillit-pool \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/pool-%j.out" \
  --error="${RUN_DIR}/logs/pool-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; export PATH=\"\${HOME}/.local/bin:\${HOME}/tools/aws/bin:\${PATH}\"; source ${RUN_DIR}/env.sh; [[ -n \"\${AWS_SESSION_ENV}\" && -f \${AWS_SESSION_ENV} ]] && source \${AWS_SESSION_ENV}; source \${VENV}/bin/activate; cd ${RUN_DIR}/scripts; python prepare_skillit_370m_data.py --recipe ${RUN_DIR}/scripts/skillit_train_recipe.json --work ${RUN_DIR}/work --pool-dir ${RUN_DIR}/pool --dataset-id \${DATASET_ID} --max-tokens-per-domain \${BUDGET_TOKENS} \${VERSION_FLAGS}'")

echo "pool_job_id=${POOL_JOB}"
echo "RUN_DIR=${RUN_DIR}"
echo "POOL_DIR=${RUN_DIR}/pool"
echo "DATASET_ID=${DATASET_ID}"
echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"
