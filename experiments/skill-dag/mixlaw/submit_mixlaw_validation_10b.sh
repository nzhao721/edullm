#!/usr/bin/env bash
# DEPRECATED / DO-NOT-USE for new work.
# Per-mix slice materialization + upload to edullm-datasets/mixlaw/ is unsupported.
# Sole supported path: submit_mixlaw_validation_pool.sh +
# submit_mixlaw_validation_370m.sh (DomainMixtureStream over an edullm-data peak pool).
#
# Historical: build 10B mixlaw validation mixtures on FarmShare; upload to
# edullm-datasets/mixlaw/. NEVER writes to regmix-10b.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-mixlaw-validation-10b-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SRC_BUCKET="${SRC_BUCKET:-edullm-datasets}"
SRC_PREFIX="${SRC_PREFIX:-olmo100b/olmo-mix-1124-30b}"
DST_BUCKET="${DST_BUCKET:-edullm-datasets}"
DST_PREFIX="${DST_PREFIX:-mixlaw}"
BUDGET_TOKENS="${BUDGET_TOKENS:-10000000000}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/pool" "${RUN_DIR}/slices" "${RUN_DIR}/plan"
cd "${RUN_DIR}"

SKILL="${EDULLM_ROOT}/experiments/skill-dag/mixlaw"
for f in mixlaw_common.py build_mixture_data.py build_working_pool_from_shards.py \
  finalize_mixlaw_upload.py write_validation_mixtures.py \
  validation_mixtures_10b.json \
  mixlaw_fit_chinchilla.json mixlaw_fit_lightgbm_chinchilla.json mixtures.json; do
  cp -a "${SKILL}/${f}" "${RUN_DIR}/scripts/"
done
cp -a "${EDULLM_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" "${RUN_DIR}/scripts/"
cp -a "${EDULLM_ROOT}/scripts/farmshare/write_aws_session_env.py" "${RUN_DIR}/scripts/"
sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{sh,py} 2>/dev/null || true

if [[ ! -x "${RUN_DIR}/venv/bin/python" ]]; then
  python3 -m venv "${RUN_DIR}/venv"
fi
# shellcheck disable=SC1091
source "${RUN_DIR}/venv/bin/activate"
pip install -U pip wheel
pip install boto3 tqdm numpy

export EDULLM_ROOT RUN_DIR
unset PREFIX || true
# shellcheck disable=SC1091
source "${RUN_DIR}/scripts/prepare_aws_session_light.sh"
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"

cd "${RUN_DIR}/scripts"
# Prefer the checked-in recipe (canonical domain weights / mix names). Regenerate
# only if missing — write_validation_mixtures.py must match the same picks.
if [[ ! -f validation_mixtures_10b.json ]]; then
  python write_validation_mixtures.py
fi
cp -a validation_mixtures_10b.json "${RUN_DIR}/plan/"
python - <<'PY'
import json
from pathlib import Path
p = Path("validation_mixtures_10b.json")
recipe = json.loads(p.read_text(encoding="utf-8"))
names = [m["run_name"] for m in recipe["mixtures"]]
print("validation recipe mixes:", ", ".join(names))
assert len(names) == 8, names
assert "ML-pilot_caps" in names and "ML-near-opt-4" in names, names
assert "LGB-min1pct" in names and "LGB-near-opt-8" in names, names
PY

aws s3 cp "s3://${SRC_BUCKET}/${SRC_PREFIX}/plan/tokenized_manifest.json" \
  "${RUN_DIR}/plan/tokenized_manifest.json"

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
VENV=${RUN_DIR}/venv
EDULLM_ROOT=${EDULLM_ROOT}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
SRC_BUCKET=${SRC_BUCKET}
SRC_PREFIX=${SRC_PREFIX}
DST_BUCKET=${DST_BUCKET}
DST_PREFIX=${DST_PREFIX}
BUDGET_TOKENS=${BUDGET_TOKENS}
BUILD_WORKERS=${BUILD_WORKERS}
EOF

# Slurm --wrap runs under /bin/sh; always use bash -lc. Put aws CLI on PATH.
_AWS_PATH='export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"'

# Stage 1: build working pool from olmohq tokenized shards (CPU, long I/O).
POOL_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=8 \
  --mem=64G \
  --time=12:00:00 \
  --job-name=mixlaw-pool \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/pool-%j.out" \
  --error="${RUN_DIR}/logs/pool-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; ${_AWS_PATH}; source ${RUN_DIR}/env.sh; source \${AWS_SESSION_ENV}; source ${RUN_DIR}/venv/bin/activate; cd ${RUN_DIR}/scripts; python build_working_pool_from_shards.py --tokenized-manifest ${RUN_DIR}/plan/tokenized_manifest.json --s3-tokenized-prefix s3://${SRC_BUCKET}/${SRC_PREFIX}/tokenized --out-dir ${RUN_DIR}/pool --mixtures-json ${RUN_DIR}/plan/validation_mixtures_10b.json --budget-tokens ${BUDGET_TOKENS}'")
echo "pool_job_id=${POOL_JOB}"
echo "${POOL_JOB}" > "${RUN_DIR}/pool_job_id.txt"

# Stage 2: plan + materialize slices.
SLICE_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --dependency="afterok:${POOL_JOB}" \
  --job-name=mixlaw-slice \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/slice-%j.out" \
  --error="${RUN_DIR}/logs/slice-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; source ${RUN_DIR}/env.sh; source ${RUN_DIR}/venv/bin/activate; cd ${RUN_DIR}/scripts; python build_mixture_data.py plan --tokenized-dir ${RUN_DIR}/pool/tokenized --out-dir ${RUN_DIR}/slices --mixtures-json ${RUN_DIR}/plan/validation_mixtures_10b.json --total-tokens ${BUDGET_TOKENS}; python build_mixture_data.py build --plan-dir ${RUN_DIR}/slices --out-dir ${RUN_DIR}/slices --workers ${BUILD_WORKERS}'")
echo "slice_job_id=${SLICE_JOB}"
echo "${SLICE_JOB}" > "${RUN_DIR}/slice_job_id.txt"

# Stage 3: upload to mixlaw/ (use login-minted aws-session.env; never remint on
# compute — sb-aws profile broker is login-only. Never write regmix).
UP_JOB=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=8 \
  --mem=32G \
  --time=12:00:00 \
  --dependency="afterok:${SLICE_JOB}" \
  --job-name=mixlaw-up \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/upload-%j.out" \
  --error="${RUN_DIR}/logs/upload-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; unset PREFIX || true; ${_AWS_PATH}; source ${RUN_DIR}/env.sh; source \${AWS_SESSION_ENV}; source ${RUN_DIR}/venv/bin/activate; cd ${RUN_DIR}/scripts; python finalize_mixlaw_upload.py --run-dir ${RUN_DIR} --mixtures-json ${RUN_DIR}/plan/validation_mixtures_10b.json --dst-bucket ${DST_BUCKET} --dst-prefix ${DST_PREFIX}'")
echo "upload_job_id=${UP_JOB}"
echo "${UP_JOB}" > "${RUN_DIR}/upload_job_id.txt"

echo "RUN_DIR=${RUN_DIR}"
echo "submitted pool=${POOL_JOB} slice=${SLICE_JOB} upload=${UP_JOB}"
