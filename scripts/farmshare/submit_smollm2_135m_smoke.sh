#!/usr/bin/env bash
# Single-GPU SmolLM2-135M throughput smoke test on ephemeral FarmShare scratch.
#
# Data: published s3://edullm-data/ (default: pretrain/lean4-mathlib-bytes), staged
# into this job's RUN_DIR. No pre-existing FineWeb memmaps, shared corpus cache, or
# s3://edullm-datasets/ paths.
#
# Durable artifacts: upload-before-end to S3_OUTPUT (default under
# s3://edullm-checkpoints/smollm2/smoke/) and/or W&B when wandb-session.env is present.
#
# Usage (FarmShare login node):
#   RUN_DIR=... bash scripts/farmshare/submit_smollm2_135m_smoke.sh
#
# Mint/push AWS (+ optional W&B) session from the laptop first:
#   bash scripts/farmshare/push_aws_session_to_farmshare.sh "$RUN_DIR"
#   bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"   # optional
set -Eeuo pipefail

SUNET="${SUNET:-${USER:-nzhao2}}"
RUN_NAME="${RUN_NAME:-smollm2-135m-smoke-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/output}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/staged-data}"
# Job-scoped venv + HF cache (empty scratch OK; bootstrap via setup if missing).
VENV="${VENV:-${RUN_DIR}/venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train_smollm2_135m_smoke.py}"
SETUP_SH="${SETUP_SH:-${SCRIPT_DIR}/setup_smollm2_train_venv.sh}"
HF_HOME="${HF_HOME:-${RUN_DIR}/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

DATASET_ID="${DATASET_ID:-pretrain/lean4-mathlib-bytes}"
DATASET_VERSION="${DATASET_VERSION:-}"  # empty → resolve_latest
SPLIT="${SPLIT:-train}"
MAX_SHARDS="${MAX_SHARDS:-}"

AWS_SESSION_ENV="${AWS_SESSION_ENV:-${RUN_DIR}/aws-session.env}"
S3_OUTPUT="${S3_OUTPUT:-s3://edullm-checkpoints/smollm2/smoke/${RUN_NAME}/}"
WANDB_PROJECT="${WANDB_PROJECT:-edullm-smollm2-smoke}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
# Prefer W&B when session file exists; otherwise S3-only is enough.
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-10}"

if [[ "${DATASET_ID}" == *"edullm-datasets"* ]]; then
  echo "refusing legacy edullm-datasets DATASET_ID=${DATASET_ID}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${STAGE_DIR}" "${HF_HOME}" "${RUN_DIR}/scripts"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "missing ${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -f "${SETUP_SH}" ]]; then
  echo "missing ${SETUP_SH} (needed to bootstrap job-scoped venv on the compute node)" >&2
  exit 2
fi

# Job-scoped copies so the Slurm job does not depend on a shared staging tree.
cp -f "${TRAIN_PY}" "${RUN_DIR}/scripts/train_smollm2_135m_smoke.py"
cp -f "${SETUP_SH}" "${RUN_DIR}/scripts/setup_smollm2_train_venv.sh"
TRAIN_PY="${RUN_DIR}/scripts/train_smollm2_135m_smoke.py"
SETUP_SH="${RUN_DIR}/scripts/setup_smollm2_train_venv.sh"
chmod +x "${SETUP_SH}"

# Require a job-scoped AWS session file (laptop-minted). Do not assume a shared
# persistent venv or corpus cache on scratch.
if [[ ! -f "${AWS_SESSION_ENV}" ]]; then
  echo "missing ${AWS_SESSION_ENV}." >&2
  echo "Mint on the laptop and push: bash scripts/farmshare/push_aws_session_to_farmshare.sh ${RUN_DIR}" >&2
  exit 2
fi
if ! grep -q 'AWS_ACCESS_KEY_ID=' "${AWS_SESSION_ENV}"; then
  echo "aws-session.env at ${AWS_SESSION_ENV} looks empty/invalid" >&2
  exit 2
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
OUT_DIR=${OUT_DIR}
STAGE_DIR=${STAGE_DIR}
VENV=${VENV}
TRAIN_PY=${TRAIN_PY}
SETUP_SH=${SETUP_SH}
HF_HOME=${HF_HOME}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
DATASET_ID=${DATASET_ID}
DATASET_VERSION=${DATASET_VERSION}
SPLIT=${SPLIT}
MAX_SHARDS=${MAX_SHARDS}
AWS_SESSION_ENV=${AWS_SESSION_ENV}
S3_OUTPUT=${S3_OUTPUT}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_RUN_NAME=${WANDB_RUN_NAME}
WANDB_MODE=${WANDB_MODE}
BATCH_SIZE=${BATCH_SIZE}
MAX_STEPS=${MAX_STEPS}
NUM_WORKERS=${NUM_WORKERS}
LOG_EVERY=${LOG_EVERY}
EOF

# Job body as a file avoids fragile nested quoting in sbatch --wrap.
cat > "${RUN_DIR}/run_smoke.sh" <<'EOS'
#!/usr/bin/env bash
set -Eeuo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true

source "${RUN_DIR}/env.sh"
set +u
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
if [[ -f "${RUN_DIR}/wandb-session.env" ]]; then
  # shellcheck disable=SC1090
  source "${RUN_DIR}/wandb-session.env"
fi
set -u

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "bootstrapping job-scoped venv at ${VENV}"
  VENV="${VENV}" bash "${SETUP_SH}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python - <<'PY'
import edullm_data, boto3
print("edullm_data+boto3 ok")
PY

export HF_HOME TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export WANDB_DIR="${OUT_DIR}/wandb"

VERSION_ARGS=()
if [[ -n "${DATASET_VERSION}" ]]; then
  VERSION_ARGS=(--dataset-version "${DATASET_VERSION}")
fi
MAX_SHARDS_ARGS=()
if [[ -n "${MAX_SHARDS}" ]]; then
  MAX_SHARDS_ARGS=(--max-shards "${MAX_SHARDS}")
fi
ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
fi

nvidia-smi -L
python -u "${TRAIN_PY}" \
  --dataset-id "${DATASET_ID}" \
  ${VERSION_ARGS[@]+"${VERSION_ARGS[@]}"} \
  --split "${SPLIT}" \
  --stage-dir "${STAGE_DIR}" \
  ${MAX_SHARDS_ARGS[@]+"${MAX_SHARDS_ARGS[@]}"} \
  --output-dir "${OUT_DIR}" \
  --s3-output "${S3_OUTPUT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --log-every "${LOG_EVERY}" \
  --save-checkpoint \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  ${ENTITY_ARGS[@]+"${ENTITY_ARGS[@]}"}
EOS
chmod +x "${RUN_DIR}/run_smoke.sh"

JOB_ID=$(sbatch --parsable --exclude=wheat-01 \
  --partition=gpu \
  --qos=gpu \
  --gpus-per-node=1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  --job-name=smollm2-smoke \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/train-%j.out" \
  --error="${RUN_DIR}/logs/train-%j.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}" \
  "${RUN_DIR}/run_smoke.sh")

echo "job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "STAGE_DIR=${STAGE_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "VENV=${VENV}"
echo "S3_OUTPUT=${S3_OUTPUT}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "submitted smollm2-smoke=${JOB_ID}"
