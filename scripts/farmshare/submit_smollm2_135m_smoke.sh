#!/usr/bin/env bash
# Single-GPU SmolLM2-135M throughput smoke test on tokenized FineWeb-Edu 1B.
#
# Usage (FarmShare login node):
#   bash scripts/farmshare/submit_smollm2_135m_smoke.sh
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-smollm2-135m-smoke-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/output}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/smollm2-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train_smollm2_135m_smoke.py}"
HF_HOME="${HF_HOME:-/scratch/users/${SUNET}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_EVERY="${LOG_EVERY:-10}"

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${HF_HOME}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv python at ${VENV}/bin/python" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "missing ${TRAIN_PY}" >&2
  exit 2
fi
if [[ ! -f "${DATA_DIR}/train_tokens.bin" || ! -f "${DATA_DIR}/meta.json" ]]; then
  echo "missing tokenized data under ${DATA_DIR}" >&2
  exit 2
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
OUT_DIR=${OUT_DIR}
DATA_DIR=${DATA_DIR}
VENV=${VENV}
TRAIN_PY=${TRAIN_PY}
HF_HOME=${HF_HOME}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
BATCH_SIZE=${BATCH_SIZE}
MAX_STEPS=${MAX_STEPS}
NUM_WORKERS=${NUM_WORKERS}
LOG_EVERY=${LOG_EVERY}
EOF

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
  --wrap="bash -lc 'set -Eeuo pipefail; source /etc/profile.d/z00_lmod.sh 2>/dev/null || true; module load python/3.12.3 2>/dev/null || true; module load cuda/12.9.0 2>/dev/null || module load cuda/12.4.0 2>/dev/null || true; source ${RUN_DIR}/env.sh; source \${VENV}/bin/activate; export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1; nvidia-smi -L; python -u \${TRAIN_PY} --data-dir \${DATA_DIR} --output-dir \${OUT_DIR} --batch-size \${BATCH_SIZE} --max-steps \${MAX_STEPS} --num-workers \${NUM_WORKERS} --log-every \${LOG_EVERY} --save-checkpoint'")

echo "job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "submitted smollm2-smoke=${JOB_ID}"
