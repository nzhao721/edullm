#!/usr/bin/env bash
# Stream 1B tokens from FineWeb-Edu and write matching raw text shards to scratch.
#
# Usage (FarmShare login node):
#   bash scripts/farmshare/submit_fineweb_edu_1b_raw.sh
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-fineweb-edu-1b-raw-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
OUT_DIR="${OUT_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-1b-smollm2-raw}"
TOKENIZER="${TOKENIZER:-HuggingFaceTB/SmolLM2-135M}"
MAX_TOKENS="${MAX_TOKENS:-1000000000}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/fineweb-tokenize}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_PY="${EXPORT_PY:-${SCRIPT_DIR}/export_fineweb_edu_subset.py}"
HF_HOME="${HF_HOME:-/scratch/users/${SUNET}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${HF_HOME}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv python at ${VENV}/bin/python" >&2
  exit 2
fi
if [[ ! -f "${EXPORT_PY}" ]]; then
  echo "missing ${EXPORT_PY}" >&2
  exit 2
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
OUT_DIR=${OUT_DIR}
VENV=${VENV}
EXPORT_PY=${EXPORT_PY}
TOKENIZER=${TOKENIZER}
MAX_TOKENS=${MAX_TOKENS}
HF_HOME=${HF_HOME}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
EOF

JOB_ID=$(sbatch --parsable --exclude=wheat-01 \
  --partition=normal \
  --cpus-per-task=8 \
  --mem=64G \
  --time=24:00:00 \
  --job-name=fwedu-1b-raw \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/export-%j.out" \
  --error="${RUN_DIR}/logs/export-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source \${VENV}/bin/activate; export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1; python -u \${EXPORT_PY} --dataset fineweb_edu --output-dir \${OUT_DIR} --tokenizer \${TOKENIZER} --max-train-tokens \${MAX_TOKENS}'")

echo "job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOKENIZER=${TOKENIZER}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "submitted fwedu-1b-raw=${JOB_ID}"
