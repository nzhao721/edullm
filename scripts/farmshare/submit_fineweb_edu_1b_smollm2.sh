#!/usr/bin/env bash
# Stream 1B tokens from FineWeb-Edu, tokenize with SmolLM2-135M tokenizer, write to scratch.
#
# Usage (FarmShare login node):
#   bash scripts/farmshare/submit_fineweb_edu_1b_smollm2.sh
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
RUN_NAME="${RUN_NAME:-fineweb-edu-1b-smollm2-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
OUT_DIR="${OUT_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
TOKENIZER="${TOKENIZER:-HuggingFaceTB/SmolLM2-135M}"
MAX_TOKENS="${MAX_TOKENS:-1000000000}"
VENV="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/fineweb-tokenize}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZE_PY="${TOKENIZE_PY:-${SCRIPT_DIR}/tokenize_fineweb_edu_subset.py}"
HF_HOME="${HF_HOME:-/scratch/users/${SUNET}/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"

mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}" "${HF_HOME}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv python at ${VENV}/bin/python" >&2
  exit 2
fi
if [[ ! -f "${TOKENIZE_PY}" ]]; then
  echo "missing ${TOKENIZE_PY}" >&2
  exit 2
fi

cat > "${RUN_DIR}/env.sh" <<EOF
RUN_DIR=${RUN_DIR}
OUT_DIR=${OUT_DIR}
VENV=${VENV}
TOKENIZE_PY=${TOKENIZE_PY}
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
  --job-name=fwedu-1b-tok \
  --chdir="${RUN_DIR}" \
  --output="${RUN_DIR}/logs/tokenize-%j.out" \
  --error="${RUN_DIR}/logs/tokenize-%j.err" \
  --wrap="bash -lc 'set -Eeuo pipefail; source ${RUN_DIR}/env.sh; source \${VENV}/bin/activate; export HF_HOME TRANSFORMERS_CACHE TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1; python -u \${TOKENIZE_PY} --dataset fineweb_edu --output-dir \${OUT_DIR} --tokenizer \${TOKENIZER} --max-train-tokens \${MAX_TOKENS}'")

echo "job_id=${JOB_ID}"
echo "${JOB_ID}" > "${RUN_DIR}/job_id.txt"
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "TOKENIZER=${TOKENIZER}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "submitted fwedu-1b-tok=${JOB_ID}"
