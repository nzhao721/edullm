#!/usr/bin/env bash
# After a successful v2 tokenize job: slice 500M, verify, submit training.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging/scripts/farmshare}"
SRC_DATA="${SRC_DATA:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-1b-smollm2-tokenized}"
DATA_DIR="${DATA_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-500m-smollm2-tokenized}"
VENV_TOK="${VENV_TOK:-/scratch/users/${SUNET}/agent-runs/venvs/fineweb-tokenize}"
VENV_TRAIN="${VENV_TRAIN:-/scratch/users/${SUNET}/agent-runs/venvs/smollm2-train}"

test -f "${SRC_DATA}/meta.json" || { echo "missing ${SRC_DATA}/meta.json" >&2; exit 2; }

echo "Slicing 500M from ${SRC_DATA} -> ${DATA_DIR}"
rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"
source "${VENV_TOK}/bin/activate"
python -u "${STAGING}/slice_tokenized_subset.py" \
  --src-dir "${SRC_DATA}" \
  --dst-dir "${DATA_DIR}" \
  --max-tokens 500000000
deactivate || true

echo "Submitting training on clean v2 data"
export DATA_DIR
export SRC_DATA_DIR="${SRC_DATA}"
export VENV="${VENV_TRAIN}"
export TRAIN_PY="${STAGING}/train_smollm2_135m_ddp.py"
export SLICE_PY="${STAGING}/slice_tokenized_subset.py"
export SLURM_MEM="${SLURM_MEM:-48G}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
unset RESUME_FROM NODELIST RUN_DIR OUT_DIR RUN_NAME
bash "${STAGING}/submit_smollm2_135m_500m_40ep.sh"
