#!/usr/bin/env bash
# Submit SmolLM2 500M/40-epoch DDP training from published edullm-data.
# Legacy name kept; does NOT slice FarmShare scratch memmaps.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUNET="${SUNET:-${USER:-nzhao2}}"
# Prefer scripts next to this wrapper (synced staging or repo checkout).
SUBMIT="${SUBMIT:-${SCRIPT_DIR}/submit_smollm2_135m_500m_40ep.sh}"
VENV_TRAIN="${VENV:-/scratch/users/${SUNET}/agent-runs/venvs/smollm2-train}"

export DATASET_ID="${DATASET_ID:-pretrain/fineweb-edu-500m}"
export VENV="${VENV_TRAIN}"
export TRAIN_PY="${TRAIN_PY:-${SCRIPT_DIR}/train_smollm2_135m_ddp.py}"
export SLURM_MEM="${SLURM_MEM:-48G}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# Refuse legacy local-slice / persistent-ckpt env vars.
unset SRC_DATA_DIR DATA_DIR SLICE_PY
unset RESUME_FROM NODELIST RUN_DIR OUT_DIR RUN_NAME
bash "${SUBMIT}"
