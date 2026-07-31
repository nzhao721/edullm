#!/usr/bin/env bash
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/scripts/farmshare
RUN_DIR=/scratch/users/nzhao2/agent-runs/smollm2-135m-500m-40ep-20260730-061208
export RUN_DIR
export OUT_DIR="${RUN_DIR}/output"
export RESUME_FROM="${OUT_DIR}/checkpoints/step0003815"
export NODELIST="oat-04,oat-06"
export SLURM_MEM="48G"
export CPUS_PER_TASK="8"
export TRAIN_PY="${STAGING}/train_smollm2_135m_ddp.py"
sed -i 's/\r$//' "${STAGING}/"*.py "${STAGING}/"*.sh
bash "${STAGING}/submit_smollm2_135m_500m_40ep.sh"
