#!/usr/bin/env bash
# Sync SmolLM2 500M/40-epoch DDP training scripts to FarmShare and submit.
# Stages published edullm-data shards at job start into the job RUN_DIR (no local FineWeb slice).
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SOCK="${SOCKET:-/tmp/farmshare-${SUNET}.sock}"
HOST="${HOST:-${SUNET}@login.farmshare.stanford.edu}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p ${STAGING}/scripts/farmshare"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/scripts/farmshare/eval_arc_task_loss_smollm.py" \
  "${SRC}/scripts/farmshare/train_smollm2_135m_ddp.py" \
  "${SRC}/scripts/farmshare/setup_smollm2_train_venv.sh" \
  "${SRC}/scripts/farmshare/submit_smollm2_135m_500m_40ep.sh" \
  "${SRC}/scripts/farmshare/slice_and_train_v2.sh" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "SUNET=${SUNET} STAGING=${STAGING} bash -s" <<'REMOTE'
set -Eeuo pipefail
sed -i 's/\r$//' "${STAGING}/scripts/farmshare/"*.sh "${STAGING}/scripts/farmshare/"*.py
chmod +x "${STAGING}/scripts/farmshare/"*.sh
bash "${STAGING}/scripts/farmshare/setup_smollm2_train_venv.sh"
REMOTE

echo "NOTE: push aws-session.env (edullm-data read + edullm-checkpoints write) and wandb-session.env to the new RUN_DIR."
echo "Default DATASET_ID=pretrain/fineweb-edu-500m — must exist as a validated catalog entry."
echo "Default CHECKPOINT_S3_URI=s3://edullm-checkpoints/smollm2/<RUN_NAME>."
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "SUNET=${SUNET} bash ${STAGING}/scripts/farmshare/submit_smollm2_135m_500m_40ep.sh"
