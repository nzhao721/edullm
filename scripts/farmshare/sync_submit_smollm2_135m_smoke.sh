#!/usr/bin/env bash
# Sync SmolLM2 smoke-train scripts to FarmShare, mint AWS (+ optional W&B) into a
# fresh RUN_DIR, bootstrap a job-scoped venv, and submit a 1-GPU smoke job.
#
# Assumes empty/ephemeral scratch for the run: stages edullm-data inside RUN_DIR,
# durable uploads go to S3 (and W&B if a key is available).
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
RUN_NAME="${RUN_NAME:-smollm2-135m-smoke-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${RUN_NAME}}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/scripts/farmshare ${RUN_DIR}/logs"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/scripts/farmshare/train_smollm2_135m_smoke.py" \
  "${SRC}/scripts/farmshare/setup_smollm2_train_venv.sh" \
  "${SRC}/scripts/farmshare/submit_smollm2_135m_smoke.sh" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING='${STAGING}'
sed -i 's/\r$//' "\${STAGING}/scripts/farmshare/"*.sh "\${STAGING}/scripts/farmshare/"*.py
chmod +x "\${STAGING}/scripts/farmshare/"*.sh
REMOTE

# Mint on laptop → this job's RUN_DIR (never rely on a shared persistent session alone).
if [[ -x "${SCRIPT_DIR}/push_aws_session_to_farmshare.sh" ]]; then
  bash "${SCRIPT_DIR}/push_aws_session_to_farmshare.sh" "${RUN_DIR}"
else
  echo "error: push_aws_session_to_farmshare.sh missing; cannot supply aws-session.env" >&2
  exit 2
fi

if [[ -x "${SCRIPT_DIR}/push_wandb_session_to_farmshare.sh" ]]; then
  bash "${SCRIPT_DIR}/push_wandb_session_to_farmshare.sh" "${RUN_DIR}" \
    || echo "warn: wandb session not pushed; S3_OUTPUT alone will be the durable sink" >&2
fi

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "SUNET=${SUNET} RUN_DIR=${RUN_DIR} RUN_NAME=${RUN_NAME} \
   VENV=${RUN_DIR}/venv \
   bash ${STAGING}/scripts/farmshare/submit_smollm2_135m_smoke.sh"

echo "RUN_DIR=${RUN_DIR}"
echo "S3_OUTPUT default: s3://edullm-checkpoints/smollm2/smoke/${RUN_NAME}/"
