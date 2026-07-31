#!/usr/bin/env bash
# Sync olmohq→olmo-127b publish scripts to FarmShare and submit.
# Payload bytes stay on FarmShare/S3 — this host only syncs scripts.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OLMOHQ_LOCAL="${REPO_ROOT}/datasets/olmohq"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/datasets/olmohq ${STAGING}/scripts/farmshare"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${OLMOHQ_LOCAL}/publish_olmohq_edullm_data.py" \
  "${OLMOHQ_LOCAL}/publish_olmohq_edullm_data.sbatch" \
  "${OLMOHQ_LOCAL}/submit_publish_olmohq_edullm_data.sh" \
  "${HOST}:${STAGING}/datasets/olmohq/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
  "${REPO_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
sed -i 's/\r\$//' "\${STAGING}/datasets/olmohq/"*.sh "\${STAGING}/datasets/olmohq/"*.py "\${STAGING}/datasets/olmohq/"*.sbatch
chmod +x "\${STAGING}/datasets/olmohq/"*.sh
bash "\${STAGING}/datasets/olmohq/submit_publish_olmohq_edullm_data.sh"
REMOTE
