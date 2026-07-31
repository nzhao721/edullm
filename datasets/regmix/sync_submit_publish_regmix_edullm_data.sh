#!/usr/bin/env bash
# Sync RegMix publish scripts to FarmShare and submit.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGMIX_LOCAL="${REPO_ROOT}/datasets/regmix"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/datasets/regmix ${STAGING}/scripts/farmshare"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REGMIX_LOCAL}/publish_regmix_edullm_data.py" \
  "${REGMIX_LOCAL}/publish_regmix_edullm_data.sbatch" \
  "${REGMIX_LOCAL}/submit_publish_regmix_edullm_data.sh" \
  "${HOST}:${STAGING}/datasets/regmix/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
  "${REPO_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
sed -i 's/\r\$//' "\${STAGING}/datasets/regmix/"*.sh "\${STAGING}/datasets/regmix/"*.py "\${STAGING}/datasets/regmix/"*.sbatch
chmod +x "\${STAGING}/datasets/regmix/"*.sh
bash "\${STAGING}/datasets/regmix/submit_publish_regmix_edullm_data.sh"
REMOTE
