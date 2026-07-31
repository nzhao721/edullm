#!/usr/bin/env bash
# Sync publish scripts to FarmShare and submit RefHQ -> edullm-data publish job.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HQ_SCRIPTS_LOCAL="${REPO_ROOT}/datasets/refhq/scripts"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p ${STAGING}/datasets/refhq/scripts ${STAGING}/scripts/farmshare"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${HQ_SCRIPTS_LOCAL}/publish_refhq_edullm_data.py" \
  "${HQ_SCRIPTS_LOCAL}/publish_refhq_edullm_data.sbatch" \
  "${HQ_SCRIPTS_LOCAL}/submit_publish_refhq_edullm_data.sh" \
  "${HQ_SCRIPTS_LOCAL}/lib.sh" \
  "${HOST}:${STAGING}/datasets/refhq/scripts/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
  "${REPO_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
REFHQ_ROOT=/scratch/users/${SUNET}/refhq-regmix-5p5b-v1
mkdir -p "\${REFHQ_ROOT}/datasets/refhq/scripts"
cp -a "\${STAGING}/datasets/refhq/scripts/." "\${REFHQ_ROOT}/datasets/refhq/scripts/"
mkdir -p "\${REFHQ_ROOT}/scripts/farmshare"
cp -a "\${STAGING}/scripts/farmshare/." "\${REFHQ_ROOT}/scripts/farmshare/"
sed -i 's/\r\$//' "\${REFHQ_ROOT}/datasets/refhq/scripts/"*.sh "\${REFHQ_ROOT}/datasets/refhq/scripts/"*.py "\${REFHQ_ROOT}/datasets/refhq/scripts/"*.sbatch
chmod +x "\${REFHQ_ROOT}/datasets/refhq/scripts/"*.sh
export EDULLM_ROOT="\${STAGING}/edullm"
bash "\${REFHQ_ROOT}/datasets/refhq/scripts/submit_publish_refhq_edullm_data.sh"
REMOTE
