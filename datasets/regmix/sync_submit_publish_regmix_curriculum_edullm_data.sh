#!/usr/bin/env bash
# Sync RegMix curriculum publish scripts to FarmShare and submit.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGMIX_LOCAL="${REPO_ROOT}/datasets/regmix"
CURRICULUM_LOCAL="${REPO_ROOT}/experiments/curriculum"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/regmix-curriculum-edullm-publish-$(date -u +%Y%m%dT%H%M%SZ)}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${STAGING}/datasets/regmix' '${STAGING}/experiments/curriculum' '${STAGING}/scripts/farmshare' '${RUN_DIR}' && chmod 700 '${RUN_DIR}'"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REGMIX_LOCAL}/publish_regmix_curriculum_edullm_data.py" \
  "${REGMIX_LOCAL}/publish_regmix_curriculum_edullm_data.sbatch" \
  "${REGMIX_LOCAL}/submit_publish_regmix_curriculum_edullm_data.sh" \
  "${HOST}:${STAGING}/datasets/regmix/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${CURRICULUM_LOCAL}/curriculum_pacing.py" \
  "${HOST}:${STAGING}/experiments/curriculum/"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/scripts/farmshare/prepare_aws_session_light.sh" \
  "${REPO_ROOT}/scripts/farmshare/write_aws_session_env.py" \
  "${HOST}:${STAGING}/scripts/farmshare/"

# FarmShare cannot refresh broker credentials. Mint on this device and push only
# the temporary, job-scoped session file before submitting the Slurm job.
FARMSHARE_SOCK="${SOCK}" FARMSHARE_HOST="${HOST}" \
  bash "${REPO_ROOT}/scripts/farmshare/push_aws_session_to_farmshare.sh" "${RUN_DIR}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
STAGING=${STAGING}
RUN_DIR=${RUN_DIR}
sed -i 's/\r\$//' "\${STAGING}/datasets/regmix/"*.sh "\${STAGING}/datasets/regmix/"*.py "\${STAGING}/datasets/regmix/"*.sbatch
chmod +x "\${STAGING}/datasets/regmix/"*.sh
RUN_DIR="\${RUN_DIR}" AWS_SESSION_ENV="\${RUN_DIR}/aws-session.env" \
  bash "\${STAGING}/datasets/regmix/submit_publish_regmix_curriculum_edullm_data.sh"
REMOTE
