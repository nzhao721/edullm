#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
EDULLM_ROOT=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
RUN_DIR=/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-edullm-publish-20260801T194621Z
REPO=/mnt/c/alpha_ai/edullm

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO}/datasets/fineweb/" \
  "${HOST}:${EDULLM_ROOT}/datasets/fineweb/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "find ${EDULLM_ROOT}/datasets/fineweb -type f \\( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' \\) -print0 | xargs -0 sed -i 's/\r$//'"

bash "${REPO}/scripts/farmshare/push_aws_session_to_farmshare.sh" "${RUN_DIR}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
mkdir -p '${RUN_DIR}/logs'
cd '${RUN_DIR}'
sbatch --exclude=wheat-01 \
  --export=ALL,RUN_DIR='${RUN_DIR}',STAGE_DIR='${RUN_DIR}/publish-stage',EDULLM_ROOT='${EDULLM_ROOT}',AWS_SESSION_ENV='${RUN_DIR}/aws-session.env' \
  '${EDULLM_ROOT}/datasets/fineweb/republish_fineweb_edu_1b.sbatch'
EOF
