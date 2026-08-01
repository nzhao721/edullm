#!/usr/bin/env bash
# Sync FineWeb 1B publish scripts to FarmShare staging and submit the Slurm job.
#
# Prereqs (laptop):
#   1. Control socket: /tmp/farmshare-nzhao2.sock
#   2. Mint+push AWS session to RUN_DIR (this script calls push_aws_session)
#   3. Start refresh loop for long uploads
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-${SUNET}.sock}"
HOST="${SUNET}@login.farmshare.stanford.edu"
EDULLM_ROOT="${EDULLM_ROOT:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/fineweb-edu-1b-edullm-publish-${STAMP}}"
STAGE_DIR="${STAGE_DIR:-${RUN_DIR}/publish-stage}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PUSH_AWS="${REPO_ROOT}/scripts/farmshare/push_aws_session_to_farmshare.sh"

echo "sync → ${EDULLM_ROOT}/datasets/fineweb + edullm_text_companion.py"
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p '${EDULLM_ROOT}/datasets/fineweb' '${RUN_DIR}/logs'"
rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/fineweb/" \
  "${HOST}:${EDULLM_ROOT}/datasets/fineweb/"
rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${REPO_ROOT}/datasets/edullm_text_companion.py" \
  "${HOST}:${EDULLM_ROOT}/datasets/edullm_text_companion.py"

echo "mint+push AWS session → ${RUN_DIR}"
bash "${PUSH_AWS}" "${RUN_DIR}"

echo "submit publish job run_dir=${RUN_DIR}"
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
mkdir -p '${RUN_DIR}/logs'
cd '${RUN_DIR}'
sbatch --exclude=wheat-01 \
  --export=ALL,RUN_DIR='${RUN_DIR}',STAGE_DIR='${STAGE_DIR}',EDULLM_ROOT='${EDULLM_ROOT}',AWS_SESSION_ENV='${RUN_DIR}/aws-session.env' \
  '${EDULLM_ROOT}/datasets/fineweb/publish_fineweb_edu_1b_edullm_data.sbatch'
EOF
echo "submitted. Start refresh loop:"
echo "  bash scripts/farmshare/loop_push_aws_session_to_farmshare.sh '${RUN_DIR}'"
