#!/usr/bin/env bash
# Sync raw-export scripts to FarmShare scratch and submit the 1B-token FineWeb-Edu job.
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p ${STAGING}/scripts/farmshare"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/scripts/farmshare/tokenize_fineweb_edu_subset.py" \
  "${SRC}/scripts/farmshare/export_fineweb_edu_subset.py" \
  "${SRC}/scripts/farmshare/submit_fineweb_edu_1b_raw.sh" \
  "${SRC}/scripts/farmshare/setup_fineweb_tokenize_venv.sh" \
  "${HOST}:${STAGING}/scripts/farmshare/"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<'REMOTE'
set -Eeuo pipefail
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
sed -i 's/\r$//' "${STAGING}/scripts/farmshare/"*.sh "${STAGING}/scripts/farmshare/"*.py
chmod +x "${STAGING}/scripts/farmshare/"*.sh
bash "${STAGING}/scripts/farmshare/setup_fineweb_tokenize_venv.sh"
REMOTE

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash "${STAGING}/scripts/farmshare/submit_fineweb_edu_1b_raw.sh"
