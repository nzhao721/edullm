#!/usr/bin/env bash
set -euo pipefail

SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p ${STAGING}/experiments/skill-dag/mixlaw ${STAGING}/experiments/skill-dag/skillit"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/mixlaw/" "${HOST}:${STAGING}/experiments/skill-dag/mixlaw/" \
  --exclude pilot_runs --exclude __pycache__ --exclude '*.pyc' --exclude '_*.sh'

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SRC}/skillit/" "${HOST}:${STAGING}/experiments/skill-dag/skillit/" \
  --exclude artifacts --exclude __pycache__ --exclude '*.pyc' --exclude tests

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -euo pipefail
STAGING=${STAGING}
find "\${STAGING}/experiments/skill-dag/mixlaw" "\${STAGING}/experiments/skill-dag/skillit" \
  -type f \( -name '*.sh' -o -name '*.py' \) -exec sed -i 's/\r$//' {} +
chmod +x "\${STAGING}/experiments/skill-dag/skillit/"*.sh \
  "\${STAGING}/experiments/skill-dag/mixlaw/"*.sh 2>/dev/null || true
ls -la "\${STAGING}/experiments/skill-dag/skillit"
REMOTE
