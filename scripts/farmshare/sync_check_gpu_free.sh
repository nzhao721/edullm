#!/usr/bin/env bash
# Install check_gpu_free.sh to permanent FarmShare staging (not a dataset run dir).
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL="${STAGING}/scripts/farmshare/check_gpu_free.sh"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" "mkdir -p ${STAGING}/scripts/farmshare ~/.local/bin"

rsync -avz -e "ssh -S ${SOCK} -o BatchMode=yes" \
  "${SCRIPT_DIR}/check_gpu_free.sh" \
  "${HOST}:${INSTALL}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<REMOTE
set -Eeuo pipefail
INSTALL="${INSTALL}"
sed -i 's/\r\$//' "\${INSTALL}"
chmod +x "\${INSTALL}"
ln -sf "\${INSTALL}" ~/.local/bin/check_gpu_free
echo "Installed: \${INSTALL}"
echo "On login node: check_gpu_free   (via ~/.local/bin)"
REMOTE
