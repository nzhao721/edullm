#!/usr/bin/env bash
# Run the permanently installed GPU check on FarmShare from a local machine (WSL).
set -Eeuo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
INSTALL="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}/scripts/farmshare/check_gpu_free.sh"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash "${INSTALL}"
