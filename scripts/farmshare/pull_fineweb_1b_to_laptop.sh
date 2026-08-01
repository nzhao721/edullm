#!/usr/bin/env bash
set -euo pipefail
SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
SRC="nzhao2@login.farmshare.stanford.edu:/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized/"
DEST="/mnt/c/Users/natha/data/fineweb-edu-1b-smollm2-tokenized/"
mkdir -p "${DEST}"
rsync -a --info=progress2 \
  -e "ssh -S ${SOCKET} -o BatchMode=yes" \
  "${SRC}" "${DEST}"
echo "DONE"
du -sh "${DEST}"
ls -lah "${DEST}"
