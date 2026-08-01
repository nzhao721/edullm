#!/usr/bin/env bash
# Pull FineWeb-Edu 1B raw text shards from FarmShare to the laptop.
set -euo pipefail
SOCKET="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
SRC="nzhao2@login.farmshare.stanford.edu:/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw/"
DEST="${DEST:-/mnt/c/Users/natha/data/fineweb-edu-1b-smollm2-raw/}"
mkdir -p "${DEST}"
rsync -a --info=progress2 \
  -e "ssh -S ${SOCKET} -o BatchMode=yes" \
  "${SRC}" "${DEST}"
echo "DONE"
du -sh "${DEST}"
find "${DEST}" -type f | wc -l
ls -lah "${DEST}"
ls -lah "${DEST}/shards" | head
