#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
echo "=== local shards ==="
find $RUN/publish-stage -name '*.u32le.bin' | wc -l
find $RUN/publish-stage -name '*.u32le.bin' | awk -F/ '{print \$(NF-1)}' | sort | uniq -c
du -sh $RUN/publish-stage
EOF
