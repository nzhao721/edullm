#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
echo "=== upload err tail ==="
tail -n 40 "$TOP/logs/upload-1666910.err"
echo "=== how far ==="
grep -c 'aws s3 cp' "$TOP/logs/upload-1666910.out" || true
grep 'tokenized/shards/topup_' "$TOP/logs/upload-1666910.out" | tail -n 3
REMOTE
