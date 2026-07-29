#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
echo "=== mixlaw upload logs ==="
ls -la "$MIX/logs"/upload-*
echo ---OUT---
cat "$MIX/logs/upload-1666359.out" 2>/dev/null || true
echo ---ERR---
cat "$MIX/logs/upload-1666359.err" 2>/dev/null || true
echo "=== slices present ==="
ls -la "$MIX/slices" | head -30
echo "=== tok remaining ==="
sacct -j 1666301 --format=JobID,State -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ {c[$2]++} END {for (s in c) print s, c[s]}'
REMOTE
