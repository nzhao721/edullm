#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
echo "=== mixlaw-up ==="
sacct -j 1666758 --format=JobID,State,ExitCode,Elapsed -n | head -5
tail -n 30 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/upload-1666758.out 2>/dev/null || true
echo ---ERR---
tail -n 20 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/upload-1666758.err 2>/dev/null || true
echo "=== tok / topup-up ==="
sacct -j 1666301 --format=JobID,State -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ {c[$2]++} END {for (s in c) print s, c[s]}'
sacct -j 1666759 --format=JobID,State,ExitCode,Elapsed -n | head -5
REMOTE
