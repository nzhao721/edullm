#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
MIX_UP=$(cat "$MIX/upload_job_id.txt")
TOP_UP=$(cat "$TOP/upload_job_id.txt")
echo "=== mixlaw-up $MIX_UP ==="
sacct -j "$MIX_UP" --format=JobID,State,ExitCode,Elapsed -n | head -5
tail -n 25 "$MIX/logs/upload-${MIX_UP}.out" 2>/dev/null || true
echo ---ERR---
tail -n 15 "$MIX/logs/upload-${MIX_UP}.err" 2>/dev/null || true
echo "=== topup tok ==="
sacct -j 1666301 --format=JobID,State -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ {c[$2]++} END {for (s in c) print s, c[s]}'
echo "=== topup-up $TOP_UP ==="
sacct -j "$TOP_UP" --format=JobID,State,ExitCode,Elapsed -n | head -5
tail -n 25 "$TOP/logs/upload-${TOP_UP}.out" 2>/dev/null || true
echo ---ERR---
tail -n 15 "$TOP/logs/upload-${TOP_UP}.err" 2>/dev/null || true
REMOTE
