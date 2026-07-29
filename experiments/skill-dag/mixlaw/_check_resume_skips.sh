#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
UP=$(cat "$TOP/upload_job_id.txt")
echo "job=$UP"
sacct -j "$UP" --format=JobID,State,ExitCode,Elapsed -n | head -5
echo "=== log head ==="
head -n 30 "$TOP/logs/upload-${UP}.out" 2>/dev/null || true
echo "=== counts ==="
echo -n "skips="; grep -c '^skip' "$TOP/logs/upload-${UP}.out" 2>/dev/null || echo 0
echo -n "cps="; grep -c 'aws s3 cp' "$TOP/logs/upload-${UP}.out" 2>/dev/null || echo 0
echo "=== tail ==="
tail -n 15 "$TOP/logs/upload-${UP}.out" 2>/dev/null || true
tail -n 10 "$TOP/logs/upload-${UP}.err" 2>/dev/null || true
REMOTE
