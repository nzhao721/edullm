#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
echo "=== job states ==="
sacct -j 1666301,1666360,1666357,1666358,1666359 --format=JobID,JobName%16,State,ExitCode,Elapsed -n | grep -E '^(1666301 |1666360|1666357|1666358|1666359)' || true
echo "tok counts:"
sacct -j 1666301 --format=State -n -P | sort | uniq -c
echo "=== pool log ==="
ls -la /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666357.* 2>/dev/null || echo "no pool log yet"
tail -n 40 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666357.out 2>/dev/null || true
tail -n 30 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666357.err 2>/dev/null || true
REMOTE
