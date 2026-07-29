#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging

# Sync fixed submit scripts into staging (no regmix writes).
scp -o ControlPath="$SOCK" -o BatchMode=yes \
  /mnt/c/alpha_ai/edullm/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh \
  "$HOST:$STAGING/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh"
scp -o ControlPath="$SOCK" -o BatchMode=yes \
  /mnt/c/alpha_ai/edullm/datasets/olmohq/submit_olmohq_topup.sh \
  "$HOST:$STAGING/datasets/olmohq/submit_olmohq_topup.sh"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
sed -i 's/\r$//' /scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh
sed -i 's/\r$//' /scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/datasets/olmohq/submit_olmohq_topup.sh
echo "synced submit scripts"
echo "=== status ==="
sacct -j 1666301,1666360,1666357,1666358,1666359 --format=JobID,JobName%16,State,ExitCode,Elapsed -n | grep -E '^(1666301 |1666360|1666357|1666358|1666359)' || true
echo "tok counts:"
sacct -j 1666301 --format=State -n -P | sort | uniq -c
echo "pool tail:"
tail -n 8 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666357.out 2>/dev/null || true
tail -n 5 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666357.err 2>/dev/null || true
REMOTE
