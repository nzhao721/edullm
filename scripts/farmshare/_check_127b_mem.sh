#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
echo "=== recent olmo127 jobs MaxRSS ==="
sacct -u nzhao2 --name=olmo127b-edullm-pub --format=JobID,State,ReqMem,MaxRSS,AveRSS,Elapsed,ExitCode -P -S 2026-07-30 | tail -20
echo "=== 1670701 detail ==="
sacct -j 1670701 --format=JobID,State,ReqMem,MaxRSS,MaxVMSize,Elapsed,ExitCode -P
echo "=== stderr OOM/kill hints ==="
grep -E 'Memory|OOM|Killed|Cannot allocate|MaxRSS|oom' /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/logs/olmo127b_edullm_publish_1670701.e 2>/dev/null | tail -20 || true
grep -E 'Memory|OOM|Killed|Cannot allocate|hash|publish|staging' /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/logs/olmo127b_edullm_publish_1670701.out 2>/dev/null | tail -30 || true
echo "=== skip sbatch mem ==="
grep -E '#SBATCH|--mem' /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/scripts/olmohq/publish_olmohq_skip_stage.sbatch
EOF
