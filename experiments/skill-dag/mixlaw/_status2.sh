#!/usr/bin/env bash
set -Eeuo pipefail
ssh -S /tmp/farmshare-nzhao2.sock -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash -s <<'REMOTE'
set -e
echo "=== topup chain sacct ==="
sacct -j 1665880,1666207,1666208,1666209 --format=JobID,JobName%25,State,ExitCode,Elapsed -n | head -40
echo "=== mixlaw sacct ==="
sacct -j 1666297,1666298,1666299 --format=JobID,JobName%25,State,ExitCode,Elapsed -n
echo "=== map log ==="
cat /scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841/logs/map-1666207.out 2>/dev/null || echo missing
echo "=== mixlaw pool ==="
ls -la /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/ 2>/dev/null | head
tail -n 50 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666297.out 2>/dev/null || echo 'no pool out'
tail -n 30 /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/logs/pool-1666297.err 2>/dev/null || echo 'no pool err'
echo "=== squeue me ==="
squeue --me | head -40
REMOTE
