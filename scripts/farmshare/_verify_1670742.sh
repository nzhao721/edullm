#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
echo "=== squeue me ==="
squeue -u nzhao2 -o '%.18i %.9P %.22j %.2t %.10M %R' | head -20
echo "=== sacct 1670742 ==="
sacct -j 1670742 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P 2>&1 | head -10
echo "=== recent olmo127 ==="
sacct -u nzhao2 --name=olmo127b-edullm-pub --format=JobID,State,ExitCode,Elapsed,End -P -S 2026-07-31 | tail -10
echo "=== logs ==="
ls -lt /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/logs/ | head -15
echo "=== skip sbatch exists ==="
ls -la /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/scripts/olmohq/publish_olmohq_skip_stage.sbatch
EOF
