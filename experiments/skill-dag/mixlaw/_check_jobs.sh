#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
echo "=== tok/upload status ==="
sacct -j 1666301,1666302 --format=JobID,JobName%16,State,ExitCode,Elapsed -n | head -30
echo "tok state counts:"
sacct -j 1666301 --format=State -n -P 2>/dev/null | sort | uniq -c
echo "=== aws path on login ==="
export PATH="${HOME}/.local/bin:${HOME}/tools/aws/bin:${PATH}"
command -v aws || echo "aws missing"
echo "=== mix env ==="
cat /scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236/env.sh
echo "=== pending/cancelled mixlaw ==="
sacct -j 1666303,1666304,1666305 --format=JobID,JobName%16,State,ExitCode,Elapsed -n
REMOTE
