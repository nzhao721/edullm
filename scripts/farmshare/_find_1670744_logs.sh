#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<EOF
set -e
echo "=== logs dir ==="
ls -lt $RUN/logs/ | head -20
echo "=== find new logs ==="
find $RUN -name '*1670744*' 2>/dev/null | head
find /scratch/users/nzhao2 -name '*1670744*' 2>/dev/null | head
echo "=== scontrol ==="
scontrol show job 1670742 2>/dev/null | head -5 || true
scontrol show job 1670744 | egrep 'JobState|WorkDir|StdOut|StdErr|Command|NumCPUs|MinMemory|BatchHost'
echo "=== cwd listing ==="
ls -la $RUN/logs/olmo127b_edullm_publish_1670744* 2>&1 | head
EOF
