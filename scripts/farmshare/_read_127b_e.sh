#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
E=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/logs/olmo127b_edullm_publish_1670701.e
wc -c "$E"
file "$E"
# strip NULs and print
tr -d '\000' < "$E" | tail -c 4000
echo
echo "=== strings ==="
strings "$E" | tail -40
EOF
