#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
ERR=$(ls -t "$RUN/logs/"*1670701*.err 2>/dev/null | head -1 || true)
OUT=$(ls -t "$RUN/logs/"*1670701*.out 2>/dev/null | head -1 || true)
echo "err=$ERR"
echo "out=$OUT"
echo "=== ERR tail ==="
[[ -n "$ERR" ]] && tail -60 "$ERR" || echo "(no err file)"
echo "=== OUT last lines with publish/error ==="
[[ -n "$OUT" ]] && grep -E 'publish|Error|Expired|landing|Traceback|hash|upload|FAILED|ClientError' "$OUT" | tail -40
sacct -j 1670701 --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList -P | head -6
EOF
