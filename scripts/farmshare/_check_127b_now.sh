#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
# Refresh creds best-effort
bash /mnt/c/alpha_ai/edullm/scripts/farmshare/push_aws_session_to_farmshare.sh \
  /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z || echo "push_failed"

ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash <<'EOF'
set -e
echo "=== $(date -u +%FT%TZ) ==="
squeue -u nzhao2 -o '%.18i %.9P %.22j %.2t %.10M %R' | head -20
echo "=== olmo127 jobs ==="
sacct -u nzhao2 --name=olmo127b-edullm-pub --format=JobID,State,ExitCode,Elapsed,End -P -S 2026-07-30 | tail -12
RUN=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z
echo "=== stage ==="
du -sh "$RUN/publish-stage" 2>/dev/null || true
if [[ -d "$RUN/publish-stage/tokens" ]]; then
  for d in "$RUN/publish-stage/tokens"/*; do
    [[ -d "$d" ]] || continue
    n=$(find "$d" -name 'train-*.u32le.bin' 2>/dev/null | wc -l)
    v=$(find "$d" -name 'val-*.u32le.bin' 2>/dev/null | wc -l)
    sz=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    echo "  $(basename "$d"): ${sz} train=${n} val=${v}"
  done
fi
OUT=$(ls -t "$RUN/logs/"*.out 2>/dev/null | head -1 || true)
echo "=== latest out: $OUT ==="
[[ -n "$OUT" ]] && tail -40 "$OUT"
ERR=$(ls -t "$RUN/logs/"*.err 2>/dev/null | head -1 || true)
if [[ -n "$ERR" ]] && grep -qE 'Traceback|Error|ExpiredToken|FAILED' "$ERR" 2>/dev/null; then
  echo "=== err ==="
  tail -25 "$ERR"
fi
echo "=== session mtime ==="
stat -c '%y' "$RUN/aws-session.env" 2>/dev/null || true
EOF
