#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
echo "=== GPU queue (nzhao2) ==="
squeue -u nzhao2 -p gpu -o "%.18i %.15j %.2t %.10M %R" 2>/dev/null | head -30
echo "=== recent chunk job outcomes (last 2h) ==="
sacct -u nzhao2 -S "$(date -d '2 hours ago' +%Y-%m-%dT%H:%M)" --format=JobID,JobName%15,State,ExitCode,Elapsed -P -n 2>/dev/null \
  | grep -E '^[0-9]+$|regmix' | grep -v batch | tail -40
echo "=== recent FAILED chunk logs ==="
ls -t "$RUN"/logs/lm-chunk-*.err 2>/dev/null | head -3 | while read -r f; do
  echo "--- $f ---"
  tail -15 "$f"
done
echo "=== newest done marker ==="
find "$RUN/lm_labels/labels/docs" -name '*.done' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -3
echo "=== chunk job states since 16:00 UTC ==="
sacct -u nzhao2 -S 2026-07-29T16:00 --name=regmix-lm-chunk --format=State -P -n 2>/dev/null | sort | uniq -c
REMOTE
