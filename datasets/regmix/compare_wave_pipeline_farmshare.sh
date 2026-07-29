#!/usr/bin/env bash
SOCKET=/tmp/farmshare-nzhao2.sock
ssh -S "$SOCKET" -o BatchMode=yes nzhao2@login.farmshare.stanford.edu bash <<'REMOTE'
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
RUN=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810
LM=$RUN/lm_labels
echo "=== completions by job type (all time) ==="
for pat in lm-label lm-retry lm-chunk; do
  n=$(grep -l '"event": "labeled_lm_chunk"' "$RUN"/logs/${pat}-*.out 2>/dev/null | wc -l | tr -d ' ')
  echo "$pat success_logs=$n"
done
echo "=== sacct outcomes since pipeline controller 1667438 (16:12 UTC) ==="
sacct -S 2026-07-29T16:12:00 -u nzhao2 --name=regmix-lm-chunk --format=State -P -n 2>/dev/null | sort | uniq -c
echo "=== sacct outcomes same window for regmix-lm-retry (wave) ==="
sacct -S 2026-07-29T14:00:00 -E 2026-07-29T16:12:00 -u nzhao2 --name=regmix-lm-retry --format=State -P -n 2>/dev/null | sort | uniq -c
echo "=== sample COMPLETED chunk jobs (pipeline) ==="
sacct -u nzhao2 --name=regmix-lm-chunk --state=COMPLETED --format=JobID,Elapsed,MaxRSS -P -n 2>/dev/null | tail -5
echo "=== sample FAILED chunk jobs ==="
sacct -u nzhao2 --name=regmix-lm-chunk --state=FAILED --format=JobID,Elapsed,ExitCode -P -n 2>/dev/null | tail -8
echo "=== duplicate inflight check (same chunk multiple running?) ==="
if [ -f "$LM/pipeline_inflight.tsv" ]; then
  awk -F'\t' '{print $2}' "$LM/pipeline_inflight.tsv" | sort | uniq -c | awk '$1>1{print}'
  echo "inflight_lines=$(wc -l < "$LM/pipeline_inflight.tsv")"
fi
echo "=== missing chunk indices (first 20) ==="
head -20 "$LM/retry_indices.txt"
REMOTE
