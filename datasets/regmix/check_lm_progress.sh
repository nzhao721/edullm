#!/bin/bash
ROOT=/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/lm_labels
LABELS=$ROOT/labels
N=$(wc -l < "$ROOT/lm_work_manifest.jsonl" | tr -d ' ')
DONE=$(find "$LABELS/docs" -name '*.done' 2>/dev/null | wc -l | tr -d ' ')
METRICS=$(find "$LABELS/metrics" -name '*.metrics.jsonl.gz' 2>/dev/null | wc -l | tr -d ' ')
echo "chunks_total=$N"
echo "done_markers=$DONE"
echo "metrics_shards=$METRICS"
if [ -f "$LABELS/READY" ]; then
  echo READY_OK
  cat "$LABELS/READY"
else
  echo READY_missing
fi
echo by_domain_done:
for d in "$LABELS"/docs/*/; do
  [ -d "$d" ] || continue
  echo "  $(basename "$d"): $(ls "$d"*.done 2>/dev/null | wc -l | tr -d ' ')"
done
echo controller_log_tail:
tail -n 25 /scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/logs/lm-ctrl-1664949.out 2>/dev/null
echo failed_tasks_sample:
sacct -j 1664949,1664951,1664995 --format=JobID,State,ExitCode,Elapsed -P -n 2>/dev/null | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2 !~ /COMPLETED/ {print}' | head -20
