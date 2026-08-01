#!/usr/bin/env bash
set -euo pipefail
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
echo "=== squeue ==="
squeue -u nzhao2 -o '%.18i %.10j %.2t %.10M %R' | grep -E 'JOBID|colmlm' || true
echo "=== entries out ==="
cat "$RUN/logs/entries-1671526.out" 2>/dev/null || true
echo "=== entries err ==="
cat "$RUN/logs/entries-1671526.err" 2>/dev/null || true
echo "=== s100 sample outs with content ==="
for f in "$RUN"/logs/s100-*.out; do
  if [[ -s "$f" ]]; then
    echo "-- $(basename "$f") --"
    cat "$f"
  fi
done | head -120
echo "=== s100 sample errs with content ==="
for f in "$RUN"/logs/s100-*.err; do
  if [[ -s "$f" ]]; then
    echo "-- $(basename "$f") --"
    cat "$f"
  fi
done | head -160
echo "=== data dir ==="
ls -la "$RUN/data" 2>/dev/null || true
ls "$RUN/data/s100" 2>/dev/null | wc -l
du -sh "$RUN/data" 2>/dev/null || true
