#!/usr/bin/env bash
set -euo pipefail
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
STAGING=/scratch/users/nzhao2/agent-runs/edullm-farmshare-staging/scripts/farmshare/colmlm_1b
echo "=== watch job ==="
cat "$RUN/job_watch.txt" 2>/dev/null || true
W=$(cat "$RUN/job_watch.txt" 2>/dev/null || true)
sacct -j "$W" -o JobID,State,ExitCode,Elapsed,NodeList -P 2>/dev/null || true
echo "=== watch out ==="
tail -80 "$RUN/logs/watch-${W}.out" 2>/dev/null || true
echo "=== watch err ==="
tail -80 "$RUN/logs/watch-${W}.err" 2>/dev/null || true
echo "=== s100 sacct sample ==="
sacct -j 1671527 -o JobID,State,ExitCode -P | head -50
echo "failed count:" $(sacct -j 1671527 -n -o State -P | grep -cE 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL' || true)
echo "=== parquet ==="
find "$RUN/data/s100" -name '*.parquet' | wc -l
echo "=== docs job? ==="
cat "$RUN/job_docs.txt" 2>/dev/null || echo none
ls -la "$RUN/data/docs_sample.parquet" 2>/dev/null || echo no_docs_yet
