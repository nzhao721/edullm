#!/usr/bin/env bash
set -euo pipefail
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
echo "=== $(date -u +%FT%TZ) colmlm-1b status ==="
echo "=== queue ==="
squeue -u nzhao2 -o '%.18i %.12j %.2t %.10M %R' | grep -E 'JOBID|colmlm' || true
echo "=== entries ==="
ENTRIES=$(cat "$RUN/job_entries.txt" 2>/dev/null || echo "")
if [[ -n "${ENTRIES}" ]]; then
  tail -8 "$RUN/logs/entries-${ENTRIES}.out" 2>/dev/null || true
  ls -lh "$RUN/data/fineweb_with_fullwiki_entries.db"* 2>/dev/null || true
fi
echo "=== s100 ==="
n=$(find "$RUN/data/s100" -name '*.parquet' 2>/dev/null | wc -l)
echo "parquet_count=${n} / 140"
du -sh "$RUN/data/s100" 2>/dev/null || true
echo "=== docs ==="
if [[ -f "$RUN/job_docs.txt" ]]; then
  DOCS=$(cat "$RUN/job_docs.txt")
  echo "docs_job=${DOCS}"
  sacct -j "$DOCS" -n -o State,ExitCode,Elapsed -P 2>/dev/null | head -3 || true
  ls -lh "$RUN/data/docs_sample.parquet" 2>/dev/null || echo "docs_sample.parquet: not yet"
  cat "$RUN/data/docs_stats.json" 2>/dev/null | head -5 || true
fi
echo "=== spans ==="
if [[ -f "$RUN/job_spans.txt" ]]; then
  SPANS=$(cat "$RUN/job_spans.txt")
  echo "spans_job=${SPANS}"
  sacct -j "$SPANS" -n -o State -P 2>/dev/null | sort | uniq -c || true
  nsp=$(find "$RUN/data/spans" -name 'spans_*.parquet' 2>/dev/null | wc -l)
  echo "span_shards=${nsp}"
  du -sh "$RUN/data/spans" 2>/dev/null || true
fi
echo "=== mark / corpus ==="
if [[ -f "$RUN/job_mark.txt" ]]; then
  MARK=$(cat "$RUN/job_mark.txt")
  echo "mark_job=${MARK}"
  sacct -j "$MARK" -n -o State,ExitCode,Elapsed -P 2>/dev/null | head -3 || true
fi
du -sh "$RUN/corpus_1b" 2>/dev/null || echo "corpus_1b: not yet"
ls -lh "$RUN/qa_report.json" 2>/dev/null || echo "qa_report.json: not yet"
echo "=== watch ==="
if [[ -f "$RUN/job_watch.txt" ]]; then
  WATCH=$(cat "$RUN/job_watch.txt")
  sacct -j "$WATCH" -n -o State,ExitCode,Elapsed -P 2>/dev/null | head -3 || true
  tail -12 "$RUN/logs/watch-${WATCH}.out" 2>/dev/null || true
fi
echo "=== data total ==="
du -sh "$RUN/data" 2>/dev/null || true
