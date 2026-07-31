#!/usr/bin/env bash
# Sync p3math code to FarmShare staging and submit download -> filter jobs.
set -euo pipefail

SOCK="${SOCKET:-/tmp/farmshare-nzhao2.sock}"
HOST="${HOST:-nzhao2@login.farmshare.stanford.edu}"
SUNET="${SUNET:-nzhao2}"
STAGING="${STAGING:-/scratch/users/${SUNET}/agent-runs/edullm-farmshare-staging}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/${SUNET}/agent-runs/p3math-${STAMP}}"
REPO_ROOT="$STAGING"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# On Windows/Git bash, prefer WSL for ssh control socket.
SSH=(ssh -S "$SOCK" -o BatchMode=yes "$HOST")

echo "[sync] staging=$STAGING scratch=$SCRATCH_ROOT"

"${SSH[@]}" "mkdir -p '$STAGING/datasets/p3math/tests' '$SCRATCH_ROOT/logs' '$SCRATCH_ROOT/manifests'"

# Copy package files
for f in \
  datasets/p3math/__init__.py \
  datasets/p3math/filters.py \
  datasets/p3math/download_sources.py \
  datasets/p3math/filter_corpus.py \
  datasets/p3math/download.sbatch \
  datasets/p3math/filter.sbatch \
  datasets/p3math/README.md \
  datasets/p3math/tests/__init__.py \
  datasets/p3math/tests/test_filters.py
do
  scp -o "ControlPath=$SOCK" -o BatchMode=yes "$ROOT/$f" "$HOST:$STAGING/$f"
done

echo "[submit] download + filter"
JOB_OUT="$("${SSH[@]}" "cd '$SCRATCH_ROOT' && \
  export SCRATCH_ROOT='$SCRATCH_ROOT' REPO_ROOT='$REPO_ROOT' && \
  DL=\$(sbatch --exclude=wheat-01 --export=ALL,SCRATCH_ROOT,REPO_ROOT \
    '$REPO_ROOT/datasets/p3math/download.sbatch' | awk '{print \$4}') && \
  FL=\$(sbatch --exclude=wheat-01 --dependency=afterok:\$DL --export=ALL,SCRATCH_ROOT,REPO_ROOT \
    '$REPO_ROOT/datasets/p3math/filter.sbatch' | awk '{print \$4}') && \
  echo SCRATCH_ROOT=$SCRATCH_ROOT && echo DOWNLOAD_JOB=\$DL && echo FILTER_JOB=\$FL")"

echo "$JOB_OUT"
