#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
echo "=== running tok tasks ==="
sacct -j 1666301 --format=JobID,State,Elapsed,NodeList -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2=="RUNNING" {print}'
echo "=== sample stuck logs (last lines) ==="
for t in $(sacct -j 1666301 --format=JobID,State -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ && $2=="RUNNING" {split($1,a,"_"); print a[2]}' | head -5); do
  echo "--- task $t ---"
  tail -n 8 "$TOP/logs/tok-1666301_${t}.out" 2>/dev/null || true
  tail -n 8 "$TOP/logs/tok-1666301_${t}.err" 2>/dev/null || true
done
echo "=== npy count vs expected 413 ==="
find "$TOP/tokenized/shards" -name '*.npy' 2>/dev/null | wc -l
# which map lines missing?
python3 - <<'PY'
from pathlib import Path
top = Path("/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841")
lines = (top/"tokenize_map.txt").read_text().splitlines()
missing = []
for i, line in enumerate(lines):
    dst = line.split("|",1)[1]
    if not Path(dst).is_file():
        missing.append(i)
print(f"map_lines={len(lines)} missing_npy={len(missing)} first_missing={missing[:20]}")
PY
REMOTE
