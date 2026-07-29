#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
TOK=$(cat "$TOP/tokenize_retry_job_id.txt")
UP=$(cat "$TOP/upload_job_id.txt")
echo "=== tok retry $TOK ==="
sacct -j "$TOK" --format=JobID,State,ExitCode,Elapsed,MaxRSS -n -P | awk -F'|' '$1 ~ /^[0-9]+_[0-9]+$/ {print}'
echo "=== missing npy ==="
python3 - <<'PY'
from pathlib import Path
top = Path("/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841")
lines = (top/"tokenize_map.txt").read_text().splitlines()
missing=[i for i,l in enumerate(lines) if not Path(l.split("|",1)[1]).is_file()]
print(f"missing={missing} count={len(missing)} total_npy={sum(1 for l in lines if Path(l.split('|',1)[1]).is_file())}/{len(lines)}")
PY
echo "=== topup-up $UP ==="
sacct -j "$UP" --format=JobID,State,ExitCode,Elapsed -n | head -5
tail -n 20 "$TOP/logs/upload-${UP}.out" 2>/dev/null || true
tail -n 10 "$TOP/logs/upload-${UP}.err" 2>/dev/null || true
REMOTE
