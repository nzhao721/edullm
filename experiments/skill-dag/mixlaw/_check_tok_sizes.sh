#!/usr/bin/env bash
set -Eeuo pipefail
SOCK=/tmp/farmshare-nzhao2.sock
HOST=nzhao2@login.farmshare.stanford.edu
ssh -S "$SOCK" -o BatchMode=yes "$HOST" bash -s <<'REMOTE'
set -Eeuo pipefail
TOP=/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841
MIX=/scratch/users/nzhao2/agent-runs/mixlaw-validation-10b-20260728-190236
echo "=== mixlaw receipt / READY ==="
cat "$MIX/mixlaw_upload_receipt.json" 2>/dev/null | head -40
cat "$MIX/READY" 2>/dev/null || true
echo "=== remaining tok input sizes ==="
python3 - <<'PY'
from pathlib import Path
top = Path("/scratch/users/nzhao2/agent-runs/olmohq-topup-20260728-185841")
lines = (top/"tokenize_map.txt").read_text().splitlines()
for i in [401,402,403,404,406,408,409]:
    src, dst = lines[i].split("|",1)
    sp, dp = Path(src), Path(dst)
    print(f"i={i} src_GiB={sp.stat().st_size/1e9:.2f} exists={sp.is_file()} dst_exists={dp.is_file()} dst_tmp={list(dp.parent.glob(dp.name+'*'))[:3]}")
PY
echo "=== squeue tok ==="
squeue -j 1666301 -o '%.18i %.9P %.8T %.10M %R' 2>/dev/null | head -20
REMOTE
