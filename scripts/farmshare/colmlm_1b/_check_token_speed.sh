#!/usr/bin/env bash
set -euo pipefail
RUN=/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745
echo "=== hf_token file ==="
ls -la "$RUN/hf_token" 2>/dev/null || echo missing
wc -c "$RUN/hf_token" 2>/dev/null || true
# show only length/prefix fingerprint, never full token
python3 - <<'PY'
from pathlib import Path
p = Path("/scratch/users/nzhao2/agent-runs/colmlm-1b-corpus-20260801-124745/hf_token")
if p.exists():
    t = p.read_text().strip()
    print(f"token_len={len(t)} prefix={t[:4]}… suffix=…{t[-4:]}")
else:
    print("no token")
PY
echo "=== env.sh token stanza ==="
grep -n HF_TOKEN "$RUN/env.sh" || true
echo "=== sourced token present? ==="
source "$RUN/env.sh"
if [[ -n "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN_len=${#HF_TOKEN}"; else echo "HF_TOKEN empty"; fi
echo "=== entries partial size over time ==="
ls -la "$RUN/data/fineweb_with_fullwiki_entries.db"* 2>/dev/null || true
sleep 20
ls -la "$RUN/data/fineweb_with_fullwiki_entries.db"* 2>/dev/null || true
echo "=== s100 parquet count / size ==="
find "$RUN/data/s100" -name '*.parquet' | wc -l
du -sh "$RUN/data/s100" 2>/dev/null || true
echo "=== entries log tail ==="
tail -20 "$RUN/logs/entries-1671526.out"
