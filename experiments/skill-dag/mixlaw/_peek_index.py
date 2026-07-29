#!/usr/bin/env python3
import json
from pathlib import Path

p = Path(r"C:\Users\natha\.cursor\projects\c-alpha-ai-edullm\agent-tools\71045e60-81ee-4e0f-8a09-23c23f6c6796.txt")
resp = json.loads(p.read_text(encoding="utf-8"))
stdout = resp["stdout"]
lines = stdout.splitlines()
print("n_lines", len(lines))
for line in lines[:3]:
    print(json.dumps(json.loads(line), indent=2))
# domain token totals from index
from collections import defaultdict
tok = defaultdict(int)
n = defaultdict(int)
for line in lines:
    if not line.strip():
        continue
    row = json.loads(line)
    d = row.get("domain")
    t = int(row.get("tokens") or 0)
    if d:
        tok[d] += t
        n[d] += 1
print("domains:")
for d, t in sorted(tok.items(), key=lambda x: -x[1]):
    print(f"  {d}: shards={n[d]} tokens={t/1e9:.3f}B")
