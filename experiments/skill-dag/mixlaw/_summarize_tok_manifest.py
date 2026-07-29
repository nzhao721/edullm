#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

p = Path(
    r"C:\Users\natha\.cursor\projects\c-alpha-ai-edullm\agent-tools\5f162c68-99be-46f7-92cd-4c85d201a3b1.txt"
)
r = json.loads(p.read_text(encoding="utf-8"))
m = json.loads(r["stdout"])
print("total", m["total_content_tokens"] / 1e9)
by: dict = defaultdict(lambda: {"tokens": 0, "n": 0, "bytes": 0})
for s in m["shards"]:
    d = s["domain"]
    by[d]["tokens"] += s["tokens"]
    by[d]["n"] += 1
    by[d]["bytes"] += s["bytes"]
for d, v in sorted(by.items()):
    print(f"{d}: n={v['n']} tokens={v['tokens']/1e9:.3f}B bytes={v['bytes']/1e9:.2f}GB")
print("sample keys", sorted(m["shards"][0].keys()))
print("sample", m["shards"][0])
# save a local copy for FarmShare scripts to use offline during development
Path("olmohq_tokenized_manifest.json").write_text(json.dumps(m) + "\n", encoding="utf-8")
print("wrote olmohq_tokenized_manifest.json")
