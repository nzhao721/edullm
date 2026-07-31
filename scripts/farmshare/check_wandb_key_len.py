#!/usr/bin/env python3
from pathlib import Path
p = Path("/mnt/c/Users/natha/.wandb_api_key")
if not p.exists():
    print("missing_key_file")
    raise SystemExit(1)
k = p.read_text(encoding="utf-8").strip()
print(f"key_len={len(k)}")
print(f"looks_like_uuid={k.count('-') == 4 and len(k) == 36}")
print(f"valid_len={len(k) >= 40}")
