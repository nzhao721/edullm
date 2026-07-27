#!/usr/bin/env python3
"""Download one OLMo-mix shard from Hugging Face into local scratch.

Used as a Slurm array worker. Does not touch AWS.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="allenai/olmo-mix-1124")
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    if args.index < 0 or args.index >= len(lines):
        print(f"index {args.index} out of range 0..{len(lines)-1}", file=sys.stderr)
        return 2
    item = json.loads(lines[args.index])
    rel = item["path"]
    dest = args.local_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    done_marker = dest.with_suffix(dest.suffix + ".done")
    if dest.exists() and done_marker.exists() and dest.stat().st_size == item["size"]:
        print(f"skip existing {rel}")
        return 0

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=rel,
        local_dir=str(args.local_root),
        local_dir_use_symlinks=False,
    )
    got = Path(path)
    if got.stat().st_size != item["size"]:
        print(f"size mismatch for {rel}: got {got.stat().st_size} expected {item['size']}", file=sys.stderr)
        return 1
    done_marker.write_text("ok\n", encoding="utf-8")
    print(f"downloaded {rel} bytes={got.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
