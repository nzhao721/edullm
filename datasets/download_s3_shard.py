#!/usr/bin/env python3
"""Download one shard from S3, preferring a verified local mirror when present.

Used as a FarmShare Slurm array worker.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--src-bucket", default="edullm-datasets")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--local-mirror",
        type=Path,
        default=None,
        help="Optional existing scratch tree with the same relative paths",
    )
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    if args.index < 0 or args.index >= len(lines):
        print(f"index {args.index} out of range 0..{len(lines)-1}", file=sys.stderr)
        return 2
    item = json.loads(lines[args.index])
    rel = item["path"]
    key = item.get("key") or f"olmo100b/olmo-mix-1124-30b/{rel}"
    expected = int(item["size"])
    dest = args.local_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = Path(str(dest) + ".done")

    if dest.exists() and done.exists() and dest.stat().st_size == expected:
        print(f"skip existing {rel}")
        return 0

    # Prefer hardlink/copy from local mirror of olmohq when size matches.
    if args.local_mirror is not None:
        candidates = [
            args.local_mirror / rel,
            args.local_mirror / "data" / rel,  # if mirror root is run/data
        ]
        for src in candidates:
            if src.exists() and src.stat().st_size == expected:
                if dest.exists():
                    dest.unlink()
                try:
                    os.link(src, dest)
                    method = "hardlink"
                except OSError:
                    shutil.copy2(src, dest)
                    method = "copy"
                done.write_text(f"{method}\n", encoding="utf-8")
                print(f"{method} {rel} bytes={dest.stat().st_size}")
                return 0

    s3 = boto3.client("s3", region_name=args.region)
    cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    print(f"download s3://{args.src_bucket}/{key} -> {dest}", flush=True)
    s3.download_file(args.src_bucket, key, str(tmp), Config=cfg)
    got = tmp.stat().st_size
    if got != expected:
        tmp.unlink(missing_ok=True)
        print(f"size mismatch for {rel}: got {got} expected {expected}", file=sys.stderr)
        return 1
    tmp.replace(dest)
    done.write_text("s3\n", encoding="utf-8")
    print(f"downloaded {rel} bytes={got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
