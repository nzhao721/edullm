#!/usr/bin/env python3
"""Upload one downloaded OLMo-mix shard from scratch to S3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="olmo-mix-1124-30b")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    if args.index < 0 or args.index >= len(lines):
        print(f"index {args.index} out of range", file=sys.stderr)
        return 2
    item = json.loads(lines[args.index])
    rel = item["path"]
    local = args.local_root / rel
    if not local.exists():
        print(f"missing local file {local}", file=sys.stderr)
        return 1

    key = f"{args.prefix.rstrip('/')}/{rel}"
    s3 = boto3.client("s3", region_name=args.region)
    # Skip if already present with matching size.
    try:
        head = s3.head_object(Bucket=args.bucket, Key=key)
        if head["ContentLength"] == local.stat().st_size:
            print(f"skip existing s3://{args.bucket}/{key}")
            return 0
    except ClientError:
        pass

    config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )
    extra = {
        "Metadata": {
            "domain": item.get("domain", ""),
            "est-tokens": str(int(item.get("est_tokens", 0))),
            "source-repo": "allenai/olmo-mix-1124",
            "sample-seed": "see-manifest",
        }
    }
    s3.upload_file(str(local), args.bucket, key, ExtraArgs=extra, Config=config)
    print(f"uploaded s3://{args.bucket}/{key} bytes={local.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
