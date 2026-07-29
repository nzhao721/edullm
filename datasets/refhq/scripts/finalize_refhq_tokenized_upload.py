#!/usr/bin/env python3
"""Upload dolma2 tokenized .npy memmaps to S3."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _aws_s3_sync(local: Path, uri: str) -> None:
    cmd = ["aws", "s3", "sync", str(local), uri, "--only-show-errors"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bucket = args.bucket or plan.get("s3_bucket") or "edullm-datasets"
    prefix = args.prefix or plan.get("s3_prefix") or "refhq/refhq-regmix-5p5b-v1"
    scratch_root = Path(plan["scratch_root"])
    tokenized_root = scratch_root / "tokenized"

    reports: dict[str, dict] = {}
    failures: list[str] = []
    total_stream_tokens = 0

    for domain in plan["domains"]:
        meta_path = tokenized_root / domain / f"{domain}.json"
        npy_path = tokenized_root / domain / f"{domain}.npy"
        if not meta_path.is_file() or not npy_path.is_file():
            failures.append(f"{domain}: missing {npy_path} or {meta_path}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stream_tokens = int(meta.get("stream_tokens_with_eos") or 0)
        reports[domain] = {
            **meta,
            "bytes": npy_path.stat().st_size,
            "within_expected": stream_tokens > 0,
        }
        total_stream_tokens += stream_tokens

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_id": "allenai/dolma2-tokenizer",
        "eos_token_id": 100257,
        "s3_bucket": bucket,
        "s3_prefix": f"{prefix}/tokenized",
        "total_stream_tokens_with_eos": total_stream_tokens,
        "domains": reports,
        "failures": failures,
        "accepted": not failures,
    }
    manifest_path = scratch_root / "manifests" / "tokenized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if failures:
        print(f"ACCEPTANCE FAILED: {len(failures)} domain(s)", flush=True)
        return 1

    if args.dry_run:
        print("skipping S3 upload", flush=True)
        return 0

    for domain in plan["domains"]:
        local = tokenized_root / domain
        uri = f"s3://{bucket}/{prefix}/tokenized/{domain}/"
        _aws_s3_sync(local, uri)

    _aws_s3_sync(manifest_path.parent, f"s3://{bucket}/{prefix}/manifests/")
    print(f"upload complete -> s3://{bucket}/{prefix}/tokenized/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
