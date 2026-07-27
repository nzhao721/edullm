#!/usr/bin/env python3
"""Upload per-shard dolma2 tokenized memmaps to the source OLMo-mix S3 prefix."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def aws_s3_sync(local: Path, uri: str) -> None:
    cmd = ["aws", "s3", "sync", str(local), uri, "--only-show-errors"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = args.run_dir
    tokenized_root = run / "tokenized"
    map_file = run / "tokenize_map.txt"
    index_path = run / "plan" / "tokenize_index.jsonl"
    index_by_npy: dict[str, dict] = {}
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index_by_npy[row["npy"]] = row

    if not map_file.is_file():
        raise SystemExit(f"missing map file: {map_file}")

    shard_reports: list[dict] = []
    failures: list[str] = []
    total_tokens = 0

    for line in map_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _inp, out = line.split("|", 1)
        out_path = Path(out)
        meta_path = out_path.with_suffix(".json")
        if not out_path.is_file():
            failures.append(f"missing npy: {out_path}")
            continue
        rel = out_path.relative_to(tokenized_root).as_posix()
        idx_row = index_by_npy.get(rel, {})
        tokens = 0
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tokens = int(meta.get("tokens") or 0)
        else:
            failures.append(f"missing meta: {meta_path}")
        total_tokens += tokens
        shard_reports.append(
            {
                "path": rel,
                "manifest_path": idx_row.get("manifest_path"),
                "domain": idx_row.get("domain"),
                "tokens": tokens,
                "bytes": out_path.stat().st_size,
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_id": "allenai/dolma2-tokenizer",
        "eos_token_id": 100257,
        "s3_bucket": args.bucket,
        "s3_prefix": f"{args.prefix.strip('/')}/tokenized",
        "total_content_tokens": total_tokens,
        "shards": shard_reports,
        "failures": failures,
        "accepted": not failures,
    }
    plan_dir = run / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plan_dir / "tokenized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"shards": len(shard_reports), "total_tokens": total_tokens, "failures": len(failures)}, indent=2))

    if failures:
        print(f"ACCEPTANCE FAILED: {len(failures)} shard(s)", flush=True)
        return 1
    if args.dry_run:
        print("skipping S3 upload", flush=True)
        return 0

    aws_s3_sync(tokenized_root, f"s3://{args.bucket}/{args.prefix.strip('/')}/tokenized/")
    manifest_uri = f"s3://{args.bucket}/{args.prefix.strip('/')}/plan/tokenized_manifest.json"
    subprocess.run(
        ["aws", "s3", "cp", str(manifest_path), manifest_uri, "--only-show-errors"],
        check=True,
    )
    if index_path.is_file():
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(index_path),
                f"s3://{args.bucket}/{args.prefix.strip('/')}/plan/tokenize_index.jsonl",
                "--only-show-errors",
            ],
            check=True,
        )
    print(f"upload complete -> s3://{args.bucket}/{args.prefix.strip('/')}/tokenized/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
