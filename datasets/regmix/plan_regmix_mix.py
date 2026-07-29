#!/usr/bin/env python3
"""Plan a RegMix-aligned 10B mix by randomly sampling shards from olmohq S3.

Lists objects under s3://SRC_BUCKET/SRC_PREFIX/data/, then for each domain
randomly selects whole shards (byte-proportional token estimates) until the
estimated tokens reach OVERSHOOT_FACTOR * target. Document-level trim later
hits the exact budget.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import boto3

# RegMix-optimized Pile weights mapped onto OLMo-mix domains.
DOMAIN_TARGETS: dict[str, int] = {
    "dclm": 3_750_000_000,
    "arxiv": 2_500_000_000,
    "starcoder": 1_406_000_000,
    "pes2o": 938_000_000,
    "open-web-math": 635_000_000,
    "algebraic-stack": 615_000_000,
    "wiki": 156_000_000,
}

DOMAIN_ORDER = list(DOMAIN_TARGETS.keys())
OVERSHOOT_FACTOR = 1.15  # shard sample slightly over target; trim brings exact


def is_data_object(key: str) -> bool:
    name = key.rsplit("/", 1)[-1]
    if name.endswith(".done"):
        return False
    return name.endswith(
        (".json.gz", ".jsonl.gz", ".jsonl.zstd", ".jsonl.zst", ".jsonl")
    )


def domain_for_key(key: str, prefix: str) -> str | None:
    rel = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else key
    # expect data/<domain>/...
    parts = rel.split("/")
    if len(parts) < 2 or parts[0] != "data":
        return None
    domain = parts[1]
    return domain if domain in DOMAIN_TARGETS else None


def list_domain_files_s3(
    s3, bucket: str, prefix: str
) -> dict[str, list[dict]]:
    paginator = s3.get_paginator("list_objects_v2")
    by_domain: dict[str, list[dict]] = defaultdict(list)
    data_prefix = f"{prefix.rstrip('/')}/data/"
    for page in paginator.paginate(Bucket=bucket, Prefix=data_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not is_data_object(key):
                continue
            domain = domain_for_key(key, prefix.rstrip("/") + "/")
            if domain is None:
                domain = domain_for_key(key, prefix)
            if domain is None:
                continue
            rel = key[len(prefix.rstrip("/")) + 1 :]  # e.g. data/arxiv/...
            by_domain[domain].append(
                {
                    "key": key,
                    "path": rel,
                    "size": int(obj["Size"]),
                    "domain": domain,
                }
            )
    for domain in by_domain:
        by_domain[domain].sort(key=lambda x: x["path"])
    return by_domain


def list_domain_files_manifest(manifest_path: Path, prefix: str) -> dict[str, list[dict]]:
    """Load shard inventory from an existing olmohq-style manifest.jsonl."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain")
        if domain not in DOMAIN_TARGETS:
            continue
        rel = row["path"]
        by_domain[domain].append(
            {
                "key": row.get("key") or f"{prefix.rstrip('/')}/{rel}",
                "path": rel,
                "size": int(row["size"]),
                "domain": domain,
            }
        )
    for domain in by_domain:
        by_domain[domain].sort(key=lambda x: x["path"])
    return by_domain


def list_domain_files_local(local_root: Path, prefix: str) -> dict[str, list[dict]]:
    """List shards from a local mirror of s3://bucket/prefix/ (root contains data/)."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    data_root = local_root / "data"
    if not data_root.is_dir():
        raise SystemExit(f"missing local data root {data_root}")
    # Resolve domain dirs so symlinks (e.g. dclm -> prior run) are followed.
    for domain_dir in sorted(data_root.iterdir()):
        if domain_dir.name not in DOMAIN_TARGETS:
            continue
        domain = domain_dir.name
        root = domain_dir.resolve()
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if not is_data_object(path.name):
                continue
            try:
                under = path.relative_to(root).as_posix()
            except ValueError:
                continue
            rel = f"data/{domain}/{under}"
            key = f"{prefix.rstrip('/')}/{rel}"
            by_domain[domain].append(
                {
                    "key": key,
                    "path": rel,
                    "size": path.stat().st_size,
                    "domain": domain,
                }
            )
    for domain in by_domain:
        by_domain[domain].sort(key=lambda x: x["path"])
    return by_domain


def greedy_random_sample(
    files: list[dict],
    target_tokens: float,
    domain_total_tokens: float,
    rng: random.Random,
) -> list[dict]:
    """Sample whole shards randomly until estimated tokens >= target."""
    if not files or target_tokens <= 0:
        return []
    total_bytes = sum(f["size"] for f in files)
    if total_bytes <= 0:
        return []
    shuffled = files[:]
    rng.shuffle(shuffled)
    selected: list[dict] = []
    est = 0.0
    for f in shuffled:
        tokens = domain_total_tokens * (f["size"] / total_bytes)
        item = dict(f)
        item["est_tokens"] = tokens
        selected.append(item)
        est += tokens
        if est >= target_tokens:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-bucket", default="edullm-datasets")
    parser.add_argument("--src-prefix", default="olmo100b/olmo-mix-1124-30b")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overshoot", type=float, default=OVERSHOOT_FACTOR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--local-data-root",
        type=Path,
        default=None,
        help="If set, list shards from this local mirror instead of calling S3",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="If set, use this olmohq manifest.jsonl as the shard inventory (preferred)",
    )
    parser.add_argument(
        "--pool-summary",
        type=Path,
        default=None,
        help="Optional local copy of olmohq plan/summary.json for pool token totals",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer published pool sizes from the olmohq summary when available.
    pool_tokens: dict[str, float] = {}
    pool_bytes: dict[str, int] = {}
    if args.pool_summary and args.pool_summary.exists():
        pool = json.loads(args.pool_summary.read_text(encoding="utf-8"))
        for domain, block in pool.get("domains", {}).items():
            if domain in DOMAIN_TARGETS:
                pool_tokens[domain] = float(
                    block.get("est_tokens_selected")
                    or block.get("target_tokens")
                    or 0
                )
                # Pass-through domains may have est_tokens_selected=0 in the
                # upsample summary; fall back to full_pool / target tokens.
                if pool_tokens[domain] <= 0:
                    pool_tokens[domain] = float(
                        block.get("full_pool_tokens")
                        or block.get("target_tokens")
                        or 0
                    )
                pool_bytes[domain] = int(block.get("bytes_selected") or 0)

    if args.source_manifest is not None:
        print(f"loading inventory from {args.source_manifest}", flush=True)
        by_domain = list_domain_files_manifest(args.source_manifest, args.src_prefix)
    elif args.local_data_root is not None:
        print(f"listing local mirror {args.local_data_root}/data/ ...", flush=True)
        by_domain = list_domain_files_local(args.local_data_root, args.src_prefix)
    else:
        s3 = boto3.client("s3", region_name=args.region)
        print(f"listing s3://{args.src_bucket}/{args.src_prefix}/data/ ...", flush=True)
        by_domain = list_domain_files_s3(s3, args.src_bucket, args.src_prefix)

    # Fill pool totals from listed bytes if summary missing / zero.
    for domain, files in by_domain.items():
        bytes_total = sum(f["size"] for f in files)
        pool_bytes.setdefault(domain, bytes_total)
        if domain not in pool_tokens or pool_tokens[domain] <= 0:
            # Fallback: assume ~1 token per 4 compressed bytes (rough); prefer summary.
            # Better fallback: use olmohq known totals from earlier inspection.
            defaults = {
                "dclm": 28_577_878_778.0,
                "starcoder": 20_274_042_335.0,
                "pes2o": 26_262_621_379.0,
                "arxiv": 20_800_000_000.0,
                "open-web-math": 12_200_000_000.0,
                "algebraic-stack": 11_800_000_000.0,
                "wiki": 3_660_000_000.0,
            }
            pool_tokens[domain] = defaults.get(domain, bytes_total / 4.0)

    rng = random.Random(args.seed)
    manifest_rows: list[dict] = []
    summary_domains: dict[str, dict] = {}

    for domain in DOMAIN_ORDER:
        files = by_domain.get(domain, [])
        if not files:
            raise SystemExit(f"no S3 objects for domain {domain}")
        target = DOMAIN_TARGETS[domain]
        sample_target = target * args.overshoot
        chosen = greedy_random_sample(
            files, sample_target, pool_tokens[domain], rng
        )
        # Ensure at least one shard even for tiny budgets.
        if not chosen:
            chosen = [dict(files[0], est_tokens=pool_tokens[domain] * (files[0]["size"] / sum(f["size"] for f in files)))]

        for i, item in enumerate(chosen):
            row = {
                "index": len(manifest_rows),
                "domain": domain,
                "path": item["path"],
                "key": item["key"],
                "size": item["size"],
                "est_tokens": item["est_tokens"],
                "shard_rank": i,
            }
            manifest_rows.append(row)

        est = sum(c["est_tokens"] for c in chosen)
        summary_domains[domain] = {
            "target_tokens": target,
            "weight": target / sum(DOMAIN_TARGETS.values()),
            "sample_target_tokens": sample_target,
            "files_available": len(files),
            "bytes_available": sum(f["size"] for f in files),
            "pool_tokens": pool_tokens[domain],
            "files_selected": len(chosen),
            "bytes_selected": sum(c["size"] for c in chosen),
            "est_tokens_selected": est,
            "selection_method": "greedy-random-shard-bytes",
        }
        print(
            f"{domain:16} target={target/1e9:.3f}B  "
            f"shards={len(chosen)}/{len(files)}  "
            f"est={est/1e9:.3f}B  bytes={sum(c['size'] for c in chosen)/1e9:.2f}GB",
            flush=True,
        )

    summary = {
        "kind": "regmix-optimized-10b",
        "source_bucket": args.src_bucket,
        "source_prefix": args.src_prefix,
        "total_target_tokens": sum(DOMAIN_TARGETS.values()),
        "seed": args.seed,
        "overshoot_factor": args.overshoot,
        "tokenizer": "allenai/dolma2-tokenizer",
        "eos_token_id": 100257,
        "domain_targets": DOMAIN_TARGETS,
        "files_selected": len(manifest_rows),
        "bytes_selected": sum(r["size"] for r in manifest_rows),
        "est_tokens_selected": sum(r["est_tokens"] for r in manifest_rows),
        "domains": summary_domains,
        "note": (
            "Whole-shard random sample from olmohq with ~15% overshoot; "
            "document-shuffle trim brings each domain to exact RegMix-mapped budget."
        ),
    }

    man_path = args.out_dir / "manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out_dir / "domain_targets.json").write_text(
        json.dumps(DOMAIN_TARGETS, indent=2), encoding="utf-8"
    )
    print(f"wrote {man_path} ({len(manifest_rows)} shards)", flush=True)
    print(f"est_tokens_selected={summary['est_tokens_selected']:,.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
