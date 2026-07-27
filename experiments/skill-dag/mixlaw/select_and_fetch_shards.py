#!/usr/bin/env python3
"""Select and download only the olmohq shards needed for the working pool.

The full ``olmo-mix-1124-30b/data/`` tree is ~130 GiB. At the default compute
budget we only need ~0.6B tokens of raw text (~a few GiB compressed). This
script:

  1. loads the lightweight shard inventory from ``plan/manifest.jsonl``
  2. for each domain, draws shards at random (without replacement) until the
     byte-proportional token estimate covers ``margin * peak_demand * overshoot``
  3. downloads *only* those objects into ``--raw-dir``, preserving domain paths

Token estimates use ``DOMAIN_AVAILABLE_TOKENS[domain] * (shard_bytes / domain_bytes)``,
the same estimator as the RegMix planner. An ``overshoot`` factor (default 1.5)
absorbs estimator noise and the fact that we stop mid-shard only at tokenize time.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mixlaw_common import (
    DEFAULT_TOKENS_PER_PARAM,
    DOMAIN_AVAILABLE_TOKENS,
    DOMAINS,
    OLMOHQ_S3,
    peak_domain_tokens,
)


def load_manifest(path: Path) -> dict[str, list[dict]]:
    assert OLMOHQ_S3.startswith("s3://")
    _, _, rest = OLMOHQ_S3.partition("s3://")
    bucket, _, prefix = rest.partition("/")
    prefix = prefix.rstrip("/")

    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        domain = row.get("domain")
        if domain not in by_domain:
            continue
        rel = row["path"]  # e.g. data/wiki/wiki-0000.json.gz
        local_rel = rel[5:] if rel.startswith("data/") else rel
        by_domain[domain].append(
            {
                "path": rel,
                "local_rel": local_rel,
                "bucket": bucket,
                "key": f"{prefix}/{rel}",
                "uri": f"s3://{bucket}/{prefix}/{rel}",
                "size": int(row["size"]),
                "domain": domain,
            }
        )
    for rows in by_domain.values():
        rows.sort(key=lambda r: r["path"])
    return by_domain


def select_shards(
    files: list[dict],
    target_tokens: float,
    pool_tokens: float,
    rng: random.Random,
) -> list[dict]:
    """Random whole-shard sample covering ``target_tokens``.

    Shards are considered in random order, but a shard that would overshoot the
    remaining need by more than 2× is skipped while any smaller-fit shard still
    exists. That keeps the draw random without pulling a multi-GiB pes2o/wiki
    shard when only ~100M tokens are required.
    """
    if not files or target_tokens <= 0:
        return []
    pool_bytes = sum(f["size"] for f in files)
    if pool_bytes <= 0 or pool_tokens <= 0:
        raise SystemExit("empty domain pool")

    def est_tokens(f: dict) -> float:
        return pool_tokens * (f["size"] / pool_bytes)

    unused = list(files)
    rng.shuffle(unused)
    chosen: list[dict] = []
    est = 0.0
    while est < target_tokens and unused:
        remaining = target_tokens - est
        limit = max(remaining * 2.0, remaining + 1.0)
        pick = next((f for f in unused if est_tokens(f) <= limit), None)
        if pick is None:
            pick = min(unused, key=lambda f: f["size"])
        unused.remove(pick)
        item = dict(pick)
        item["est_tokens"] = est_tokens(pick)
        chosen.append(item)
        est += item["est_tokens"]
    return chosen


def aws_cp(uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    cmd = ["aws", "s3", "cp", uri, str(tmp), "--only-show-errors"]
    subprocess.run(cmd, check=True)
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Local copy of olmo-mix-1124-30b/plan/manifest.jsonl",
    )
    ap.add_argument("--raw-dir", type=Path, required=True, help="Download root (…/data)")
    ap.add_argument("--out-plan", type=Path, required=True, help="Writes shard_selection.json")
    ap.add_argument("--tokens-per-param", type=float, default=DEFAULT_TOKENS_PER_PARAM)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument(
        "--overshoot",
        type=float,
        default=1.5,
        help="Extra factor on token estimate so shard granularity / estimator noise cannot undershoot",
    )
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--fetch-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="Plan only; do not download")
    args = ap.parse_args()

    by_domain = load_manifest(args.manifest)
    peaks = peak_domain_tokens(args.tokens_per_param)
    rng = random.Random(args.seed)

    selection: dict = {
        "tokens_per_param": args.tokens_per_param,
        "margin": args.margin,
        "overshoot": args.overshoot,
        "seed": args.seed,
        "source": OLMOHQ_S3,
        "domains": {},
    }
    to_fetch: list[dict] = []
    total_bytes = 0
    total_est = 0.0

    print(f"{'domain':<18}{'need':>10}{'est':>10}{'shards':>8}{'GiB':>8}")
    for domain in DOMAINS:
        files = by_domain.get(domain) or []
        if not files:
            raise SystemExit(f"no manifest rows for {domain}")
        need = peaks[domain] * args.margin * args.overshoot
        chosen = select_shards(files, need, float(DOMAIN_AVAILABLE_TOKENS[domain]), rng)
        if not chosen:
            raise SystemExit(f"{domain}: selection empty for need={need:,.0f}")
        est = sum(c["est_tokens"] for c in chosen)
        nbytes = sum(c["size"] for c in chosen)
        selection["domains"][domain] = {
            "target_tokens": need,
            "peak_tokens": peaks[domain],
            "est_tokens": est,
            "bytes": nbytes,
            "n_shards": len(chosen),
            "n_available": len(files),
            "shards": [
                {
                    "path": c["path"],
                    "local_rel": c["local_rel"],
                    "uri": c["uri"],
                    "size": c["size"],
                    "est_tokens": c["est_tokens"],
                }
                for c in chosen
            ],
        }
        to_fetch.extend(chosen)
        total_bytes += nbytes
        total_est += est
        print(
            f"{domain:<18}{need/1e6:>9.1f}M{est/1e6:>9.1f}M"
            f"{len(chosen):>8}{nbytes/2**30:>8.2f}"
        )

    selection["total_bytes"] = total_bytes
    selection["total_est_tokens"] = total_est
    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    args.out_plan.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nselected {len(to_fetch)} shards | "
        f"{total_est/1e6:.1f}M est tokens | {total_bytes/2**30:.2f} GiB | "
        f"plan -> {args.out_plan}"
    )

    if args.dry_run:
        print("dry-run: skipping download")
        return

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    # Drop a pointer so tokenize_working_pool can restrict to these shards if desired.
    (args.raw_dir / "shard_selection.json").write_text(
        args.out_plan.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print(f"fetching {len(to_fetch)} shards into {args.raw_dir} ({args.fetch_workers} workers)")
    errors: list[str] = []

    def one(item: dict) -> str:
        dest = args.raw_dir / item["local_rel"]
        aws_cp(item["uri"], dest)
        return item["path"]

    with ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        futs = {pool.submit(one, item): item for item in to_fetch}
        done = 0
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                path = fut.result()
                done += 1
                if done % 5 == 0 or done == len(to_fetch):
                    print(f"  fetched {done}/{len(to_fetch)}: {path}", flush=True)
            except Exception as exc:
                errors.append(f"{item['uri']}: {exc}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(f"{len(errors)} downloads failed")
    print(f"download complete under {args.raw_dir}")


if __name__ == "__main__":
    main()
