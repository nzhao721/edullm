#!/usr/bin/env python3
"""Build final manifest and bulk-upload OLMo-mix corpus to S3.

Uses byte-proportional token estimates (published domain totals) and whole-shard
selection for capped domains. Pass-through domains and DCLM are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from olmo_shard_utils import (
    CAP_SELECT_DOMAINS,
    DOMAIN_TOKENS,
    NON_DCLM_DOMAINS,
    PASS_THROUGH_DOMAINS,
    greedy_random_sample,
)


def local_shard_path(local_root: Path, rel: str) -> Path:
    return local_root / rel


def collect_domain_shards(
    rows: list[dict], local_root: Path
) -> dict[str, list[dict]]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        domain = row["domain"]
        rel = row["path"]
        path = local_shard_path(local_root, rel)
        if not path.exists():
            raise SystemExit(f"missing shard {path}")
        item = dict(row)
        item["size"] = path.stat().st_size
        item["local_path"] = str(path)
        by_domain[domain].append(item)
    return by_domain


def select_shards(
    summary: dict,
    by_domain: dict[str, list[dict]],
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    selected: list[dict] = []
    selection_info: dict[str, dict] = {}

    for domain in NON_DCLM_DOMAINS:
        files = sorted(by_domain.get(domain, []), key=lambda x: x["path"])
        if not files:
            raise SystemExit(f"no local shards for domain {domain}")
        target = float(summary["domains"][domain]["target_tokens"])

        if domain in PASS_THROUGH_DOMAINS:
            chosen = files
            method = "pass-through-full-pool"
        elif domain in CAP_SELECT_DOMAINS:
            chosen = greedy_random_sample(
                files, target, DOMAIN_TOKENS[domain], rng
            )
            method = "greedy-shard-sample-bytes"
        else:
            raise SystemExit(f"unknown selection policy for {domain}")

        for item in chosen:
            item = dict(item)
            item.pop("local_path", None)
            selected.append(item)

        est = sum(f.get("est_tokens", 0) for f in chosen)
        selection_info[domain] = {
            "method": method,
            "files_selected": len(chosen),
            "files_available": len(files),
            "bytes_selected": sum(f["size"] for f in chosen),
            "est_tokens_selected": est,
            "target_tokens": target,
        }
        summary["domains"][domain]["files_selected"] = len(chosen)
        summary["domains"][domain]["bytes_selected"] = sum(f["size"] for f in chosen)
        summary["domains"][domain]["est_tokens_selected"] = est
        summary["domains"][domain]["selection_method"] = method

    return selected, selection_info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="edullm-datasets")
    parser.add_argument("--prefix", default="olmo100b/olmo-mix-1124-30b")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = args.run_di
    plan_dir = run / "plan"
    local_root = run / "data"
    summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
    plan_manifest = [
        json.loads(line)
        for line in (plan_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    dclm_rows = [m for m in plan_manifest if m.get("domain") == "dclm"]
    non_dclm_rows = [m for m in plan_manifest if m.get("domain") != "dclm"]

    by_domain = collect_domain_shards(non_dclm_rows, local_root)
    selected_non_dclm, selection_info = select_shards(summary, by_domain, args.seed)

    new_manifest = list(dclm_rows) + selected_non_dclm
    new_manifest.sort(key=lambda x: x["path"])
    for i, row in enumerate(new_manifest):
        row["index"] = i

    summary["files_selected"] = len(new_manifest)
    summary["bytes_selected"] = sum(m["size"] for m in new_manifest)
    summary["est_tokens_selected"] = sum(
        m.get("est_tokens", 0) for m in new_manifest if m.get("domain") != "dclm"
    ) + sum(
        m.get("measured_tokens", m.get("est_tokens", 0))
        for m in new_manifest
        if m.get("domain") == "dclm"
    )
    summary["selection"] = {
        "method": "byte-proportional-shard-greedy",
        "seed": args.seed,
        "pass_through_domains": list(PASS_THROUGH_DOMAINS),
        "cap_select_domains": list(CAP_SELECT_DOMAINS),
        "domains": selection_info,
    }
    summary["note"] = (
        "DCLM reused from base run. Pass-through domains use all downloaded shards. "
        "Capped domains use greedy whole-shard selection with est_tokens = "
        "domain_pool_tokens * (shard_bytes / domain_total_bytes)."
    )

    (plan_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in new_manifest) + "\n", encoding="utf-8"
    )
    (plan_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (plan_dir / "selection.json").write_text(
        json.dumps(selection_info, indent=2), encoding="utf-8"
    )

    readme = f"""# edullm OLMo-mix-1124 rebalanced corpus

- Source: allenai/olmo-mix-1124
- DCLM: reused unchanged from prior 30B stratified sample
- open-web-math, algebraic-stack, wiki: full domain (downloaded shards as-is)
- starcoder, pes2o, arxiv: greedy whole-shard sample to ~20B est tokens each
- Token estimates: proportional to shard bytes vs published domain totals
- Estimated total tokens: {summary['est_tokens_selected']:,.0f}
- See plan/summary.json and plan/selection.json
"""
    (run / "README.md").write_text(readme, encoding="utf-8")

    staging = run / "s3-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(run / "README.md", staging / "README.md")
    shutil.copytree(plan_dir, staging / "plan")

    dclm_src = (local_root / "data" / "dclm").resolve()
    if not dclm_src.exists():
        dclm_src = (local_root / "dclm").resolve()
    dclm_dest = staging / "data" / "dclm"
    dclm_dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(dclm_src, dclm_dest, target_is_directory=True)

    for row in selected_non_dclm:
        rel = row["path"]
        src = local_shard_path(local_root, rel)
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(src.resolve(), dest)

    s3_uri = f"s3://{args.bucket}/{args.prefix.strip('/')}/"
    sync_cmd = [
        "aws",
        "s3",
        "sync",
        str(staging),
        s3_uri,
        "--region",
        args.region,
        "--follow-symlinks",
    ]
    if args.dry_run:
        sync_cmd.append("--dryrun")
    print(" ".join(sync_cmd), flush=True)
    subprocess.run(sync_cmd, check=True)

    print(
        json.dumps(
            {
                "s3_uri": s3_uri,
                "files": len(new_manifest),
                "est_tokens": summary["est_tokens_selected"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
