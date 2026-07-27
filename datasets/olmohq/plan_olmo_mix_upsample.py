#!/usr/bin/env python3
"""Plan rebalanced OLMo-mix: DCLM from base run, shard-select capped domains."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from olmo_shard_utils import (
    CAP_SELECT_DOMAINS,
    DOMAIN_TOKENS,
    NON_DCLM_DOMAINS,
    PASS_THROUGH_DOMAINS,
    domain_for_path,
    greedy_random_sample,
)

REPO_ID = "allenai/olmo-mix-1124"
DEFAULT_CAP = 20e9


def is_data_file(path: str) -> bool:
    return bool(re.search(r"\.(json\.gz|jsonl\.gz|jsonl\.zstd|jsonl\.zst)$", path))


def load_base_dclm_manifest(base_manifest: Path) -> list[dict]:
    rows = []
    for line in base_manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("domain") == "dclm":
            rows.append(item)
    if not rows:
        raise SystemExit(f"no DCLM rows in {base_manifest}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--cap-tokens", type=float, default=DEFAULT_CAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    base_manifest = args.base_run_dir / "plan" / "manifest.jsonl"
    if not base_manifest.exists():
        raise SystemExit(f"missing {base_manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    api = HfApi(token=args.token)

    domain_targets = {
        domain: min(args.cap_tokens, DOMAIN_TOKENS[domain]) for domain in NON_DCLM_DOMAINS
    }

    print("listing repo files for non-DCLM domains...", flush=True)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for info in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
        path = getattr(info, "path", None)
        size = getattr(info, "size", None)
        if not path or size is None or not is_data_file(path):
            continue
        domain = domain_for_path(path)
        if domain is None or domain == "dclm":
            continue
        by_domain[domain].append({"path": path, "size": int(size), "domain": domain})

    dclm_rows = load_base_dclm_manifest(base_manifest)
    manifest_rows = list(dclm_rows)
    summary_domains: dict[str, dict] = {
        "dclm": {
            "files_available": len(dclm_rows),
            "files_selected": len(dclm_rows),
            "bytes_selected": sum(r.get("size", 0) for r in dclm_rows),
            "est_tokens_selected": sum(
                r.get("measured_tokens", r.get("est_tokens", 0)) for r in dclm_rows
            ),
            "target_tokens": sum(
                r.get("measured_tokens", r.get("est_tokens", 0)) for r in dclm_rows
            ),
            "source": "reused-from-base-run",
            "selection_method": "pass-through",
            "base_run_dir": str(args.base_run_dir),
        }
    }

    for domain in NON_DCLM_DOMAINS:
        files = sorted(by_domain.get(domain, []), key=lambda x: x["path"])
        if not files:
            raise SystemExit(f"no HF files for domain {domain}")
        target = domain_targets[domain]
        total_bytes = sum(f["size"] for f in files)

        if domain in PASS_THROUGH_DOMAINS:
            chosen = []
            for f in files:
                item = dict(f)
                item["est_tokens"] = DOMAIN_TOKENS[domain] * (f["size"] / total_bytes)
                chosen.append(item)
            method = "pass-through-full-pool"
        elif domain in CAP_SELECT_DOMAINS:
            chosen = greedy_random_sample(files, target, DOMAIN_TOKENS[domain], rng)
            method = "greedy-shard-sample-bytes"
        else:
            raise SystemExit(f"no selection policy for {domain}")

        manifest_rows.extend(chosen)
        summary_domains[domain] = {
            "files_available": len(files),
            "bytes_available": total_bytes,
            "files_selected": len(chosen),
            "bytes_selected": sum(f["size"] for f in chosen),
            "est_tokens_selected": sum(f.get("est_tokens", 0) for f in chosen),
            "target_tokens": target,
            "full_pool_tokens": DOMAIN_TOKENS[domain],
            "cap_tokens": args.cap_tokens,
            "selection_method": method,
        }
        print(
            f"{domain}: select {len(chosen)}/{len(files)} files, "
            f"est={summary_domains[domain]['est_tokens_selected']/1e9:.3f}B "
            f"target={target/1e9:.3f}B method={method}",
            flush=True,
        )

    manifest_rows.sort(key=lambda x: x["path"])
    manifest_path = args.out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(manifest_rows):
            row = dict(item)
            row["index"] = i
            fh.write(json.dumps(row) + "\n")

    non_dclm = [r for r in manifest_rows if r.get("domain") != "dclm"]
    (args.out_dir / "manifest_non_dclm.jsonl").write_text(
        "\n".join(json.dumps(r) for r in non_dclm) + "\n", encoding="utf-8"
    )

    est_total = summary_domains["dclm"]["est_tokens_selected"] + sum(
        summary_domains[d]["est_tokens_selected"] for d in NON_DCLM_DOMAINS
    )
    summary = {
        "repo_id": REPO_ID,
        "kind": "olmo-mix-rebalanced",
        "base_run_dir": str(args.base_run_dir),
        "cap_tokens": args.cap_tokens,
        "seed": args.seed,
        "domain_targets": domain_targets,
        "files_selected": len(manifest_rows),
        "bytes_selected": sum(r.get("size", 0) for r in manifest_rows),
        "est_tokens_selected": est_total,
        "domains": summary_domains,
        "note": (
            "Byte-proportional shard selection. Pass-through domains download all shards. "
            "Capped domains download only greedily-selected shards until est_tokens >= target."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(manifest_rows), "est_tokens": est_total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
