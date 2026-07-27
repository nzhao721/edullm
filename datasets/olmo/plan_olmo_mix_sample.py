#!/usr/bin/env python3
"""Plan a stratified random ~N-token sample from allenai/olmo-mix-1124.

Writes:
  - manifest.jsonl  one selected file per line
  - summary.json    domain budgets and estimated tokens
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "allenai/olmo-mix-1124"

# Published OLMo-mix-1124 domain token totals (tech report / HF card).
DOMAIN_TOKENS = {
    "dclm": 3.70e12,
    "starcoder": 83.0e9,
    "pes2o": 58.6e9,
    "arxiv": 20.8e9,
    "open-web-math": 12.2e9,
    "algebraic-stack": 11.8e9,
    "wiki": 3.66e9,
}
TOTAL_TOKENS = sum(DOMAIN_TOKENS.values())


def domain_for_path(path: str) -> str | None:
    if path.startswith("data/dclm/"):
        return "dclm"
    if path.startswith("data/starcoder/"):
        return "starcoder"
    if path.startswith("data/pes2o/"):
        return "pes2o"
    if path.startswith("data/arxiv/"):
        return "arxiv"
    if path.startswith("data/open-web-math/"):
        return "open-web-math"
    if path.startswith("data/algebraic-stack/"):
        return "algebraic-stack"
    if path.startswith("data/wiki/"):
        return "wiki"
    return None


def is_data_file(path: str) -> bool:
    return bool(re.search(r"\.(json\.gz|jsonl\.gz|jsonl\.zstd|jsonl\.zst)$", path))


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
    parser.add_argument("--target-tokens", type=float, default=30e9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--token", default=None, help="HF token if needed")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    api = HfApi(token=args.token)

    print("listing repo files...", flush=True)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for info in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
        path = getattr(info, "path", None)
        size = getattr(info, "size", None)
        if not path or size is None:
            continue
        if not is_data_file(path):
            continue
        domain = domain_for_path(path)
        if domain is None:
            continue
        by_domain[domain].append({"path": path, "size": int(size), "domain": domain})

    fraction = args.target_tokens / TOTAL_TOKENS
    selected_all: list[dict] = []
    summary_domains = {}
    for domain, files in sorted(by_domain.items()):
        target = DOMAIN_TOKENS[domain] * fraction
        chosen = greedy_random_sample(files, target, DOMAIN_TOKENS[domain], rng)
        selected_all.extend(chosen)
        summary_domains[domain] = {
            "files_available": len(files),
            "bytes_available": sum(f["size"] for f in files),
            "files_selected": len(chosen),
            "bytes_selected": sum(f["size"] for f in chosen),
            "est_tokens_selected": sum(f["est_tokens"] for f in chosen),
            "target_tokens": target,
        }
        print(
            f"{domain}: selected {len(chosen)}/{len(files)} files, "
            f"est_tokens={summary_domains[domain]['est_tokens_selected']/1e9:.3f}B",
            flush=True,
        )

    selected_all.sort(key=lambda x: x["path"])
    manifest_path = args.out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(selected_all):
            item = dict(item)
            item["index"] = i
            fh.write(json.dumps(item) + "\n")

    summary = {
        "repo_id": REPO_ID,
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "fraction": fraction,
        "files_selected": len(selected_all),
        "bytes_selected": sum(f["size"] for f in selected_all),
        "est_tokens_selected": sum(f["est_tokens"] for f in selected_all),
        "domains": summary_domains,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("files_selected", "bytes_selected", "est_tokens_selected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
