#!/usr/bin/env python3
"""Write HQ reference corpus plan manifests for FarmShare."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from hq_reference_sources import (
    DEFAULT_SEED,
    HQ_BUDGETS,
    HQ_DOMAINS,
    HQ_SOURCES,
    REGMIX_5P5_BUDGETS,
    REGMIX_5P5_DCLM_MAX_FILES,
    TOKENIZER_ID,
    UNFILTERED_POOL_TOKENS,
    scratch_layout,
)

BUDGET_PROFILES: dict[str, dict[str, float]] = {
    "default": HQ_BUDGETS,
    "regmix-5p5": REGMIX_5P5_BUDGETS,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/scratch/users/nzhao2/hq-reference-v1"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument(
        "--budget-profile",
        default="default",
        choices=sorted(BUDGET_PROFILES),
        help="Token budget profile (default: 4B HQ; regmix-5p5: ~5.514B RegMix weights)",
    )
    parser.add_argument("--s3-bucket", default="edullm-dataset-olmohq")
    parser.add_argument("--s3-prefix", default="hq-reference-v1")
    parser.add_argument(
        "--domains",
        nargs="*",
        default=list(HQ_DOMAINS),
        help="Subset of domains to plan (default: all)",
    )
    args = parser.parse_args()

    layout = scratch_layout(args.scratch_root)
    for key in ("root", "raw", "work", "out", "manifests", "logs"):
        layout[key].mkdir(parents=True, exist_ok=True)

    budgets = BUDGET_PROFILES[args.budget_profile]
    domains = [d for d in args.domains if d in budgets]
    unknown = set(args.domains) - set(budgets)
    if unknown:
        raise SystemExit(f"unknown domains: {sorted(unknown)}")

    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scratch_root": str(args.scratch_root),
        "tokenizer_id": args.tokenizer,
        "seed": args.seed,
        "budget_profile": args.budget_profile,
        "s3_bucket": args.s3_bucket,
        "s3_prefix": args.s3_prefix,
        "domains": {},
    }
    for domain in domains:
        src = HQ_SOURCES[domain]
        source_overrides: dict = {}
        if args.budget_profile == "regmix-5p5" and domain == "dclm":
            source_overrides["max_files"] = REGMIX_5P5_DCLM_MAX_FILES
        plan["domains"][domain] = {
            "budget_tokens": budgets[domain],
            "unfiltered_pool_tokens": UNFILTERED_POOL_TOKENS[domain],
            "source": src,
            "paths": {
                "raw": str(layout["raw"] / domain),
                "work": str(layout["work"] / domain),
                "out": str(layout["out"] / domain),
                "stats": str(layout["manifests"] / f"{domain}-stats.json"),
            },
            "filter": src["filter"],
            "seed": args.seed + HQ_DOMAINS.index(domain),
            "source_overrides": source_overrides,
        }
        Path(plan["domains"][domain]["paths"]["raw"]).mkdir(parents=True, exist_ok=True)
        Path(plan["domains"][domain]["paths"]["work"]).mkdir(parents=True, exist_ok=True)
        Path(plan["domains"][domain]["paths"]["out"]).mkdir(parents=True, exist_ok=True)

    plan_path = layout["manifests"] / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    domain_list = layout["manifests"] / "domains.txt"
    domain_list.write_text("\n".join(domains) + "\n", encoding="utf-8")

    print(f"wrote {plan_path}", flush=True)
    total = sum(budgets[d] for d in domains)
    print(f"profile={args.budget_profile} domains={len(domains)} total_budget={total:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
