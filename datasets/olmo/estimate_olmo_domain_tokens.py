#!/usr/bin/env python3
"""Print per-domain estimated token counts for a rebalanced OLMo-mix run."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from olmo_shard_utils import (
    CAP_SELECT_DOMAINS,
    DOMAIN_TOKENS,
    NON_DCLM_DOMAINS,
    PASS_THROUGH_DOMAINS,
    greedy_random_sample,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plan = args.run_dir / "plan"
    summary = json.loads((plan / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (plan / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rng = random.Random(args.seed)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    dclm_rows = []
    for row in rows:
        if row["domain"] == "dclm":
            dclm_rows.append(row)
        else:
            by_domain[row["domain"]].append(row)

    results = []
    total = 0.0

    dclm_tokens = sum(
        r.get("measured_tokens", r.get("est_tokens", 0)) for r in dclm_rows
    )
    results.append(
        {
            "domain": "dclm",
            "files_selected": len(dclm_rows),
            "files_available": len(dclm_rows),
            "est_tokens": dclm_tokens,
            "target_tokens": summary["domains"]["dclm"]["target_tokens"],
            "method": "pass-through (base 30B run)",
        }
    )
    total += dclm_tokens

    for domain in NON_DCLM_DOMAINS:
        files = sorted(by_domain[domain], key=lambda x: x["path"])
        total_bytes = sum(f["size"] for f in files)
        target = float(summary["domains"][domain]["target_tokens"])

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
            raise SystemExit(f"unknown domain policy: {domain}")

        est = sum(f.get("est_tokens", 0) for f in chosen)
        results.append(
            {
                "domain": domain,
                "files_selected": len(chosen),
                "files_available": len(files),
                "est_tokens": est,
                "target_tokens": target,
                "method": method,
            }
        )
        total += est

    out = {
        "seed": args.seed,
        "est_tokens_total": total,
        "domains": results,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
