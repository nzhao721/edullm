#!/usr/bin/env python3
"""Verify HQ domain budgets and upload out/ to S3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from hq_reference_sources import BUDGET_TOLERANCE, within_budget


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
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bucket = args.bucket or plan.get("s3_bucket") or "edullm-dataset-olmohq"
    prefix = args.prefix or plan.get("s3_prefix") or "hq-reference-v1"
    manifests_dir = Path(plan["scratch_root"]) / "manifests"

    domain_reports: dict[str, dict] = {}
    failures: list[str] = []
    total_tokens = 0.0

    for domain, domain_plan in plan["domains"].items():
        stats_path = Path(domain_plan["paths"]["stats"])
        if not stats_path.is_file():
            failures.append(f"{domain}: missing stats {stats_path}")
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        budget = float(domain_plan["budget_tokens"])
        realized = float(stats.get("realized_tokens", 0))
        ok = within_budget(realized, budget, BUDGET_TOLERANCE)
        domain_reports[domain] = {
            **stats,
            "budget_tokens": budget,
            "unfiltered_pool_tokens": domain_plan.get("unfiltered_pool_tokens"),
            "within_budget": ok,
            "tolerance": BUDGET_TOLERANCE,
            "out_dir": domain_plan["paths"]["out"],
        }
        total_tokens += realized
        if not ok:
            failures.append(
                f"{domain}: realized={realized:.3e} budget={budget:.3e} "
                f"(tol={BUDGET_TOLERANCE:.0%})"
            )

    final = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_id": plan.get("tokenizer_id"),
        "seed": plan.get("seed"),
        "s3_bucket": bucket,
        "s3_prefix": prefix,
        "total_realized_tokens": total_tokens,
        "total_budget_tokens": sum(
            float(domain_plan["budget_tokens"]) for domain_plan in plan["domains"].values()
        ),
        "domains": domain_reports,
        "failures": failures,
        "accepted": not failures,
    }
    final_path = manifests_dir / "final_manifest.json"
    final_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)

    if failures:
        print(f"ACCEPTANCE FAILED: {len(failures)} domain(s)", flush=True)
        for line in failures:
            print(f"  - {line}", flush=True)
        return 1

    if args.skip_upload or args.dry_run:
        print("skipping S3 upload", flush=True)
        return 0

    for domain, domain_plan in plan["domains"].items():
        local = Path(domain_plan["paths"]["out"])
        uri = f"s3://{bucket}/{prefix}/{domain}/"
        _aws_s3_sync(local, uri)
    manifest_uri = f"s3://{bucket}/{prefix}/manifests/"
    _aws_s3_sync(manifests_dir, manifest_uri)
    print(f"upload complete -> s3://{bucket}/{prefix}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
