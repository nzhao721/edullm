#!/usr/bin/env python3
"""Plan an append-only top-up of olmohq domains so plan≈measured within 10%.

Uses empirical tokens/byte from the existing tokenized_manifest (not the HF
published domain totals) so selected shards should land near the measured target.

Does NOT modify regmix-10b. Writes only a local plan under --out-dir.
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

# Planned availability (mixlaw_common / earlier inventory).
PLANNED_AVAILABLE = {
    "dclm": 28_600_000_000,
    "arxiv": 20_800_000_000,
    "starcoder": 20_300_000_000,
    "pes2o": 26_300_000_000,
    "open-web-math": 12_200_000_000,
    "algebraic-stack": 11_800_000_000,
    "wiki": 3_660_000_000,
}

# Measured dolma2 totals from S3_DATASETS.md / tokenized_manifest.
# Overridden at runtime when --tokenized-manifest is provided.


def domain_for_path(path: str) -> str | None:
    for d in (
        "dclm",
        "starcoder",
        "pes2o",
        "arxiv",
        "open-web-math",
        "algebraic-stack",
        "wiki",
    ):
        if path.startswith(f"data/{d}/"):
            return d
    return None


def is_data_file(path: str) -> bool:
    return bool(re.search(r"\.(json\.gz|jsonl\.gz|jsonl\.zstd|jsonl\.zst)$", path))


def measured_from_manifest(path: Path) -> dict[str, dict]:
    man = json.loads(path.read_text(encoding="utf-8"))
    by: dict[str, dict] = defaultdict(lambda: {"tokens": 0, "bytes": 0, "n": 0, "paths": set()})
    for s in man["shards"]:
        d = s["domain"]
        by[d]["tokens"] += int(s["tokens"])
        by[d]["bytes"] += int(s["bytes"])
        by[d]["n"] += 1
        mp = s.get("manifest_path")
        if mp:
            by[d]["paths"].add(mp)
    return by


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenized-manifest", type=Path, required=True)
    ap.add_argument("--pool-manifest", type=Path, required=True, help="Existing olmohq plan/manifest.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--domains", nargs="+", default=["starcoder", "pes2o"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance", type=float, default=0.10, help="|plan-meas|/meas target")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    measured = measured_from_manifest(args.tokenized_manifest)
    existing_paths = set()
    for line in args.pool_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        existing_paths.add(json.loads(line)["path"])

    # Target band: plan/(1+tol) ≤ measured ≤ plan/(1-tol)
    # Select enough new bytes to reach the floor, but stop at the ceiling so
    # optimistic rate estimates cannot overshoot the 10% band.
    targets_lo = {
        d: PLANNED_AVAILABLE[d] / (1.0 + args.tolerance) for d in args.domains
    }
    targets_hi = {
        d: PLANNED_AVAILABLE[d] / (1.0 - args.tolerance) for d in args.domains
    }

    api = HfApi(token=args.token)
    print("listing HF repo files...", flush=True)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for info in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
        path = getattr(info, "path", None)
        size = getattr(info, "size", None)
        if not path or size is None or not is_data_file(path):
            continue
        domain = domain_for_path(path)
        if domain not in args.domains:
            continue
        if path in existing_paths:
            continue
        by_domain[domain].append({"path": path, "size": int(size), "domain": domain})

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    topup_rows: list[dict] = []
    summary: dict = {"tolerance": args.tolerance, "domains": {}}

    for domain in args.domains:
        cur = measured[domain]["tokens"]
        need_total = targets_lo[domain]
        hi = targets_hi[domain]
        short = max(0.0, need_total - cur)
        rate = measured[domain]["tokens"] / max(1, measured[domain]["bytes"])
        files = by_domain[domain][:]
        rng.shuffle(files)
        # Prefer smaller files near the end of selection for finer control.
        files.sort(key=lambda f: f["size"])
        chosen: list[dict] = []
        est = 0.0
        for f in files:
            if cur + est >= need_total:
                break
            est_tok = rate * f["size"]
            if cur + est + est_tok > hi and cur + est >= need_total:
                break
            if cur + est + est_tok > hi and cur + est < need_total:
                # Would overshoot ceiling while still short — skip this file.
                continue
            item = dict(f)
            item["est_tokens"] = est_tok
            item["est_tokens_rate"] = rate
            item["selection"] = "topup-empirical-rate"
            chosen.append(item)
            est += est_tok
            topup_rows.append(item)

        projected = cur + est
        rel_err = abs(PLANNED_AVAILABLE[domain] - projected) / projected if projected else None
        summary["domains"][domain] = {
            "planned_available": PLANNED_AVAILABLE[domain],
            "measured_before": cur,
            "target_measured_min": need_total,
            "target_measured_max": hi,
            "shortfall_before": short,
            "tokens_per_byte": rate,
            "files_available_new": len(files),
            "files_selected": len(chosen),
            "est_tokens_added": est,
            "projected_measured": projected,
            "projected_plan_vs_meas_rel_err": rel_err,
            "within_tolerance": bool(rel_err is not None and rel_err <= args.tolerance),
        }
        print(
            f"{domain}: before={cur/1e9:.3f}B need>={need_total/1e9:.3f}B "
            f"add≈{est/1e9:.3f}B ({len(chosen)} shards) "
            f"proj={projected/1e9:.3f}B err={rel_err:.1%}" if rel_err is not None else "",
            flush=True,
        )

    (args.out_dir / "topup_manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in topup_rows) + ("\n" if topup_rows else ""),
        encoding="utf-8",
    )
    (args.out_dir / "topup_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'topup_manifest.jsonl'} ({len(topup_rows)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
