#!/usr/bin/env python3
"""Trim olmohq top-up inventory so |plan-meas|/meas <= 10%.

Keeps pre-topup shards plus a prefix of topup shards per domain until measured
lands in [plan/1.1, plan/0.9]. Excess topup objects remain on S3 but are dropped
from the active manifests. Never touches regmix-10b.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PLANNED = {
    "dclm": 28_600_000_000,
    "arxiv": 20_800_000_000,
    "starcoder": 20_300_000_000,
    "pes2o": 26_300_000_000,
    "open-web-math": 12_200_000_000,
    "algebraic-stack": 11_800_000_000,
    "wiki": 3_660_000_000,
}


def band(plan: int, tol: float = 0.10) -> tuple[float, float]:
    return plan / (1.0 + tol), plan / (1.0 - tol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args()
    run = args.run_di
    pre = json.loads((run / "plan/tokenized_manifest.json").read_text(encoding="utf-8"))
    # Prefer merged if present (post-upload local); else use current S3 copy already
    # overwritten — fall back to pre + topup reports.
    merged_path = run / "plan/tokenized_manifest_merged.json"
    if merged_path.is_file():
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
    else:
        merged = pre

    pre_paths = {s["path"] for s in pre["shards"]}
    topup_by_domain: dict[str, list[dict]] = {}
    for s in merged["shards"]:
        if s["path"] in pre_paths:
            continue
        topup_by_domain.setdefault(s["domain"], []).append(s)

    # Prefer smaller topup shards first so we can land inside the band.
    for d in topup_by_domain:
        topup_by_domain[d].sort(key=lambda r: (int(r.get("tokens") or 0), r["path"]))

    keep = list(pre["shards"])
    kept_topup_paths: set[str] = set()
    report: dict = {"updated_at": datetime.now(timezone.utc).isoformat(), "domains": {}}

    for domain, plan in PLANNED.items():
        lo, hi = band(plan, args.tolerance)
        cur = sum(int(s.get("tokens") or 0) for s in keep if s["domain"] == domain)
        added = 0
        added_tok = 0
        for s in topup_by_domain.get(domain, []):
            if cur >= lo:
                break
            tok = int(s.get("tokens") or 0)
            nxt = cur + tok
            if nxt <= hi:
                keep.append(s)
                kept_topup_paths.add(s["path"])
                cur = nxt
                added += 1
                added_tok += tok
                continue
            # Shard alone would exceed hi — only take it if still short of lo and
            # the overshoot still satisfies |plan-meas|/meas <= tol.
            err_if = abs(plan - nxt) / nxt if nxt else None
            if err_if is not None and err_if <= args.tolerance:
                keep.append(s)
                kept_topup_paths.add(s["path"])
                cur = nxt
                added += 1
                added_tok += tok
            # Otherwise skip this large shard and try the next (already sorted).
        err = abs(plan - cur) / cur if cur else None
        report["domains"][domain] = {
            "planned": plan,
            "measured": cur,
            "band": [lo, hi],
            "rel_err": err,
            "within_10pct": bool(err is not None and err <= args.tolerance),
            "topup_shards_kept": added,
            "topup_tokens_kept": added_tok,
            "topup_shards_available": len(topup_by_domain.get(domain, [])),
        }

    # Also trim raw manifest.jsonl to matching kept topup paths (by manifest_path).
    pre_raw = [
        json.loads(l)
        for l in (run / "plan/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    pre_raw_paths = {r["path"] for r in pre_raw}
    # Map tokenized topup path → manifest_path
    tok_to_raw = {
        s["path"]: s.get("manifest_path")
        for s in merged["shards"]
        if s["path"] not in pre_paths
    }
    kept_raw = set()
    for tp in kept_topup_paths:
        mp = tok_to_raw.get(tp)
        if mp:
            kept_raw.add(mp)

    merged_raw_path = run / "plan/manifest_merged.jsonl"
    if merged_raw_path.is_file():
        all_raw = [
            json.loads(l)
            for l in merged_raw_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    else:
        all_raw = pre_raw
    trimmed_raw = [r for r in all_raw if r["path"] in pre_raw_paths or r["path"] in kept_raw]

    total = sum(int(s.get("tokens") or 0) for s in keep)
    out_tok = dict(pre)
    out_tok["shards"] = keep
    out_tok["total_content_tokens"] = total
    out_tok["topup_trimmed_at"] = report["updated_at"]
    out_tok["topup_note"] = (
        "Active inventory trimmed so |plan-meas|/meas<=10%; excess topup objects "
        "remain on S3 but are excluded from manifests."
    )

    by_domain = {}
    for s in keep:
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + int(s.get("tokens") or 0)
    availability = {
        "updated_at": report["updated_at"],
        "measured_tokens_by_domain": by_domain,
        "planned_available": PLANNED,
        "rel_err": {d: report["domains"][d]["rel_err"] for d in PLANNED},
        "within_10pct": {d: report["domains"][d]["within_10pct"] for d in PLANNED},
        "trim_report": report,
    }

    (run / "plan/tokenized_manifest_trimmed.json").write_text(
        json.dumps(out_tok, indent=2) + "\n", encoding="utf-8"
    )
    (run / "plan/manifest_trimmed.jsonl").write_text(
        "\n".join(json.dumps(r) for r in trimmed_raw) + "\n", encoding="utf-8"
    )
    (run / "plan/availability_after_topup.json").write_text(
        json.dumps(availability, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(availability, indent=2))
    if not all(availability["within_10pct"].values()):
        bad = [d for d, ok in availability["within_10pct"].items() if not ok]
        raise SystemExit(f"still outside 10% after trim: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
