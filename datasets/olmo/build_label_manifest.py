#!/usr/bin/env python3
"""Build a labeling manifest for the finalized ~30B OLMo-mix sample.

Uses trimmed shards for non-DCLM domains (exact token budget) and the
selected raw DCLM shards from plan/manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: RUN_DIR/labels/label_manifest.jsonl",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    plan_manifest = run_dir / "plan" / "manifest.jsonl"
    summary_path = run_dir / "plan" / "summary.json"
    out = args.out or (run_dir / "labels" / "label_manifest.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trimmed_by_domain = {
        item["domain"]: item["output_shard"]
        for item in summary.get("trim", {}).get("results", [])
    }

    rows: list[dict] = []
    for line in plan_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        domain = item["domain"]
        if domain in trimmed_by_domain:
            # One labeling task per trimmed domain (handled once below).
            continue
        rel = item["path"]
        local = run_dir / "data" / rel
        rows.append(
            {
                "domain": domain,
                "source_kind": "raw_selected",
                "path": str(local),
                "rel_path": rel,
                "est_tokens": item.get("est_tokens"),
                "size": item.get("size"),
            }
        )

    for domain, shard in sorted(trimmed_by_domain.items()):
        path = Path(shard)
        rel = f"trim/{domain}/{path.name}"
        rows.append(
            {
                "domain": domain,
                "source_kind": "trimmed",
                "path": str(path),
                "rel_path": rel,
                "est_tokens": summary["domains"][domain].get("measured_tokens"),
                "size": path.stat().st_size if path.exists() else None,
            }
        )

    rows.sort(key=lambda r: (r["domain"], r["rel_path"]))
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row = dict(row)
            row["index"] = index
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    meta = {
        "n_shards": len(rows),
        "by_domain": {},
        "manifest": str(out),
    }
    for row in rows:
        meta["by_domain"].setdefault(row["domain"], 0)
        meta["by_domain"][row["domain"]] += 1
    meta_path = out.with_suffix(".summary.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
