#!/usr/bin/env python3
"""Build a labeling manifest for the RegMix 10B trimmed domain shards."""

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
    trim_results_path = run_dir / "plan" / "trim_results.json"
    if not trim_results_path.exists():
        raise SystemExit(f"missing trim results: {trim_results_path}")

    out = args.out or (run_dir / "labels" / "label_manifest.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    trim_results = json.loads(trim_results_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for index, item in enumerate(trim_results):
        domain = item["domain"]
        local = Path(item["output_shard"])
        if not local.exists():
            raise SystemExit(f"missing trimmed shard for {domain}: {local}")
        rel_path = f"trim/{domain}/{local.name}"
        rows.append(
            {
                "index": index,
                "domain": domain,
                "source_kind": "trimmed",
                "path": str(local),
                "rel_path": rel_path,
                "est_tokens": item.get("tokens_after"),
                "docs": item.get("docs_after"),
                "size": local.stat().st_size,
            }
        )

    rows.sort(key=lambda row: (row["domain"], row["rel_path"]))
    for index, row in enumerate(rows):
        row["index"] = index

    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    meta = {
        "corpus": "regmix-10b",
        "n_shards": len(rows),
        "by_domain": {row["domain"]: 1 for row in rows},
        "manifest": str(out),
    }
    meta_path = out.with_suffix(".summary.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
