#!/usr/bin/env python3
"""Materialize a sorted/filtered curriculum from labeled OLMo docs.

Works from the portable labels/ tree (metrics index + docs shards), so training
can happen on or off FarmShare after copying labels/.

Examples:
  # Easy→hard by compression ratio (ascending)
  python materialize_curriculum.py --labels-root labels \\
    --metric compression_ratio --order asc --out-dir curriculum/compression_asc

  # Keep top 20% hardest by MTLD
  python materialize_curriculum.py --labels-root labels \\
    --metric mtld --order desc --fraction 0.2 --out-dir curriculum/mtld_hard20
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_metric_rows(index_path: Path, metric: str) -> list[tuple[float, str, str]]:
    rows: list[tuple[float, str, str]] = []
    with open_maybe_gzip(index_path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            value = obj.get(metric)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            rows.append((float(value), obj["id"], obj["domain"]))
    return rows


def select_ids(
    rows: list[tuple[float, str, str]],
    order: str,
    fraction: float | None,
    limit: int | None,
) -> list[str]:
    reverse = order == "desc"
    rows.sort(key=lambda item: item[0], reverse=reverse)
    if fraction is not None:
        keep = max(1, int(len(rows) * fraction))
        rows = rows[:keep]
    if limit is not None:
        rows = rows[:limit]
    return [doc_id for _, doc_id, _ in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument(
        "--metric",
        required=True,
        choices=["compression_ratio", "flesch_reading_ease", "mtld"],
    )
    parser.add_argument("--order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-docs", type=int, default=10000)
    args = parser.parse_args()

    labels_root = args.labels_root
    index_path = labels_root / "metrics_index.jsonl.gz"
    if not index_path.exists():
        raise SystemExit(f"missing metrics index: {index_path}")

    rows = load_metric_rows(index_path, args.metric)
    ordered_ids = select_ids(rows, args.order, args.fraction, args.limit)
    wanted = set(ordered_ids)
    rank = {doc_id: i for i, doc_id in enumerate(ordered_ids)}

    # Stream all labeled docs, keep matches, then emit in metric order.
    selected: dict[str, dict] = {}
    docs_root = labels_root / "docs"
    for path in sorted(docs_root.rglob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_id = obj.get("id")
                if doc_id in wanted:
                    selected[doc_id] = obj

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    written = 0
    handle = None
    try:
        for doc_id in ordered_ids:
            obj = selected.get(doc_id)
            if obj is None:
                continue
            if handle is None or written % args.shard_docs == 0:
                if handle is not None:
                    handle.close()
                out_path = args.out_dir / f"shard-{shard_idx:05d}.jsonl.gz"
                handle = gzip.open(out_path, "wt", encoding="utf-8")
                shard_idx += 1
            obj = dict(obj)
            obj["curriculum_rank"] = rank[doc_id]
            obj["curriculum_metric"] = args.metric
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    finally:
        if handle is not None:
            handle.close()

    manifest = {
        "metric": args.metric,
        "order": args.order,
        "fraction": args.fraction,
        "limit": args.limit,
        "n_docs": written,
        "n_shards": shard_idx,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "curriculum_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
