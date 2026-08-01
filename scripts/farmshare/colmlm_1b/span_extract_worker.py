#!/usr/bin/env python3
"""Extract FineWeb fact spans from Co-LMLM entries.db for the MD5-mod sample.

Worker usage:
  python span_extract_worker.py --db entries.db --lo LO --hi HI --out spans_000.parquet
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow as pa
import pyarrow.parquet as pq

from hash_sample import doc_in_sample


def extract_range(
    db_path: Path,
    lo: int,
    hi: int,
    *,
    modulus: int,
    residue: int,
    batch_rows: int = 50_000,
) -> tuple[list[str], list[int], list[str], dict]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute(
        "SELECT entry_id, data FROM entries WHERE rowid >= ? AND rowid < ?",
        (lo, hi),
    )
    doc_ids: list[str] = []
    fact_idxs: list[int] = []
    spans: list[str] = []
    stats = {
        "scanned": 0,
        "fineweb": 0,
        "selected": 0,
        "json_errors": 0,
        "sample_id_mismatch": 0,
    }
    while True:
        rows = cur.fetchmany(batch_rows)
        if not rows:
            break
        for entry_id, data in rows:
            stats["scanned"] += 1
            if not entry_id.startswith("<urn:uuid:"):
                continue
            stats["fineweb"] += 1
            doc_id = entry_id.split("_", 1)[0]
            if not doc_in_sample(doc_id, modulus=modulus, residue=residue):
                continue
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            meta = d.get("metadata") or {}
            sample_id = meta.get("sample_id")
            if sample_id is not None and sample_id != doc_id:
                stats["sample_id_mismatch"] += 1
                # Prefer entry_id prefix (validated in plan); still keep the row.
            fact_idx = int(meta["fact_idx"])
            span = d["fact_span"]
            if not isinstance(span, str):
                continue
            doc_ids.append(doc_id)
            fact_idxs.append(fact_idx)
            spans.append(span)
            stats["selected"] += 1
    con.close()
    return doc_ids, fact_idxs, spans, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modulus", type=int, default=100)
    ap.add_argument("--residue", type=int, default=0)
    args = ap.parse_args()

    doc_ids, fact_idxs, spans, stats = extract_range(
        args.db, args.lo, args.hi, modulus=args.modulus, residue=args.residue
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "doc_id": pa.array(doc_ids, type=pa.string()),
            "fact_idx": pa.array(fact_idxs, type=pa.int32()),
            "span": pa.array(spans, type=pa.string()),
        }
    )
    pq.write_table(table, args.out, compression="zstd")
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(doc_ids), **stats}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
