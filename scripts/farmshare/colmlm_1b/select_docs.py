#!/usr/bin/env python3
"""Pass B: filter sample-100BT to the MD5-mod document sample via DuckDB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb

from hash_sample import duckdb_predicate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-glob", required=True, help="e.g. /data/s100/*.parquet")
    ap.add_argument("--out", type=Path, required=True, help="output parquet path")
    ap.add_argument("--work-db", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--modulus", type=int, default=100)
    ap.add_argument("--residue", type=int, default=0)
    ap.add_argument("--stats-out", type=Path, default=None)
    args = ap.parse_args()

    work_db = args.work_db or args.out.with_suffix(".duckdb")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if work_db.exists():
        work_db.unlink()

    pred = duckdb_predicate("id", modulus=args.modulus, residue=args.residue)
    con = duckdb.connect(str(work_db))
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute(
        f"""
        CREATE TABLE docs AS
        SELECT
          id AS doc_id,
          text,
          token_count,
          dump,
          url
        FROM read_parquet('{args.parquet_glob}')
        WHERE {pred}
        """
    )
    n_docs, n_tokens = con.execute(
        "SELECT count(*), coalesce(sum(token_count), 0) FROM docs"
    ).fetchone()
    con.execute(
        f"COPY docs TO '{args.out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    stats = {
        "n_docs": int(n_docs),
        "n_tokens": int(n_tokens),
        "modulus": args.modulus,
        "residue": args.residue,
        "out": str(args.out),
    }
    print(json.dumps(stats), flush=True)
    if args.stats_out:
        args.stats_out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    con.close()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
