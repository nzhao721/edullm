#!/usr/bin/env python3
"""QA report for the Co-LMLM 1B span-marked corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-stats", type=Path, required=True)
    ap.add_argument("--mark-stats", type=Path, required=True)
    ap.add_argument("--corpus-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--token-target", type=float, default=1e9)
    ap.add_argument("--token-tol", type=float, default=0.02)
    args = ap.parse_args()

    docs = json.loads(args.docs_stats.read_text(encoding="utf-8"))
    mark = json.loads(args.mark_stats.read_text(encoding="utf-8"))

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    dist = con.execute(
        f"""
        SELECT
          count(*) AS n_docs,
          sum(token_count) AS n_tokens,
          avg(token_count) AS mean_tokens,
          approx_quantile(token_count, 0.5) AS p50_tokens,
          approx_quantile(token_count, 0.95) AS p95_tokens,
          avg(n_facts) AS mean_facts,
          avg(n_facts_placed) AS mean_facts_placed,
          sum(CASE WHEN n_facts = 0 THEN 1 ELSE 0 END) AS zero_fact_docs
        FROM read_parquet('{args.corpus_glob}')
        """
    ).fetchone()
    dump_shares = con.execute(
        f"""
        SELECT dump, count(*) AS n, sum(token_count) AS tokens
        FROM read_parquet('{args.corpus_glob}')
        GROUP BY 1
        ORDER BY tokens DESC
        """
    ).fetchall()

    n_tokens = int(docs["n_tokens"])
    rel_err = abs(n_tokens - args.token_target) / args.token_target
    report = {
        "docs_stats": docs,
        "mark_stats": mark,
        "corpus_agg": {
            "n_docs": int(dist[0]),
            "n_tokens": int(dist[1] or 0),
            "mean_tokens": float(dist[2] or 0),
            "p50_tokens": float(dist[3] or 0),
            "p95_tokens": float(dist[4] or 0),
            "mean_facts": float(dist[5] or 0),
            "mean_facts_placed": float(dist[6] or 0),
            "zero_fact_docs": int(dist[7] or 0),
        },
        "dump_shares": [
            {"dump": d, "n_docs": int(n), "tokens": int(t or 0)} for d, n, t in dump_shares
        ],
        "token_target_check": {
            "target": args.token_target,
            "n_tokens": n_tokens,
            "rel_error": rel_err,
            "within_tol": rel_err <= args.token_tol,
            "tol": args.token_tol,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["token_target_check"]), flush=True)
    return 0 if report["token_target_check"]["within_tol"] else 2


if __name__ == "__main__":
    sys.exit(main())
