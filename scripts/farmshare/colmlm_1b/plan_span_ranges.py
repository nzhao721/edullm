#!/usr/bin/env python3
"""Plan rowid ranges for parallel span extraction over entries.db."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM entries")
    lo, hi, n = cur.fetchone()
    con.close()
    if lo is None or hi is None:
        raise SystemExit("empty entries table")
    # rowid ranges are [start, end) covering [lo, hi]
    total_span = hi - lo + 1
    workers = max(1, int(args.workers))
    chunk = int(math.ceil(total_span / workers))
    ranges = []
    start = lo
    for i in range(workers):
        end = min(start + chunk, hi + 1)
        if start >= end:
            break
        ranges.append({"shard": i, "lo": start, "hi": end})
        start = end

    payload = {
        "db": str(args.db),
        "min_rowid": lo,
        "max_rowid": hi,
        "count": n,
        "workers": len(ranges),
        "ranges": ranges,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": n, "workers": len(ranges), "out": str(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
