#!/usr/bin/env python3
"""Join selected docs with spans and insert inline <FACT>...</FACT> markers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def mark(text: str, spans: list[str]) -> tuple[str, int]:
    out: list[str] = []
    cur = 0
    placed = 0
    for s in spans:
        i = text.find(s, cur)
        if i < 0:
            continue
        out.append(text[cur:i])
        out.append("<FACT>")
        out.append(s)
        out.append("</FACT>")
        cur = i + len(s)
        placed += 1
    out.append(text[cur:])
    return "".join(out), placed


def _worker(batch: list[tuple]) -> list[tuple]:
    results = []
    for doc_id, text, token_count, dump, url, spans in batch:
        marked, placed = mark(text, spans)
        results.append(
            (
                doc_id,
                marked,
                int(token_count),
                int(len(spans)),
                int(placed),
                dump or "",
                url or "",
            )
        )
    return results


def _iter_docs(docs_path: Path, columns: list[str]):
    pf = pq.ParquetFile(docs_path)
    for batch in pf.iter_batches(batch_size=2048, columns=columns):
        cols = {name: batch.column(name).to_pylist() for name in columns}
        n = len(cols[columns[0]])
        for i in range(n):
            yield tuple(cols[c][i] for c in columns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--spans-glob", required=True, help="e.g. /data/spans/spans_*.parquet")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--stats-out", type=Path, default=None)
    args = ap.parse_args()

    # Load all spans into memory grouped by doc_id (ordered by fact_idx).
    # ~19.4M rows of short strings — typically tens of GB; FarmShare mem sized accordingly.
    spans_by_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    import glob

    span_files = sorted(glob.glob(args.spans_glob))
    if not span_files:
        raise SystemExit(f"no span files matched {args.spans_glob}")
    for sp in span_files:
        table = pq.read_table(sp, columns=["doc_id", "fact_idx", "span"])
        for doc_id, fact_idx, span in zip(
            table["doc_id"].to_pylist(),
            table["fact_idx"].to_pylist(),
            table["span"].to_pylist(),
        ):
            spans_by_doc[doc_id].append((int(fact_idx), span))
    for doc_id in spans_by_doc:
        spans_by_doc[doc_id].sort(key=lambda x: x[0])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Partition writers by dump.
    writers: dict[str, pq.ParquetWriter] = {}
    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("text_marked", pa.string()),
            ("token_count", pa.int64()),
            ("n_facts", pa.int32()),
            ("n_facts_placed", pa.int32()),
            ("dump", pa.string()),
            ("url", pa.string()),
        ]
    )

    totals = {
        "n_docs": 0,
        "n_tokens": 0,
        "n_facts": 0,
        "n_facts_placed": 0,
        "n_zero_fact_docs": 0,
    }

    def flush_rows(rows: list[tuple]) -> None:
        if not rows:
            return
        by_dump: dict[str, list[tuple]] = defaultdict(list)
        for r in rows:
            by_dump[r[5] or "unknown"].append(r)
        for dump, dump_rows in by_dump.items():
            part_dir = args.out_dir / f"dump={dump}"
            part_dir.mkdir(parents=True, exist_ok=True)
            out_path = part_dir / "part-0.parquet"
            key = dump
            if key not in writers:
                writers[key] = pq.ParquetWriter(out_path, schema, compression="zstd")
            table = pa.table(
                {
                    "doc_id": [r[0] for r in dump_rows],
                    "text_marked": [r[1] for r in dump_rows],
                    "token_count": [r[2] for r in dump_rows],
                    "n_facts": [r[3] for r in dump_rows],
                    "n_facts_placed": [r[4] for r in dump_rows],
                    "dump": [r[5] for r in dump_rows],
                    "url": [r[6] for r in dump_rows],
                },
                schema=schema,
            )
            writers[key].write_table(table)

    batch: list[tuple] = []
    pending = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for doc_id, text, token_count, dump, url in _iter_docs(
            args.docs, ["doc_id", "text", "token_count", "dump", "url"]
        ):
            ordered = [s for _, s in spans_by_doc.get(doc_id, [])]
            batch.append((doc_id, text, token_count, dump, url, ordered))
            if len(batch) >= args.batch_size:
                pending.append(pool.submit(_worker, batch))
                batch = []
            if len(pending) >= args.workers * 2:
                for fut in as_completed(pending):
                    rows = fut.result()
                    for r in rows:
                        totals["n_docs"] += 1
                        totals["n_tokens"] += r[2]
                        totals["n_facts"] += r[3]
                        totals["n_facts_placed"] += r[4]
                        if r[3] == 0:
                            totals["n_zero_fact_docs"] += 1
                    flush_rows(rows)
                pending = []
        if batch:
            pending.append(pool.submit(_worker, batch))
        for fut in as_completed(pending):
            rows = fut.result()
            for r in rows:
                totals["n_docs"] += 1
                totals["n_tokens"] += r[2]
                totals["n_facts"] += r[3]
                totals["n_facts_placed"] += r[4]
                if r[3] == 0:
                    totals["n_zero_fact_docs"] += 1
            flush_rows(rows)

    for w in writers.values():
        w.close()

    if totals["n_facts"] > 0:
        totals["placement_rate"] = totals["n_facts_placed"] / totals["n_facts"]
    else:
        totals["placement_rate"] = None
    print(json.dumps(totals), flush=True)
    if args.stats_out:
        args.stats_out.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
