#!/usr/bin/env python3
"""Sample one OLMo-mix domain up to target tokens (with replacement if needed)."""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from olmo_shard_utils import (
    TOKENIZER_ID,
    build_offsets,
    count_batch,
    doc_text,
    load_offsets,
    materialize_docs,
    read_doc_at,
    worker_init,
)


def sample_domain(args: argparse.Namespace) -> dict:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
    ]
    domain = args.domain
    items = [m for m in manifest if m["domain"] == domain]
    if not items:
        raise SystemExit(f"no manifest rows for {domain}")

    target = int(summary["domains"][domain]["target_tokens"])
    run_dir = Path(args.run_dir)
    shard_paths = [run_dir / "data" / m["path"] for m in items]
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} shard(s), e.g. {missing[0]}")

    work = run_dir / "upsample" / domain
    work.mkdir(parents=True, exist_ok=True)
    docs_jsonl = work / "all_docs.jsonl"
    offsets_path = work / "line_offsets.bin"
    out_gz = run_dir / "data" / domain / f"{domain}-upsampled.json.gz"
    out_gz.parent.mkdir(parents=True, exist_ok=True)

    if args.force or not docs_jsonl.exists():
        n_docs = materialize_docs(shard_paths, docs_jsonl)
    else:
        n_docs = sum(1 for _ in docs_jsonl.open("rb"))

    if args.force or not offsets_path.exists():
        n2 = build_offsets(docs_jsonl, offsets_path)
        assert n2 == n_docs
    offsets = load_offsets(offsets_path, n_docs)
    print(f"[{domain}] docs={n_docs:,} target_tokens={target:,}", flush=True)

    order = list(range(n_docs))
    rng = random.Random(args.seed)
    rng.shuffle(order)

    tokens_kept = 0
    docs_written = 0
    docs_scanned = 0
    cycles = 0
    pos = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.tokenizer,),
    ) as pool, gzip.open(out_gz, "wt", encoding="utf-8") as out_fh:
        inflight = []

        def next_batch_idxs() -> list[int]:
            nonlocal pos, cycles
            if pos >= n_docs:
                cycles += 1
                pos = 0
                rng.shuffle(order)
                print(f"[{domain}] reshuffle cycle={cycles}", flush=True)
            batch = order[pos : pos + args.batch_size]
            pos += len(batch)
            return batch

        def pump() -> None:
            while len(inflight) < args.workers * 2 and tokens_kept < target:
                batch_idxs = next_batch_idxs()
                docs = [read_doc_at(docs_jsonl, offsets[j]) for j in batch_idxs]
                texts = [doc_text(d) for d in docs]
                fut = pool.submit(count_batch, texts)
                inflight.append((fut, batch_idxs, docs))

        pump()
        while inflight and tokens_kept < target:
            fut, _batch_idxs, docs = inflight.pop(0)
            counts = fut.result()
            for doc, ntok in zip(docs, counts):
                docs_scanned += 1
                if tokens_kept >= target:
                    break
                if (
                    tokens_kept >= int(target * 0.98)
                    and tokens_kept + ntok > int(target * 1.02)
                    and tokens_kept > 0
                ):
                    continue
                tokens_kept += int(ntok)
                out_fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                docs_written += 1
                if tokens_kept >= target:
                    break
            if tokens_kept < target:
                pump()
            if docs_scanned % 5000 == 0:
                print(
                    f"[{domain}] scanned={docs_scanned:,} written={docs_written:,} "
                    f"tokens={tokens_kept:,}/{target:,}",
                    flush=True,
                )

    result = {
        "domain": domain,
        "docs_pool": n_docs,
        "docs_written": docs_written,
        "docs_scanned": docs_scanned,
        "reshuffle_cycles": cycles,
        "tokens_after": tokens_kept,
        "target_tokens": target,
        "relative_error": (tokens_kept - target) / target if target else None,
        "output_shard": str(out_gz),
        "output_bytes": out_gz.stat().st_size,
        "source_shards": [str(p) for p in shard_paths],
        "tokenizer": args.tokenizer,
    }
    (work / "upsample_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sample_domain(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
