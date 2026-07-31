#!/usr/bin/env python3
"""Slice the first N tokens from a tokenized corpus directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice a tokenized subset by token count.")
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    return parser.parse_args()


def copy_prefix(src: Path, dst: Path, num_tokens: int) -> None:
    src_mm = np.memmap(src, dtype=np.uint32, mode="r")
    if num_tokens > len(src_mm):
        raise ValueError(f"requested {num_tokens:,} tokens but {src} has {len(src_mm):,}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    with dst.open("wb") as fh:
        fh.truncate(num_tokens * np.dtype(np.uint32).itemsize)
    dst_mm = np.memmap(dst, dtype=np.uint32, mode="r+", shape=(num_tokens,))
    chunk = 50_000_000
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        dst_mm[start:end] = src_mm[start:end]
    dst_mm.flush()
    del dst_mm
    del src_mm


def count_docs(doc_ids_path: Path, num_tokens: int) -> int:
    doc_ids = np.memmap(doc_ids_path, dtype=np.uint32, mode="r", shape=(num_tokens,))
    changes = 0
    prev = int(doc_ids[0])
    chunk = 50_000_000
    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        arr = np.asarray(doc_ids[start:end], dtype=np.int64)
        if start > 0 and int(arr[0]) != prev:
            changes += 1
        changes += int(np.count_nonzero(np.diff(arr)))
        prev = int(arr[-1])
    return changes + 1


def verify_slice(dst_dir: Path, num_tokens: int) -> dict:
    tokens = np.memmap(dst_dir / "train_tokens.bin", dtype=np.uint32, mode="r", shape=(num_tokens,))
    head = np.asarray(tokens[:1_000_000], dtype=np.uint32)
    mid = np.asarray(tokens[num_tokens // 2 : num_tokens // 2 + 1_000_000], dtype=np.uint32)
    tail = np.asarray(tokens[max(num_tokens - 1_000_000, 0) :], dtype=np.uint32)
    stats = {
        "head_nonzero_frac": float(np.mean(head != 0)),
        "mid_nonzero_frac": float(np.mean(mid != 0)),
        "tail_nonzero_frac": float(np.mean(tail != 0)),
        "head_unique": int(len(np.unique(head))),
        "mid_unique": int(len(np.unique(mid))),
        "identical_head_mid": bool(np.array_equal(head, mid)),
    }
    if stats["head_nonzero_frac"] < 0.5 or stats["identical_head_mid"]:
        raise RuntimeError(f"Slice integrity check failed: {stats}")
    if (dst_dir / "train_doc_ids.bin").exists():
        stats["num_docs"] = count_docs(dst_dir / "train_doc_ids.bin", num_tokens)
    return stats


def main() -> None:
    args = parse_args()
    src_meta_path = args.src_dir / "meta.json"
    if not src_meta_path.exists():
        raise FileNotFoundError(src_meta_path)
    src_meta = json.loads(src_meta_path.read_text(encoding="utf-8"))
    src_tokens = int(src_meta["num_tokens"])
    if args.max_tokens > src_tokens:
        raise ValueError(f"max_tokens={args.max_tokens:,} exceeds source {src_tokens:,}")

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train_tokens.bin", "train_doc_ids.bin", "train_positions.bin"):
        src = args.src_dir / name
        if src.exists():
            print(f"copying {name} -> {args.dst_dir / name}", flush=True)
            copy_prefix(src, args.dst_dir / name, args.max_tokens)

    stats = verify_slice(args.dst_dir, args.max_tokens)
    print(f"slice integrity ok: {json.dumps(stats)}", flush=True)

    dst_meta = dict(src_meta)
    dst_meta["num_tokens"] = args.max_tokens
    dst_meta["source_dir"] = str(args.src_dir)
    dst_meta["slice_max_tokens"] = args.max_tokens
    if "num_docs" in stats:
        dst_meta["num_docs"] = stats["num_docs"]
    dst_meta["slice_integrity"] = stats
    (args.dst_dir / "meta.json").write_text(json.dumps(dst_meta, indent=2), encoding="utf-8")
    print(f"wrote {args.max_tokens:,} token subset to {args.dst_dir}", flush=True)


if __name__ == "__main__":
    main()
