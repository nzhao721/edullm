#!/usr/bin/env python3
"""Tokenize one OLMo-mix shard into a contiguous uint32 .npy memmap (dolma2)."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np


def open_text_stream(path: Path):
    name = path.name.lower()
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".jsonl.zstd") or name.endswith(".jsonl.zst") or name.endswith(".zstd"):
        import zstandard as zstd

        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        import io

        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    if name.endswith(".jsonl") or name.endswith(".json"):
        return open(path, "rt", encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported shard format: {path}")


def iter_texts(path: Path):
    with open_text_stream(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if not text:
                continue
            yield text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    ap.add_argument("--eos-token-id", type=int, default=100257)
    ap.add_argument("--chunk-docs", type=int, default=64)
    ap.add_argument("--hf-token-file", type=Path, default=None)
    args = ap.parse_args()

    if args.hf_token_file and args.hf_token_file.is_file():
        os.environ.setdefault("HF_TOKEN", args.hf_token_file.read_text().strip())
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # First pass: count tokens roughly by encoding in chunks and writing growable memmap.
    capacity = 1 << 20  # 1M tokens start
    mm = np.memmap(tmp, mode="w+", dtype=np.uint32, shape=(capacity,))
    n = 0
    buf_texts: list[str] = []

    def flush():
        nonlocal mm, n, capacity, buf_texts
        if not buf_texts:
            return
        enc = tok(
            buf_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            ids = list(ids) + [args.eos_token_id]
            need = n + len(ids)
            if need > capacity:
                while capacity < need:
                    capacity = int(capacity * 1.5) + (1 << 20)
                mm.flush()
                # grow by rewrite
                old = np.memmap(tmp, mode="r", dtype=np.uint32, shape=(n,))
                data = np.array(old, dtype=np.uint32)
                del old
                mm = np.memmap(tmp, mode="w+", dtype=np.uint32, shape=(capacity,))
                mm[:n] = data
            mm[n : n + len(ids)] = np.asarray(ids, dtype=np.uint32)
            n += len(ids)
        buf_texts = []

    docs = 0
    for text in iter_texts(args.input):
        buf_texts.append(text)
        docs += 1
        if len(buf_texts) >= args.chunk_docs:
            flush()
    flush()
    mm.flush()
    del mm

    # Shrink to exact size
    final = np.memmap(tmp, mode="r", dtype=np.uint32, shape=(n,))
    out = np.memmap(args.output, mode="w+", dtype=np.uint32, shape=(n,))
    out[:] = final
    out.flush()
    del out, final
    tmp.unlink(missing_ok=True)

    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "tokenizer": args.tokenizer,
        "eos_token_id": args.eos_token_id,
        "docs": docs,
        "tokens": n,
        "bytes": int(args.output.stat().st_size),
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
