#!/usr/bin/env python3
"""Trim one domain to its token budget AND save tokenized uint32 .npy.

Uses the canonical OLMo-2 tokenizer: allenai/dolma2-tokenizer.
Writes:
  trim/<domain>/<domain>-trimmed.json.gz   -- kept documents
  tokenized/<domain>/<domain>.npy          -- contiguous uint32 token stream
  tokenized/<domain>/<domain>.json         -- token metadata
  trim/<domain>/trim_result.json           -- accounting
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import os
import random
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Canonical OLMo-2 tokenizer (allenai OLMo-ladder / dolma2).
TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257  # dolma2 <|endoftext|>
_TOKENIZER = None


def _worker_init(tokenizer_id: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)


def _encode_batch(texts: list[str]) -> list[list[int]]:
    assert _TOKENIZER is not None
    enc = _TOKENIZER(texts, add_special_tokens=False, padding=False, truncation=False)
    return [list(ids) for ids in enc["input_ids"]]


def open_text_stream(path: Path):
    name = path.name
    if name.endswith(".jsonl.zstd") or name.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8"), raw
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8"), None
    if name.endswith(".jsonl") or name.endswith(".json"):
        return path.open("rt", encoding="utf-8"), None
    raise ValueError(f"unsupported format: {path}")


def doc_text(obj: dict[str, Any]) -> str:
    for key in ("text", "content", "code", "body"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    return "\n".join(v for v in obj.values() if isinstance(v, str))


def iter_docs(path: Path) -> Iterator[dict[str, Any]]:
    stream, raw = open_text_stream(path)
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and doc_text(obj):
                yield obj
    finally:
        stream.close()
        if raw is not None:
            raw.close()


def materialize_docs(shard_paths: list[Path], out_jsonl: Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as out:
        for shard in shard_paths:
            print(f"materialize {shard}", flush=True)
            for obj in iter_docs(shard):
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
                if n % 100000 == 0:
                    print(f"  wrote {n} docs", flush=True)
    return n


def build_offsets(jsonl: Path, offsets_path: Path) -> int:
    offsets = []
    with jsonl.open("rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            offsets.append(pos)
    offsets_path.write_bytes(struct.pack(f"{len(offsets)}Q", *offsets))
    return len(offsets)


def load_offsets(offsets_path: Path, n: int) -> list[int]:
    data = offsets_path.read_bytes()
    return list(struct.unpack(f"{n}Q", data))


def read_doc_at(jsonl: Path, offset: int) -> dict[str, Any]:
    with jsonl.open("rb") as fh:
        fh.seek(offset)
        return json.loads(fh.readline())


class TokenWriter:
    """Growable uint32 memmap writer for token ids."""

    def __init__(self, path: Path, eos_token_id: int) -> None:
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")
        self.eos = eos_token_id
        self.capacity = 1 << 20
        if self.tmp.exists():
            self.tmp.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mm = np.memmap(self.tmp, mode="w+", dtype=np.uint32, shape=(self.capacity,))
        self.n = 0

    def _grow(self, need: int) -> None:
        if need <= self.capacity:
            return
        while self.capacity < need:
            self.capacity = int(self.capacity * 1.5) + (1 << 20)
        self.mm.flush()
        old = np.array(self.mm[: self.n], dtype=np.uint32)
        del self.mm
        self.mm = np.memmap(self.tmp, mode="w+", dtype=np.uint32, shape=(self.capacity,))
        self.mm[: self.n] = old

    def write_doc_ids(self, ids: list[int]) -> int:
        """Append token ids + EOS; return tokens written (including EOS)."""
        payload = list(ids) + [self.eos]
        need = self.n + len(payload)
        self._grow(need)
        self.mm[self.n : need] = np.asarray(payload, dtype=np.uint32)
        self.n = need
        return len(payload)

    def finalize(self) -> int:
        self.mm.flush()
        del self.mm
        final = np.memmap(self.tmp, mode="r", dtype=np.uint32, shape=(self.n,))
        out = np.memmap(self.path, mode="w+", dtype=np.uint32, shape=(self.n,))
        out[:] = final
        out.flush()
        del out, final
        self.tmp.unlink(missing_ok=True)
        return self.n


def trim_and_tokenize(args: argparse.Namespace) -> dict[str, Any]:
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
    work = run_dir / "trim" / domain
    work.mkdir(parents=True, exist_ok=True)
    tok_dir = run_dir / "tokenized" / domain
    tok_dir.mkdir(parents=True, exist_ok=True)

    docs_jsonl = work / "all_docs.jsonl"
    offsets_path = work / "line_offsets.bin"
    out_gz = work / f"{domain}-trimmed.json.gz"
    out_npy = tok_dir / f"{domain}.npy"
    out_meta = tok_dir / f"{domain}.json"

    if args.force:
        for p in (docs_jsonl, offsets_path, out_gz, out_npy, out_meta):
            if p.exists():
                p.unlink()

    if not docs_jsonl.exists():
        n_docs = materialize_docs(shard_paths, docs_jsonl)
    else:
        n_docs = sum(1 for _ in docs_jsonl.open("rb"))

    if not offsets_path.exists():
        n2 = build_offsets(docs_jsonl, offsets_path)
        assert n2 == n_docs
    offsets = load_offsets(offsets_path, n_docs)
    print(
        f"[{domain}] docs={n_docs:,} target_tokens={target:,} "
        f"tokenizer={args.tokenizer} eos={args.eos_token_id}",
        flush=True,
    )

    order = list(range(n_docs))
    rng = random.Random(args.seed)
    rng.shuffle(order)

    keep_idxs: list[int] = []
    tokens_kept = 0  # content tokens only (matches prior trim accounting)
    tokens_with_eos = 0
    docs_scanned = 0
    writer = TokenWriter(out_npy, args.eos_token_id)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.tokenizer,),
    ) as pool, gzip.open(out_gz, "wt", encoding="utf-8") as out_fh:
        i = 0
        inflight = []

        def pump() -> None:
            nonlocal i
            while i < n_docs and len(inflight) < args.workers * 2 and tokens_kept < target:
                batch_idxs = order[i : i + args.batch_size]
                i += len(batch_idxs)
                docs = [read_doc_at(docs_jsonl, offsets[j]) for j in batch_idxs]
                texts = [doc_text(d) for d in docs]
                fut = pool.submit(_encode_batch, texts)
                inflight.append((fut, batch_idxs, docs))

        pump()
        while inflight and tokens_kept < target:
            fut, batch_idxs, docs = inflight.pop(0)
            id_batches = fut.result()
            for idx, doc, ids in zip(batch_idxs, docs, id_batches):
                docs_scanned += 1
                ntok = len(ids)
                if tokens_kept >= target:
                    break
                if (
                    tokens_kept >= int(target * 0.98)
                    and tokens_kept + ntok > int(target * 1.02)
                    and tokens_kept > 0
                ):
                    continue
                keep_idxs.append(idx)
                tokens_kept += ntok
                tokens_with_eos += writer.write_doc_ids(ids)
                out_fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                if tokens_kept >= target:
                    break
            if tokens_kept < target:
                pump()
            if docs_scanned % 5000 == 0:
                print(
                    f"[{domain}] scanned={docs_scanned:,} kept={len(keep_idxs):,} "
                    f"tokens={tokens_kept:,}/{target:,}",
                    flush=True,
                )

    npy_tokens = writer.finalize()
    assert npy_tokens == tokens_with_eos

    tok_meta = {
        "domain": domain,
        "output": str(out_npy),
        "tokenizer": args.tokenizer,
        "eos_token_id": args.eos_token_id,
        "docs": len(keep_idxs),
        "tokens_content": tokens_kept,
        "tokens_with_eos": tokens_with_eos,
        "bytes": int(out_npy.stat().st_size),
        "dtype": "uint32",
    }
    out_meta.write_text(json.dumps(tok_meta, indent=2) + "\n", encoding="utf-8")

    result = {
        "domain": domain,
        "docs_before": n_docs,
        "docs_after": len(keep_idxs),
        "docs_scanned": docs_scanned,
        "tokens_after": tokens_kept,
        "tokens_with_eos": tokens_with_eos,
        "target_tokens": target,
        "relative_error": (tokens_kept - target) / target if target else None,
        "output_shard": str(out_gz),
        "output_bytes": out_gz.stat().st_size,
        "tokenized_npy": str(out_npy),
        "tokenized_meta": str(out_meta),
        "source_shards": [str(p) for p in shard_paths],
        "tokenizer": args.tokenizer,
        "eos_token_id": args.eos_token_id,
    }
    (work / "trim_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    parser.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    trim_and_tokenize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
