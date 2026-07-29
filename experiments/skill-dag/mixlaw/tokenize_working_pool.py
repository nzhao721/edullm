#!/usr/bin/env python3
"""Tokenize a *working pool* from olmohq raw shards for the 24 mixing-law probes.

The olmohq pool (``s3://edullm-datasets/olmo100b/olmo-mix-1124-30b``) stores raw
``data/<domain>/*.json.gz`` documents, not uint32 memmaps. Training needs memmaps,
but tokenizing the entire ~95B-token pool would waste ~400 GiB and hours of CPU.

Instead this script tokenizes only what the slice planner needs: for each domain,
``margin * peak_tokens(domain)`` across the 24 mixtures at the chosen
``--tokens-per-param``. At the default compute-limited budget (tokens/param=5)
that is a few hundred million tokens total — a few GiB on disk.

Docs are drawn randomly from the local olmohq shard mirror (without replacement
within a domain) and encoded with ``allenai/dolma2-tokenizer``, matching the
rest of the OLMo-ladder stack.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from mixlaw_common import (
    DEFAULT_TOKENS_PER_PARAM,
    DOMAINS,
    EOS_TOKEN_ID,
    TOKENIZER_ID,
    domain_npy_name,
    peak_domain_tokens,
)

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
    if name.endswith((".jsonl.zstd", ".jsonl.zst")):
        import zstandard as zstd

        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8"), raw
    if name.endswith((".json.gz", ".jsonl.gz")):
        return gzip.open(path, "rt", encoding="utf-8"), None
    if name.endswith((".jsonl", ".json")):
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


def list_shards(data_dir: Path, domain: str) -> list[Path]:
    ddir = data_dir / domain
    if not ddir.is_dir():
        raise SystemExit(f"missing domain shard dir: {ddir}")

    # If select_and_fetch_shards wrote a plan here, only read those files.
    selection_path = data_dir / "shard_selection.json"
    if selection_path.is_file():
        plan = json.loads(selection_path.read_text(encoding="utf-8"))
        rels = [
            shard["local_rel"]
            for shard in plan.get("domains", {}).get(domain, {}).get("shards", [])
        ]
        shards = []
        for rel in rels:
            p = data_dir / rel
            if not p.is_file():
                raise SystemExit(f"selected shard missing: {p}")
            shards.append(p)
        if shards:
            return shards

    shards = sorted(
        p
        for p in ddir.rglob("*")
        if p.is_file()
        and p.name.endswith((".json.gz", ".jsonl.gz", ".jsonl", ".json", ".jsonl.zst", ".jsonl.zstd"))
        and not p.name.endswith(".done")
    )
    if not shards:
        raise SystemExit(f"no shards under {ddir}")
    return shards


class TokenWriter:
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


def tokenize_domain(
    *,
    data_dir: Path,
    out_dir: Path,
    domain: str,
    target_tokens: int,
    seed: int,
    workers: int,
    batch_size: int,
    tokenizer_id: str,
    eos_token_id: int,
) -> dict[str, Any]:
    shards = list_shards(data_dir, domain)
    rng = random.Random(seed)
    rng.shuffle(shards)

    out_npy = out_dir / domain / domain_npy_name(domain)
    out_meta = out_dir / domain / f"{domain}.json"
    if out_npy.is_file() and out_meta.is_file():
        meta = json.loads(out_meta.read_text(encoding="utf-8"))
        if int(meta.get("tokens_with_eos", 0)) >= target_tokens:
            print(f"[{domain}] reuse existing {out_npy} ({meta['tokens_with_eos']:,} tokens)")
            return meta

    writer = TokenWriter(out_npy, eos_token_id)
    tokens = 0
    docs = 0
    shard_i = 0

    print(
        f"[{domain}] target={target_tokens:,} shards={len(shards)} "
        f"tokenizer={tokenizer_id}",
        flush=True,
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(tokenizer_id,),
    ) as pool:
        while tokens < target_tokens and shard_i < len(shards):
            shard = shards[shard_i]
            shard_i += 1
            print(f"[{domain}] reading {shard.name}", flush=True)
            batch_docs: list[dict[str, Any]] = []
            inflight = []

            def flush_batch() -> None:
                nonlocal tokens, docs
                if not batch_docs:
                    return
                texts = [doc_text(d) for d in batch_docs]
                fut = pool.submit(_encode_batch, texts)
                inflight.append((fut, list(batch_docs)))
                batch_docs.clear()

            for doc in iter_docs(shard):
                if tokens >= target_tokens:
                    break
                batch_docs.append(doc)
                if len(batch_docs) >= batch_size:
                    flush_batch()
                while inflight and (len(inflight) >= workers * 2 or tokens >= target_tokens):
                    fut, src_docs = inflight.pop(0)
                    for src, ids in zip(src_docs, fut.result()):
                        del src
                        if tokens >= target_tokens:
                            break
                        tokens += writer.write_doc_ids(ids)
                        docs += 1
                        if docs % 5000 == 0:
                            print(
                                f"[{domain}] docs={docs:,} tokens={tokens:,}/{target_tokens:,}",
                                flush=True,
                            )
            flush_batch()
            while inflight and tokens < target_tokens:
                fut, src_docs = inflight.pop(0)
                for src, ids in zip(src_docs, fut.result()):
                    del src
                    if tokens >= target_tokens:
                        break
                    tokens += writer.write_doc_ids(ids)
                    docs += 1

    if tokens < target_tokens:
        raise SystemExit(
            f"[{domain}] only tokenized {tokens:,} of {target_tokens:,} "
            f"(exhausted {shard_i}/{len(shards)} shards)"
        )

    npy_tokens = writer.finalize()
    meta = {
        "domain": domain,
        "output": str(out_npy),
        "tokenizer": tokenizer_id,
        "eos_token_id": eos_token_id,
        "docs": docs,
        "tokens_with_eos": npy_tokens,
        "target_tokens": target_tokens,
        "bytes": int(out_npy.stat().st_size),
        "dtype": "uint32",
        "shards_used": shard_i,
        "shards_available": len(shards),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Local olmohq .../data mirror")
    ap.add_argument("--out-dir", type=Path, required=True, help="Writes tokenized/<domain>/")
    ap.add_argument("--tokens-per-param", type=float, default=DEFAULT_TOKENS_PER_PARAM)
    ap.add_argument(
        "--margin",
        type=float,
        default=1.15,
        help="Tokenize this multiple of peak per-domain demand (headroom for plan)",
    )
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--domains", nargs="*", default=None)
    args = ap.parse_args()

    peaks = peak_domain_tokens(args.tokens_per_param)
    domains = list(args.domains) if args.domains else list(DOMAINS)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "tokens_per_param": args.tokens_per_param,
        "margin": args.margin,
        "tokenizer": TOKENIZER_ID,
        "domains": {},
    }
    for domain in domains:
        target = int(peaks[domain] * args.margin)
        if target <= 0:
            print(f"[{domain}] peak=0, skipping")
            continue
        meta = tokenize_domain(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            domain=domain,
            target_tokens=target,
            seed=args.seed + DOMAINS.index(domain),
            workers=args.workers,
            batch_size=args.batch_size,
            tokenizer_id=TOKENIZER_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        summary["domains"][domain] = meta

    (args.out_dir / "working_pool.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'working_pool.json'}")


if __name__ == "__main__":
    main()
