#!/usr/bin/env python3
"""Shared helpers for OLMo-mix shard IO and token counting."""

from __future__ import annotations

import gzip
import io
import json
import struct
from pathlib import Path
from typing import Any, Iterator

TOKENIZER_ID = "allenai/OLMo-2-0425-1B"
_TOKENIZER = None

# Published OLMo-mix-1124 domain token totals (HF card / tech report).
DOMAIN_TOKENS = {
    "dclm": 3.70e12,
    "starcoder": 83.0e9,
    "pes2o": 58.6e9,
    "arxiv": 20.8e9,
    "open-web-math": 12.2e9,
    "algebraic-stack": 11.8e9,
    "wiki": 3.66e9,
}
TOTAL_TOKENS = sum(DOMAIN_TOKENS.values())
NON_DCLM_DOMAINS = [d for d in DOMAIN_TOKENS if d != "dclm"]
# Domains whose target equals the full HF pool — use downloaded shards as-is.
PASS_THROUGH_DOMAINS = ("open-web-math", "algebraic-stack", "wiki")
# Domains capped below full pool — shard-select via byte-proportional token estimates.
CAP_SELECT_DOMAINS = ("starcoder", "pes2o", "arxiv")


def est_tokens_for_bytes(size: int, domain: str, domain_total_bytes: int) -> float:
    """Map shard bytes to tokens using published domain totals."""
    if domain_total_bytes <= 0:
        return 0.0
    return DOMAIN_TOKENS[domain] * (size / domain_total_bytes)


def greedy_random_sample(
    files: list[dict],
    target_tokens: float,
    domain_total_tokens: float,
    rng,
) -> list[dict]:
    """Sample whole shards randomly until estimated tokens >= target."""
    if not files or target_tokens <= 0:
        return []
    total_bytes = sum(f["size"] for f in files)
    if total_bytes <= 0:
        return []

    shuffled = files[:]
    rng.shuffle(shuffled)
    selected: list[dict] = []
    est = 0.0
    for f in shuffled:
        tokens = est_tokens_for_bytes(f["size"], f["domain"], total_bytes)
        item = dict(f)
        item["est_tokens"] = tokens
        selected.append(item)
        est += tokens
        if est >= target_tokens:
            break
    return selected


def domain_for_path(path: str) -> str | None:
    if path.startswith("data/dclm/"):
        return "dclm"
    if path.startswith("data/starcoder/"):
        return "starcoder"
    if path.startswith("data/pes2o/"):
        return "pes2o"
    if path.startswith("data/arxiv/"):
        return "arxiv"
    if path.startswith("data/open-web-math/"):
        return "open-web-math"
    if path.startswith("data/algebraic-stack/"):
        return "algebraic-stack"
    if path.startswith("data/wiki/"):
        return "wiki"
    return None


def worker_init(tokenizer_id: str) -> None:
    global _TOKENIZER
    import os

    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kwargs = {"use_fast": True}
    if token:
        kwargs["token"] = token
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — fall back for tokenizers ModelWrapper mismatches
        print(f"fast tokenizer failed ({exc!r}); retrying use_fast=False", flush=True)
        kwargs["use_fast"] = False
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)


def count_batch(texts: list[str]) -> list[int]:
    assert _TOKENIZER is not None
    enc = _TOKENIZER(texts, add_special_tokens=False, padding=False, truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


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
