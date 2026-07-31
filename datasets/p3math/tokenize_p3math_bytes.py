#!/usr/bin/env python3
"""Materialize the full P3Math mix as packed UTF-8 byte token shards.

Sources (train):
  - open-web-math jsonl.zst
  - algebraic-stack jsonl.zst
  - arxiv-math-pure jsonl.zst
  - lean4-mathlib parquet (all but val carve)

Val (held-out):
  - ONLY 0.15% of Lean bytes (no OWM / AlgebraicStack / arXiv in val)

Layout:
  tokens/<source>/train-XXXXX.u32le.bin
  tokens/mathlib/val-00000.u32le.bin

Each UTF-8 byte becomes a uint32 token id. Documents joined with blank lines.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import zstandard as zstd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tokenize_lean4_bytes import DOC_SEP, VAL_FRACTION, row_to_text, write_u32le_stream  # noqa: E402

try:
    import numpy as np

    _HAS_NP = True
except ImportError:
    _HAS_NP = False


DEFAULT_SHARD_TOKENS = 500_000_000  # ~2 GiB per .u32le.bin


def _open_text_lines(path: Path) -> Iterator[str]:
    name = path.name.lower()
    if name.endswith(".jsonl.zst"):
        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                yield from text
    elif name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    elif name.endswith(".jsonl"):
        with path.open("rt", encoding="utf-8") as fh:
            yield from fh
    else:
        raise ValueError(f"unsupported jsonl path: {path}")


def iter_jsonl_texts(path: Path) -> Iterator[str]:
    for line in _open_text_lines(path):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        text = obj.get("text") or ""
        if isinstance(text, str) and text.strip():
            yield text


def iter_lean_texts(parquet: Path) -> Iterator[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet)
    cols = table.to_pydict()
    n = len(next(iter(cols.values())))
    for i in range(n):
        row = {k: (cols[k][i] if cols[k][i] is not None else "") for k in cols}
        text = row_to_text(row)
        if text.strip():
            yield text


class ShardWriter:
    """Stream UTF-8 bytes into rotating train-*.u32le.bin shards."""

    def __init__(self, out_dir: Path, *, shard_tokens: int, split: str = "train") -> None:
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.split = split
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.glob(f"{split}-*.u32le.bin"):
            old.unlink()
        self.shard_idx = 0
        self.tokens_in_shard = 0
        self.total_tokens = 0
        self.docs = 0
        self._fh: Any = None
        self._need_sep = False
        self._open_shard()

    def _shard_path(self) -> Path:
        return self.out_dir / f"{self.split}-{self.shard_idx:05d}.u32le.bin"

    def _open_shard(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self._shard_path()
        self._fh = path.open("wb")
        self.tokens_in_shard = 0
        print(f"[shard] open {path}", flush=True)

    def _rotate_if_needed(self) -> None:
        if self.tokens_in_shard >= self.shard_tokens:
            self.shard_idx += 1
            self._open_shard()
            self._need_sep = False

    def _write_raw_bytes(self, blob: bytes) -> None:
        if not blob:
            return
        if _HAS_NP:
            arr = np.frombuffer(memoryview(blob), dtype=np.uint8).astype(np.uint32, copy=False)
            self._fh.write(arr.astype("<u4", copy=False).tobytes())
        else:
            step = 1024 * 1024
            for i in range(0, len(blob), step):
                chunk = blob[i : i + step]
                self._fh.write(struct.pack(f"<{len(chunk)}I", *chunk))
        n = len(blob)
        self.tokens_in_shard += n
        self.total_tokens += n

    def write_doc(self, text: str) -> None:
        blob = text.encode("utf-8")
        if not blob.strip():
            return
        if not blob.endswith(b"\n"):
            blob += b"\n"
        self._rotate_if_needed()
        if self._need_sep:
            # Prefer keeping DOC_SEP contiguous inside a shard.
            if self.tokens_in_shard + len(DOC_SEP) + len(blob) > self.shard_tokens and self.tokens_in_shard > 0:
                self.shard_idx += 1
                self._open_shard()
                self._need_sep = False
            else:
                self._write_raw_bytes(DOC_SEP)
        self._write_raw_bytes(blob)
        self._need_sep = True
        self.docs += 1
        if self.docs % 50_000 == 0:
            print(
                f"[shard] {self.out_dir.name}/{self.split} docs={self.docs:,} "
                f"tokens={self.total_tokens:,}",
                flush=True,
            )

    def close(self) -> dict[str, Any]:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return {
            "docs": self.docs,
            "tokens": self.total_tokens,
            "shards": self.shard_idx + 1 if self.docs else 0,
            "split": self.split,
            "dir": str(self.out_dir),
        }


def tokenize_jsonl_source(
    *,
    name: str,
    path: Path,
    out_root: Path,
    shard_tokens: int,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[{name}] start {path}", flush=True)
    writer = ShardWriter(out_root / "tokens" / name, shard_tokens=shard_tokens, split="train")
    for text in iter_jsonl_texts(path):
        writer.write_doc(text)
    stats = writer.close()
    stats.update({"source": name, "in": str(path), "seconds": round(time.time() - t0, 1)})
    print(f"[{name}] done {json.dumps(stats)}", flush=True)
    return stats


def tokenize_lean(
    *,
    parquet: Path,
    out_root: Path,
    shard_tokens: int,
    val_fraction: float,
) -> dict[str, Any]:
    """Pack Lean: first write all docs to a temp byte blob stream via shards, then
    carve val from the *end* of the concatenated Lean train stream.

    Implementation: accumulate Lean UTF-8 into one contiguous byte file (content-only),
    then split into train shards + val-00000. Lean is only ~81MB of bytes so this fits RAM.
    """
    t0 = time.time()
    print(f"[mathlib] start {parquet}", flush=True)
    docs: list[bytes] = []
    empty = 0
    for text in iter_lean_texts(parquet):
        blob = text.encode("utf-8")
        if not blob.strip():
            empty += 1
            continue
        if not blob.endswith(b"\n"):
            blob += b"\n"
        docs.append(blob)
    content = DOC_SEP.join(docs)
    if content and not content.endswith(b"\n"):
        content += b"\n"
    n = len(content)
    val_n = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    if val_n >= n:
        raise SystemExit("lean val-fraction leaves no train tokens")
    train_blob = content[:-val_n]
    val_blob = content[-val_n:]

    out_dir = out_root / "tokens" / "mathlib"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.u32le.bin"):
        old.unlink()

    # Shard train blob.
    train_tokens = 0
    shard_idx = 0
    offset = 0
    while offset < len(train_blob):
        chunk = train_blob[offset : offset + shard_tokens]
        path = out_dir / f"train-{shard_idx:05d}.u32le.bin"
        write_u32le_stream(path, chunk)
        train_tokens += len(chunk)
        print(f"[mathlib] wrote {path} tokens={len(chunk):,}", flush=True)
        offset += len(chunk)
        shard_idx += 1

    val_path = out_dir / "val-00000.u32le.bin"
    write_u32le_stream(val_path, val_blob)
    print(f"[mathlib] wrote {val_path} tokens={len(val_blob):,}", flush=True)

    stats = {
        "source": "mathlib",
        "in": str(parquet),
        "docs": len(docs),
        "empty_docs": empty,
        "content_tokens": n,
        "train_tokens": train_tokens,
        "val_tokens": len(val_blob),
        "val_fraction": val_fraction,
        "train_shards": shard_idx,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[mathlib] done {json.dumps(stats)}", flush=True)
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch-root", type=Path, required=True, help="P3Math scratch with filtered/")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION, help="Lean-only val fraction")
    p.add_argument(
        "--sources",
        nargs="*",
        default=["open-web-math", "algebraic-stack", "arxiv-math-pure", "mathlib"],
    )
    args = p.parse_args()

    filtered = args.scratch_root / "filtered"
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "tokens").mkdir(parents=True, exist_ok=True)

    paths = {
        "open-web-math": filtered / "open-web-math" / "open-web-math.jsonl.zst",
        "algebraic-stack": filtered / "algebraic-stack" / "algebraic-stack.jsonl.zst",
        "arxiv-math-pure": filtered / "arxiv" / "arxiv-math-pure.jsonl.zst",
        "mathlib": filtered / "lean4-mathlib" / "data" / "train-00000-of-00001.parquet",
    }

    summary: dict[str, Any] = {"sources": {}, "val_policy": "lean_only", "val_fraction": args.val_fraction}
    t0 = time.time()
    for name in args.sources:
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(path)
        if name == "mathlib":
            summary["sources"][name] = tokenize_lean(
                parquet=path,
                out_root=args.out_root,
                shard_tokens=args.shard_tokens,
                val_fraction=args.val_fraction,
            )
        else:
            summary["sources"][name] = tokenize_jsonl_source(
                name=name,
                path=path,
                out_root=args.out_root,
                shard_tokens=args.shard_tokens,
            )

    train_tokens = 0
    val_tokens = 0
    for name, st in summary["sources"].items():
        if name == "mathlib":
            train_tokens += int(st["train_tokens"])
            val_tokens += int(st["val_tokens"])
        else:
            train_tokens += int(st["tokens"])
    summary["train_tokens"] = train_tokens
    summary["val_tokens"] = val_tokens
    summary["alphabet_marker_tokens"] = 0
    summary["tokenization"] = "raw-utf8-bytes-as-uint32"
    summary["seconds"] = round(time.time() - t0, 1)

    meta_path = args.out_root / "build_meta.json"
    meta_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
