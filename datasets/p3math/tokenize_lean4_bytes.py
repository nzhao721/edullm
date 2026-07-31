#!/usr/bin/env python3
"""Materialize Lean4-Mathlib as packed UTF-8 byte token shards (.u32le.bin).

Layout: tokens/<source>/{train,val}-00000.u32le.bin

Each declaration becomes a Lean-like document; UTF-8 bytes are written as uint32
token ids (id == byte value). Documents are separated by blank lines.

Default: content only (no Gate A alphabet markers). A small val carve satisfies
the pretrain family's held-out split requirement.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pyarrow.parquet as pq

DOC_SEP = b"\n\n"
ALPHABET = list(range(256))
ALPHABET_PERIOD = 2048
VAL_FRACTION = 0.0015


def row_to_text(row: dict) -> str:
    decl_type = (row.get("type") or "").strip()
    name = (row.get("symbolic_name") or "").strip()
    statement = (row.get("statement") or "").strip()
    proof = (row.get("proof") or "").strip()
    docstring = (row.get("docstring") or "").strip()
    parts: list[str] = []
    if docstring:
        parts.append(f"/-- {docstring} -/")
    head = " ".join(p for p in (decl_type, name, statement) if p).strip()
    if head:
        parts.append(head)
    if proof:
        if proof.lstrip().startswith("by") or proof.lstrip().startswith(":=by"):
            parts.append(proof)
        else:
            parts.append(":= by\n" + proof)
    return "\n".join(parts).strip() + "\n"


def inject_alphabet(tokens: list[int], *, period: int = ALPHABET_PERIOD) -> list[int]:
    """Interleave full 0..255 markers (legacy Gate A workaround; contaminates training)."""
    out: list[int] = []
    out.extend(ALPHABET)
    out.extend(ALPHABET)
    for i, tok in enumerate(tokens):
        out.append(tok)
        if (i + 1) % period == 0:
            out.extend(ALPHABET)
    out.extend(ALPHABET)
    out.extend(ALPHABET)
    return out


def write_u32le_stream(path: Path, blob: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(blob)
    step = 1024 * 1024
    with path.open("wb") as fh:
        for i in range(0, n, step):
            chunk = blob[i : i + step]
            fh.write(struct.pack(f"<{len(chunk)}I", *chunk))
    return n


def write_u32le_ids(path: Path, token_ids: list[int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    step = 1024 * 1024
    with path.open("wb") as fh:
        for i in range(0, len(token_ids), step):
            chunk = token_ids[i : i + step]
            fh.write(struct.pack(f"<{len(chunk)}I", *chunk))
    return len(token_ids)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--source-name", default="mathlib")
    p.add_argument(
        "--alphabet-inject",
        action="store_true",
        help="Insert repeating 0..255 markers (legacy; contaminates training)",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help="Fraction of packed stream reserved for val-00000",
    )
    args = p.parse_args()

    table = pq.read_table(args.parquet)
    cols = table.to_pydict()
    n = len(next(iter(cols.values())))
    print(f"[lean4] rows={n:,} from {args.parquet}", flush=True)

    docs: list[bytes] = []
    empty = 0
    for i in range(n):
        row = {k: (cols[k][i] if cols[k][i] is not None else "") for k in cols}
        text = row_to_text(row)
        if not text.strip():
            empty += 1
            continue
        docs.append(text.encode("utf-8"))
    print(f"[lean4] docs={len(docs):,} empty={empty}", flush=True)

    blob = DOC_SEP.join(docs)
    if blob and not blob.endswith(b"\n"):
        blob += b"\n"
    content_tokens = len(blob)
    print(f"[lean4] content_utf8_bytes={content_tokens:,}", flush=True)

    src_dir = args.out_root / "tokens" / args.source_name
    if src_dir.exists():
        for old in src_dir.glob("*.u32le.bin"):
            old.unlink()
    src_dir.mkdir(parents=True, exist_ok=True)

    if args.alphabet_inject:
        token_ids = inject_alphabet(list(blob))
        alphabet_tokens = len(token_ids) - content_tokens
        val_n = max(1, int(round(len(token_ids) * args.val_fraction))) if args.val_fraction > 0 else 0
        if val_n >= len(token_ids):
            raise SystemExit("val-fraction leaves no train tokens")
        train_ids = token_ids[:-val_n] if val_n else token_ids
        val_ids = token_ids[-val_n:] if val_n else []
        write_u32le_ids(src_dir / "train-00000.u32le.bin", train_ids)
        if val_ids:
            write_u32le_ids(src_dir / "val-00000.u32le.bin", val_ids)
        train_tokens = len(train_ids)
        val_tokens = len(val_ids)
    else:
        alphabet_tokens = 0
        val_n = max(1, int(round(content_tokens * args.val_fraction))) if args.val_fraction > 0 else 0
        if val_n >= content_tokens:
            raise SystemExit("val-fraction leaves no train tokens")
        train_blob = blob[:-val_n] if val_n else blob
        val_blob = blob[-val_n:] if val_n else b""
        write_u32le_stream(src_dir / "train-00000.u32le.bin", train_blob)
        if val_blob:
            write_u32le_stream(src_dir / "val-00000.u32le.bin", val_blob)
        train_tokens = len(train_blob)
        val_tokens = len(val_blob)

    print(
        f"[lean4] wrote train_tokens={train_tokens:,} val_tokens={val_tokens:,} "
        f"(alphabet_markers={alphabet_tokens:,})",
        flush=True,
    )

    meta = {
        "source": args.source_name,
        "rows_in": n,
        "docs": len(docs),
        "empty_docs": empty,
        "content_tokens": content_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "alphabet_marker_tokens": alphabet_tokens,
        "doc_sep": "\\n\\n",
        "tokenization": "raw-utf8-bytes-as-uint32",
        "layout": f"tokens/{args.source_name}/{{train,val}}-00000.u32le.bin",
        "parquet": str(args.parquet),
        "val_fraction": args.val_fraction,
    }
    (args.out_root / "build_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
