#!/usr/bin/env python3
"""Materialize a tokenized CE corpus from learnability-doc filter outputs.

Reads ``kept_ids.txt`` (or ``kept_ids.jsonl.gz``) plus labeled docs under
``<labels-root>/docs/**/*.jsonl.gz``, re-tokenizes kept texts with
``allenai/dolma2-tokenizer`` (EOS 100257), and writes per-domain uint32 memmaps
compatible with the control / BLADE CE trainers::

    <out-dir>/tokenized/<domain>/<domain>.npy
    <out-dir>/tokenized/<domain>/<domain>.json
    <out-dir>/paths_train.txt
    <out-dir>/corpus_manifest.json

Training upsamples by cycling the memmaps to a 10B token budget (~2384 steps);
this script only builds the kept subset once.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100_257


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_kept_ids(filter_dir: Path) -> set[str]:
    txt = filter_dir / "kept_ids.txt"
    if txt.is_file():
        return {ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()}
    gz = filter_dir / "kept_ids.jsonl.gz"
    if not gz.is_file():
        raise SystemExit(f"missing kept ids under {filter_dir} (run filter_learnability_docs.py)")
    ids: set[str] = set()
    with gzip.open(gz, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("id")
            if doc_id:
                ids.add(str(doc_id))
    return ids


def iter_labeled_docs(docs_root: Path) -> Iterator[dict[str, Any]]:
    if not docs_root.is_dir():
        raise SystemExit(f"missing docs root: {docs_root}")
    for path in sorted(docs_root.rglob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


class DomainWriter:
    def __init__(self, out_npy: Path, *, initial_capacity: int = 1 << 20) -> None:
        self.out_npy = out_npy
        self.tmp = out_npy.with_suffix(out_npy.suffix + ".tmp")
        if self.tmp.exists():
            self.tmp.unlink()
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = int(initial_capacity)
        self.n = 0
        self.docs = 0
        self.mm = np.memmap(self.tmp, mode="w+", dtype=np.uint32, shape=(self.capacity,))

    def _grow(self, need: int) -> None:
        if need <= self.capacity:
            return
        while self.capacity < need:
            self.capacity = int(self.capacity * 1.5) + (1 << 20)
        self.mm.flush()
        old = np.memmap(self.tmp, mode="r", dtype=np.uint32, shape=(self.n,))
        data = np.array(old, dtype=np.uint32)
        del old
        self.mm = np.memmap(self.tmp, mode="w+", dtype=np.uint32, shape=(self.capacity,))
        if self.n:
            self.mm[: self.n] = data

    def append_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        need = self.n + len(ids)
        self._grow(need)
        self.mm[self.n : need] = np.asarray(ids, dtype=np.uint32)
        self.n = need

    def finalize(self) -> dict[str, Any]:
        self.mm.flush()
        del self.mm
        if self.n == 0:
            if self.tmp.exists():
                self.tmp.unlink()
            return {"docs": 0, "tokens": 0, "tokenized_npy": None}
        final = np.memmap(self.tmp, mode="r", dtype=np.uint32, shape=(self.n,))
        out = np.memmap(self.out_npy, mode="w+", dtype=np.uint32, shape=(self.n,))
        out[:] = final
        out.flush()
        del out, final
        self.tmp.unlink(missing_ok=True)
        meta = {
            "docs": self.docs,
            "tokens": self.n,
            "bytes": int(self.out_npy.stat().st_size),
            "tokenized_npy": str(self.out_npy.resolve()),
            "tokenizer": TOKENIZER_ID,
            "eos_token_id": EOS_TOKEN_ID,
        }
        self.out_npy.with_suffix(".json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-root", type=Path, required=True)
    ap.add_argument(
        "--filter-dir",
        type=Path,
        required=True,
        help="Directory with kept_ids.txt / filter_manifest.json",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    ap.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    ap.add_argument("--chunk-docs", type=int, default=64)
    ap.add_argument("--hf-token-file", type=Path, default=None)
    args = ap.parse_args()

    if args.hf_token_file and args.hf_token_file.is_file():
        token = args.hf_token_file.read_text(encoding="utf-8").strip()
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    from transformers import AutoTokenizer

    kept_ids = load_kept_ids(args.filter_dir)
    if not kept_ids:
        raise SystemExit("kept id set is empty")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    docs_root = args.labels_root / "docs"
    out_dir = args.out_dir
    tok_root = out_dir / "tokenized"
    tok_root.mkdir(parents=True, exist_ok=True)

    writers: dict[str, DomainWriter] = {}
    buffers: dict[str, list[str]] = {}
    seen = 0
    missing_text = 0

    def flush_domain(domain: str) -> None:
        texts = buffers.get(domain) or []
        if not texts:
            return
        writer = writers[domain]
        enc = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            seq = list(ids) + [int(args.eos_token_id)]
            writer.append_ids(seq)
            writer.docs += 1
        buffers[domain] = []

    for obj in iter_labeled_docs(docs_root):
        doc_id = obj.get("id")
        if doc_id not in kept_ids:
            continue
        text = obj.get("text")
        if not isinstance(text, str) or not text:
            missing_text += 1
            continue
        domain = str(obj.get("domain") or "unknown")
        if domain not in writers:
            writers[domain] = DomainWriter(tok_root / domain / f"{domain}.npy")
            buffers[domain] = []
        buffers[domain].append(text)
        seen += 1
        if len(buffers[domain]) >= int(args.chunk_docs):
            flush_domain(domain)

    for domain in list(buffers):
        flush_domain(domain)

    domain_meta: dict[str, Any] = {}
    paths: list[Path] = []
    total_tokens = 0
    total_docs = 0
    for domain, writer in sorted(writers.items()):
        meta = writer.finalize()
        domain_meta[domain] = meta
        if meta.get("tokenized_npy"):
            paths.append(Path(meta["tokenized_npy"]))
            total_tokens += int(meta["tokens"])
            total_docs += int(meta["docs"])

    if not paths:
        raise SystemExit(
            f"no tokenized output written (matched_docs={seen}, missing_text={missing_text}, "
            f"kept_ids={len(kept_ids)})"
        )

    paths_file = out_dir / "paths_train.txt"
    paths_file.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

    filter_manifest_path = args.filter_dir / "filter_manifest.json"
    filter_manifest = None
    if filter_manifest_path.is_file():
        filter_manifest = json.loads(filter_manifest_path.read_text(encoding="utf-8"))

    corpus_manifest = {
        "arm": "learnability-doc",
        "labels_root": str(args.labels_root.resolve()),
        "filter_dir": str(args.filter_dir.resolve()),
        "filter_manifest": filter_manifest,
        "tokenizer": args.tokenizer,
        "eos_token_id": int(args.eos_token_id),
        "n_kept_ids": len(kept_ids),
        "n_docs_matched": seen,
        "n_docs_missing_text": missing_text,
        "n_docs_written": total_docs,
        "total_tokens": total_tokens,
        "domains": domain_meta,
        "paths_train": str(paths_file.resolve()),
        "upsample_note": (
            "Train with --length-tokens 10000000000; InfiniteBatchStream cycles "
            "the kept memmaps to fill the 10B / ~2384-step budget."
        ),
    }
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(corpus_manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
