#!/usr/bin/env python3
"""Materialize + tokenize the Middle-PPL document-filtered RegMix corpus.

Reads ``keep_manifest.jsonl.gz`` (from ``filter_middle_ppl_docs.py``) and the
LM-label docs tree under ``--labels-root/docs/``, then writes:

  <out-dir>/docs/<domain>/*.jsonl.gz     filtered documents (with text)
  <out-dir>/tokenized/<domain>/<domain>.npy
  <out-dir>/tokenized/<domain>/<domain>.json
  <out-dir>/tokenized/paths.txt
  <out-dir>/corpus_manifest.json

Tokenizer matches RegMix / RefHQ: ``allenai/dolma2-tokenizer`` with EOS 100257.
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257  # dolma2 <|endoftext|>
_TOKENIZER = None


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def load_keep_ids(keep_manifest: Path, keep_ids_file: Optional[Path]) -> Set[str]:
    ids: Set[str] = set()
    if keep_ids_file is not None and keep_ids_file.is_file():
        for line in keep_ids_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(line.strip())
        return ids
    with open_maybe_gzip(keep_manifest, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("id")
            if doc_id:
                ids.add(str(doc_id))
    return ids


def iter_label_docs(docs_root: Path) -> Iterator[dict[str, Any]]:
    if not docs_root.is_dir():
        raise SystemExit(f"missing docs tree: {docs_root}")
    for path in sorted(docs_root.rglob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _worker_init(tokenizer_id: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)


def _encode_batch(texts: List[str]) -> List[List[int]]:
    assert _TOKENIZER is not None
    enc = _TOKENIZER(texts, add_special_tokens=False, padding=False, truncation=False)
    return [list(ids) for ids in enc["input_ids"]]


class TokenWriter:
    """Growable uint32 memmap writer (ids + EOS per document)."""

    def __init__(self, path: Path, eos_token_id: int) -> None:
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")
        self.eos = int(eos_token_id)
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

    def write_doc_ids(self, ids: List[int]) -> int:
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


def write_docs_shard(path: Path, docs: List[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for obj in docs:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def tokenize_domain(
    domain: str,
    docs: List[dict[str, Any]],
    tok_root: Path,
    *,
    tokenizer_id: str,
    eos_token_id: int,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    out_npy = tok_root / domain / f"{domain}.npy"
    out_meta = tok_root / domain / f"{domain}.json"
    writer = TokenWriter(out_npy, eos_token_id)
    n_docs = 0
    content_tokens = 0

    texts = []
    for obj in docs:
        text = obj.get("text")
        if not isinstance(text, str) or not text:
            continue
        texts.append(text)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(tokenizer_id,),
    ) as pool:
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        for encoded in pool.map(_encode_batch, batches):
            for ids in encoded:
                if not ids:
                    continue
                writer.write_doc_ids(ids)
                content_tokens += len(ids)
                n_docs += 1

    total = writer.finalize()
    meta = {
        "domain": domain,
        "n_docs": n_docs,
        "content_tokens": content_tokens,
        "tokens_with_eos": total,
        "tokenizer": tokenizer_id,
        "eos_token_id": eos_token_id,
        "dtype": "uint32",
        "path": str(out_npy),
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-root", type=Path, required=True)
    ap.add_argument(
        "--keep-manifest",
        type=Path,
        required=True,
        help="keep_manifest.jsonl.gz from filter_middle_ppl_docs.py",
    )
    ap.add_argument(
        "--keep-ids",
        type=Path,
        default=None,
        help="Optional keep_ids.txt (defaults to sibling of keep-manifest)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    ap.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shard-docs", type=int, default=10_000)
    args = ap.parse_args()

    keep_ids_path = args.keep_ids
    if keep_ids_path is None:
        sibling = args.keep_manifest.parent / "keep_ids.txt"
        if sibling.is_file():
            keep_ids_path = sibling
    wanted = load_keep_ids(args.keep_manifest, keep_ids_path)
    if not wanted:
        raise SystemExit("empty keep set")

    docs_root = args.labels_root / "docs"
    by_domain: Dict[str, List[dict[str, Any]]] = {}
    seen = 0
    for obj in iter_label_docs(docs_root):
        doc_id = obj.get("id")
        if doc_id is None or str(doc_id) not in wanted:
            continue
        domain = str(obj.get("domain") or "unknown")
        by_domain.setdefault(domain, []).append(obj)
        seen += 1

    missing = len(wanted) - seen
    if missing > 0:
        print(
            json.dumps(
                {
                    "event": "warn_missing_docs",
                    "n_keep_ids": len(wanted),
                    "n_found": seen,
                    "n_missing": missing,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if seen == 0:
        raise SystemExit(f"no kept docs found under {docs_root}")

    out = args.out_dir
    docs_out = out / "docs"
    tok_root = out / "tokenized"
    docs_out.mkdir(parents=True, exist_ok=True)
    tok_root.mkdir(parents=True, exist_ok=True)

    domain_metas: List[dict[str, Any]] = []
    for domain in sorted(by_domain):
        docs = by_domain[domain]
        # Stable order by id for reproducibility.
        docs.sort(key=lambda d: str(d.get("id") or ""))
        shard_idx = 0
        for i in range(0, len(docs), args.shard_docs):
            chunk = docs[i : i + args.shard_docs]
            write_docs_shard(
                docs_out / domain / f"middle-ppl-{shard_idx:05d}.jsonl.gz",
                chunk,
            )
            shard_idx += 1
        meta = tokenize_domain(
            domain,
            docs,
            tok_root,
            tokenizer_id=args.tokenizer,
            eos_token_id=int(args.eos_token_id),
            workers=int(args.workers),
            batch_size=int(args.batch_size),
        )
        domain_metas.append(meta)
        print(json.dumps({"event": "domain_tokenized", **meta}, sort_keys=True), flush=True)

    npy_paths = [m["path"] for m in domain_metas]
    (tok_root / "paths.txt").write_text("\n".join(npy_paths) + "\n", encoding="utf-8")

    manifest = {
        "arm": "middle-ppl-doc",
        "keep_manifest": str(args.keep_manifest.resolve()),
        "labels_root": str(args.labels_root.resolve()),
        "n_keep_ids": len(wanted),
        "n_docs_materialized": seen,
        "n_missing_docs": max(0, len(wanted) - seen),
        "tokenizer": args.tokenizer,
        "eos_token_id": int(args.eos_token_id),
        "domains": domain_metas,
        "total_content_tokens": sum(m["content_tokens"] for m in domain_metas),
        "total_tokens_with_eos": sum(m["tokens_with_eos"] for m in domain_metas),
        "paths_file": str((tok_root / "paths.txt").resolve()),
        "tokenized_root": str(tok_root.resolve()),
    }
    (out / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
