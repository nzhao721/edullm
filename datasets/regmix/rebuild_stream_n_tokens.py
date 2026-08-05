#!/usr/bin/env python3
"""Rebuild stream-faithful n_tokens from trim text using dolma2 packing rules.

LM labeling re-tokenized documents and drifted ~6.6% vs the packed
``tokenized/<domain>/<domain>.npy`` streams. Curriculum parent-pool mapping
requires ``sum(n_tokens + 1 EOS) == source_total_tokens`` exactly.

This script re-encodes ``trim/<domain>/*-trimmed.json.gz`` with the same
conventions as ``datasets/trim_and_tokenize_regmix.py``:
  - allenai/dolma2-tokenizer
  - add_special_tokens=False
  - one EOS (100257) appended per document in the packed stream

Outputs a repaired LM metrics index where ``n_tokens`` is stream-faithful.
Learnability scores and ids are preserved; only ``n_tokens`` is replaced.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger("rebuild_stream_n_tokens")

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def doc_text(obj: dict) -> str:
    for key in ("text", "content", "code", "body"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    return "\n".join(v for v in obj.values() if isinstance(v, str))


def iter_trim_texts(path: Path) -> Iterator[str]:
    with open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            text = doc_text(obj)
            if text:
                yield text


def encode_lengths(texts: List[str], tokenizer) -> List[int]:
    enc = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def rebuild_domain(
    *,
    domain: str,
    trim_path: Path,
    meta_path: Path,
    tokenizer,
    batch_size: int,
) -> List[int]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_docs = int(meta["docs"])
    expected_with_eos = int(meta["tokens_with_eos"])
    expected_content = int(meta["tokens_content"])

    lengths: List[int] = []
    batch: List[str] = []
    for text in iter_trim_texts(trim_path):
        batch.append(text)
        if len(batch) >= batch_size:
            lengths.extend(encode_lengths(batch, tokenizer))
            batch.clear()
            if len(lengths) % 50000 < batch_size:
                log.info("%s: encoded %d docs", domain, len(lengths))
    if batch:
        lengths.extend(encode_lengths(batch, tokenizer))

    if len(lengths) != expected_docs:
        raise SystemExit(
            f"{domain}: trim docs {len(lengths)} != meta docs {expected_docs}"
        )
    content = sum(lengths)
    with_eos = content + len(lengths)
    if content != expected_content or with_eos != expected_with_eos:
        raise SystemExit(
            f"{domain}: retokenize mismatch content={content} (expected {expected_content}), "
            f"with_eos={with_eos} (expected {expected_with_eos})"
        )
    log.info(
        "%s: ok docs=%d content=%d with_eos=%d",
        domain,
        len(lengths),
        content,
        with_eos,
    )
    return lengths


def load_lm_by_domain(path: Path) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = {}
    with open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            domain = obj.get("domain")
            if not domain:
                raise SystemExit(f"lm row missing domain: {obj.get('id')}")
            by.setdefault(str(domain), []).append(obj)
    for domain, rows in by.items():
        rows.sort(key=lambda r: int(r["source_doc"]))
        ordinals = [int(r["source_doc"]) for r in rows]
        if ordinals != list(range(len(rows))):
            raise SystemExit(
                f"{domain}: lm source_doc not contiguous 0..{len(rows)-1}"
            )
    return by


def write_repaired_index(
    *,
    lm_by_domain: Dict[str, List[dict]],
    lengths_by_domain: Dict[str, List[int]],
    out_path: Path,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_changed = 0
    old_sum = 0
    new_sum = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as handle:
        for domain in sorted(lm_by_domain):
            rows = lm_by_domain[domain]
            lengths = lengths_by_domain[domain]
            if len(rows) != len(lengths):
                raise SystemExit(
                    f"{domain}: lm rows {len(rows)} != stream lengths {len(lengths)}"
                )
            for row, n_tokens in zip(rows, lengths):
                old = int(row.get("n_tokens") or 0)
                old_sum += old
                new_sum += int(n_tokens)
                if old != int(n_tokens):
                    n_changed += 1
                out = dict(row)
                out["n_tokens"] = int(n_tokens)
                out["n_tokens_source"] = "stream_faithful_dolma2_retokenize"
                out["n_tokens_lm_labeling"] = old
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_rows += 1
    return {
        "n_rows": n_rows,
        "n_changed": n_changed,
        "old_n_tokens_sum": old_sum,
        "new_n_tokens_sum": new_sum,
        "out": str(out_path),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regmix-root", type=Path, required=True)
    ap.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Subset of domains (default: all under tokenized/)",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--out-index",
        type=Path,
        default=None,
        help="Repaired metrics_index.jsonl.gz path (required unless --lengths-only)",
    )
    ap.add_argument(
        "--lengths-dir",
        type=Path,
        required=True,
        help="Directory for per-domain length arrays (json)",
    )
    ap.add_argument(
        "--skip-retokenize",
        action="store_true",
        help="Reuse lengths-dir JSON files; only rewrite the metrics index",
    )
    ap.add_argument(
        "--lengths-only",
        action="store_true",
        help="Only write per-domain length files; do not rewrite the metrics index",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.lengths_only and args.skip_retokenize:
        raise SystemExit("nothing to do: --lengths-only with --skip-retokenize")
    if not args.lengths_only and args.out_index is None:
        raise SystemExit("--out-index is required unless --lengths-only")

    root = args.regmix_root
    tokenized = root / "tokenized"
    trim_root = root / "trim"
    lm_index = root / "lm_labels" / "labels" / "metrics_index.jsonl.gz"
    if not lm_index.is_file():
        raise SystemExit(f"missing {lm_index}")

    domains = args.domains or sorted(p.name for p in tokenized.iterdir() if p.is_dir())
    args.lengths_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    lengths_by_domain: Dict[str, List[int]] = {}
    if not args.skip_retokenize:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True)
        if int(tokenizer.eos_token_id) != EOS_TOKEN_ID:
            raise SystemExit(
                f"unexpected eos_token_id {tokenizer.eos_token_id}, expected {EOS_TOKEN_ID}"
            )

    for domain in domains:
        lengths_path = args.lengths_dir / f"{domain}.n_tokens.json"
        if args.skip_retokenize:
            lengths = json.loads(lengths_path.read_text(encoding="utf-8"))
            if not isinstance(lengths, list):
                raise SystemExit(f"{lengths_path}: expected list")
            lengths_by_domain[domain] = [int(x) for x in lengths]
            continue
        assert tokenizer is not None
        trim_path = trim_root / domain / f"{domain}-trimmed.json.gz"
        meta_path = tokenized / domain / f"{domain}.json"
        if not trim_path.is_file():
            raise SystemExit(f"missing trim: {trim_path}")
        if not meta_path.is_file():
            raise SystemExit(f"missing meta: {meta_path}")
        lengths = rebuild_domain(
            domain=domain,
            trim_path=trim_path,
            meta_path=meta_path,
            tokenizer=tokenizer,
            batch_size=int(args.batch_size),
        )
        lengths_path.write_text(json.dumps(lengths) + "\n", encoding="utf-8")
        lengths_by_domain[domain] = lengths
        log.info("wrote %s", lengths_path)

    if args.lengths_only:
        print(
            json.dumps(
                {
                    "lengths_only": True,
                    "domains": {
                        d: len(lengths_by_domain[d]) for d in sorted(lengths_by_domain)
                    },
                    "lengths_dir": str(args.lengths_dir),
                },
                indent=2,
            )
        )
        return 0

    log.info("loading LM metrics index %s", lm_index)
    lm_by_domain = load_lm_by_domain(lm_index)
    all_domains = sorted(lm_by_domain)
    missing = [d for d in all_domains if d not in lengths_by_domain]
    if missing:
        # Load any remaining length files so a merge can cover the full corpus.
        for domain in missing:
            lengths_path = args.lengths_dir / f"{domain}.n_tokens.json"
            if not lengths_path.is_file():
                raise SystemExit(
                    f"missing stream lengths for {domain}: {lengths_path} "
                    "(run per-domain rebuilds first)"
                )
            lengths = json.loads(lengths_path.read_text(encoding="utf-8"))
            lengths_by_domain[domain] = [int(x) for x in lengths]

    assert args.out_index is not None
    stats = write_repaired_index(
        lm_by_domain=lm_by_domain,
        lengths_by_domain=lengths_by_domain,
        out_path=args.out_index,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
