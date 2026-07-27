#!/usr/bin/env python3
"""Tokenize HQ reference text shards (out/) to dolma2 uint32 .npy memmaps."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from olmo_shard_utils import doc_text, iter_docs
from trim_and_tokenize_regmix import TokenWriter, _encode_batch, _worker_init

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257


def _list_out_shards(out_dir: Path) -> list[Path]:
    shards = sorted(out_dir.glob("documents-*.json.gz"))
    if not shards:
        raise SystemExit(f"no documents-*.json.gz under {out_dir}")
    return shards


def tokenize_domain(
    *,
    out_dir: Path,
    tok_dir: Path,
    domain: str,
    tokenizer: str,
    eos_token_id: int,
    workers: int,
    batch_size: int,
) -> dict:
    tok_dir.mkdir(parents=True, exist_ok=True)
    out_npy = tok_dir / f"{domain}.npy"
    out_meta = tok_dir / f"{domain}.json"

    writer = TokenWriter(out_npy, eos_token_id)
    docs = 0
    content_tokens = 0

    def _iter_text_batches():
        nonlocal docs
        batch_texts: list[str] = []
        for shard in _list_out_shards(out_dir):
            print(f"read {shard}", flush=True)
            for obj in iter_docs(shard):
                text = doc_text(obj)
                if not text:
                    continue
                batch_texts.append(text)
                docs += 1
                if len(batch_texts) >= batch_size:
                    yield batch_texts
                    batch_texts = []
                if docs % 100_000 == 0:
                    print(f"progress docs={docs:,} stream_tokens={content_tokens:,}", flush=True)
        if batch_texts:
            yield batch_texts

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(tokenizer,),
    ) as pool:
        for batch_texts in _iter_text_batches():
            id_batches = pool.submit(_encode_batch, batch_texts).result()
            for ids in id_batches:
                content_tokens += len(ids)
                writer.write_doc_ids(ids)

    stream_tokens = writer.finalize()
    meta = {
        "domain": domain,
        "tokenizer": tokenizer,
        "eos_token_id": eos_token_id,
        "docs": docs,
        "content_tokens": content_tokens,
        "stream_tokens_with_eos": stream_tokens,
        "tokenized_npy": str(out_npy),
        "source_out_dir": str(out_dir),
        "input_shards": [str(p) for p in _list_out_shards(out_dir)],
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.domain not in plan["domains"]:
        raise SystemExit(f"domain {args.domain} missing from plan")
    domain_plan = plan["domains"][args.domain]
    out_dir = Path(domain_plan["paths"]["out"])
    scratch_root = Path(plan["scratch_root"])
    tok_dir = scratch_root / "tokenized" / args.domain

    tokenize_domain(
        out_dir=out_dir,
        tok_dir=tok_dir,
        domain=args.domain,
        tokenizer=args.tokenizer,
        eos_token_id=args.eos_token_id,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
