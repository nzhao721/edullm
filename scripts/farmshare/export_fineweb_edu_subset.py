#!/usr/bin/env python3
"""Stream FineWeb-Edu and write raw text jsonl shards matching a token budget."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenize

from tokenize_fineweb_edu_subset import DATASET_SPECS, open_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a streaming corpus subset as raw text.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="fineweb_edu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--max-train-tokens", type=int, required=True)
    parser.add_argument("--docs-per-shard", type=int, default=50_000)
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def shard_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / "shards" / f"train-{shard_idx:05d}.jsonl.gz"


def open_shard(output_dir: Path, shard_idx: int):
    path = shard_path(output_dir, shard_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8"), path


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]

    log(
        f"export_fineweb_edu_subset: dataset={args.dataset} "
        f"max_tokens={args.max_train_tokens:,} tokenizer={args.tokenizer} "
        f"out={args.output_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream = open_stream(args.dataset)
    progress = tqdm(desc=f"export {args.dataset}", unit="tok")

    doc_id = 0
    total_tokens = 0
    shard_idx = 0
    docs_in_shard = 0
    shard_files: list[str] = []
    shard_handle, shard_path_obj = open_shard(args.output_dir, shard_idx)
    shard_files.append(str(shard_path_obj.relative_to(args.output_dir)))

    for row in stream:
        text = row.get(spec["text_column"]) or ""
        if not text.strip():
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        remaining = args.max_train_tokens - total_tokens
        if remaining <= 0:
            break

        included_tokens = min(len(token_ids), remaining)
        truncated = included_tokens < len(token_ids)
        export_text = (
            tokenizer.decode(token_ids[:included_tokens], skip_special_tokens=True)
            if truncated
            else text
        )

        if docs_in_shard >= args.docs_per_shard:
            shard_handle.close()
            shard_idx += 1
            docs_in_shard = 0
            shard_handle, shard_path_obj = open_shard(args.output_dir, shard_idx)
            shard_files.append(str(shard_path_obj.relative_to(args.output_dir)))

        record = {
            "doc_id": doc_id,
            "num_tokens": len(token_ids),
            "included_tokens": included_tokens,
            "truncated": truncated,
            "text": export_text,
        }
        shard_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_tokens += included_tokens
        progress.update(included_tokens)
        doc_id += 1
        docs_in_shard += 1

        if total_tokens >= args.max_train_tokens:
            break

    shard_handle.close()
    progress.close()

    meta = {
        "dataset": args.dataset,
        "hf_path": spec["hf_path"],
        "hf_name": spec["hf_name"],
        "tokenizer": args.tokenizer,
        "num_tokens": int(total_tokens),
        "num_docs": int(doc_id),
        "docs_per_shard": args.docs_per_shard,
        "shards": shard_files,
        "format": "jsonl.gz",
        "fields": ["doc_id", "num_tokens", "included_tokens", "truncated", "text"],
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {doc_id:,} documents ({total_tokens:,} tokens) to {args.output_dir}")


if __name__ == "__main__":
    main()
