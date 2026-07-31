#!/usr/bin/env python3
"""Stream FineWeb-Edu (or SlimPajama), tokenize, and write uint32 memmaps.

Writes into pre-allocated files of exact ``max_train_tokens`` length to avoid
numpy memmap grow/truncate corruption (opening the same path with mode=w+ while
another memmap is live can zero the prefix).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenize

DATASET_SPECS = {
    # Globally shuffled FineWeb-Edu ~100BT dump (stream until max_train_tokens).
    "fineweb_edu": {
        "hf_path": "HuggingFaceFW/fineweb_edu_100BT-shuffled",
        "hf_name": None,
        "text_column": "text",
    },
    # Original unshuffled FineWeb-Edu sample (kept for reference / A-B).
    "fineweb_edu_unshuffled": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-100BT",
        "text_column": "text",
    },
    "slimpajama": {
        "hf_path": "gmongaras/SlimPajama-627B_Reupload",
        "hf_name": None,
        "text_column": "text",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a streaming corpus subset.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="fineweb_edu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-train-tokens", type=int, required=True)
    parser.add_argument("--val-tokens-target", type=int, default=320_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def open_stream(dataset_name: str):
    spec = DATASET_SPECS[dataset_name]
    kwargs = {"path": spec["hf_path"], "split": "train", "streaming": True}
    if spec["hf_name"] is not None:
        kwargs["name"] = spec["hf_name"]
    log(f"Opening HuggingFace stream: {kwargs}")
    return load_dataset(**kwargs)


def write_val_manifest(
    val_dir: Path,
    *,
    seq_len: int,
    num_sequences: int,
    num_loss_tokens: int,
) -> None:
    manifest = {
        "seq_len": seq_len,
        "num_sequences": num_sequences,
        "num_loss_tokens": num_loss_tokens,
    }
    (val_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def open_preallocated(path: Path, num_tokens: int) -> np.memmap:
    """Create a fresh uint32 memmap of exact length (never grow in-place)."""
    if path.exists():
        path.unlink()
    # Create sparse/zero-filled file of exact byte size, then map it.
    with path.open("wb") as fh:
        fh.truncate(num_tokens * np.dtype(np.uint32).itemsize)
    return np.memmap(path, dtype=np.uint32, mode="r+", shape=(num_tokens,))


def truncate_bin(path: Path, num_tokens: int) -> None:
    with path.open("r+b") as fh:
        fh.truncate(num_tokens * np.dtype(np.uint32).itemsize)


def integrity_check(tokens_path: Path, doc_ids_path: Path, num_tokens: int) -> dict:
    """Fail loudly if the written corpus looks like the old all-zero corruption."""
    tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r", shape=(num_tokens,))
    doc_ids = np.memmap(doc_ids_path, dtype=np.uint32, mode="r", shape=(num_tokens,))

    sample_starts = [0]
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        sample_starts.append(int(frac * num_tokens))
    sample_starts = sorted({min(s, max(num_tokens - 10_000, 0)) for s in sample_starts})

    nonzero_token_windows = 0
    unique_doc_ids: set[int] = set()
    for start in sample_starts:
        end = min(start + 10_000, num_tokens)
        tok_w = np.asarray(tokens[start:end], dtype=np.uint32)
        doc_w = np.asarray(doc_ids[start:end], dtype=np.uint32)
        if int(np.count_nonzero(tok_w)) > 0:
            nonzero_token_windows += 1
        unique_doc_ids.update(int(x) for x in np.unique(doc_w))

    # Broader stats on first/middle/last 1M
    def window_stats(start: int, size: int = 1_000_000) -> dict:
        end = min(start + size, num_tokens)
        tok_w = np.asarray(tokens[start:end], dtype=np.uint32)
        doc_w = np.asarray(doc_ids[start:end], dtype=np.uint32)
        return {
            "start": start,
            "len": end - start,
            "token_nonzero_frac": float(np.mean(tok_w != 0)),
            "token_unique": int(len(np.unique(tok_w))),
            "doc_id_min": int(doc_w.min()) if end > start else -1,
            "doc_id_max": int(doc_w.max()) if end > start else -1,
        }

    stats = {
        "num_tokens": num_tokens,
        "nonzero_token_windows": nonzero_token_windows,
        "sampled_windows": len(sample_starts),
        "unique_doc_ids_in_samples": len(unique_doc_ids),
        "head": window_stats(0),
        "mid": window_stats(num_tokens // 2),
        "tail": window_stats(max(num_tokens - 1_000_000, 0)),
    }

    if nonzero_token_windows < max(3, len(sample_starts) // 2):
        raise RuntimeError(
            "Integrity check failed: too many all-zero token windows. "
            f"stats={json.dumps(stats)}"
        )
    if stats["head"]["token_nonzero_frac"] < 0.5:
        raise RuntimeError(
            "Integrity check failed: head of train_tokens.bin is mostly zeros. "
            f"stats={json.dumps(stats)}"
        )
    if stats["unique_doc_ids_in_samples"] < 10:
        raise RuntimeError(
            "Integrity check failed: too few distinct doc_ids in samples. "
            f"stats={json.dumps(stats)}"
        )
    return stats


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    spec = DATASET_SPECS[args.dataset]

    log(
        f"tokenize_fineweb_edu_subset: dataset={args.dataset} "
        f"max_tokens={args.max_train_tokens:,} tokenizer={args.tokenizer} "
        f"out={args.output_dir}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens_path = args.output_dir / "train_tokens.bin"
    train_doc_ids_path = args.output_dir / "train_doc_ids.bin"
    train_positions_path = args.output_dir / "train_positions.bin"

    # Pre-allocate exact capacity so we never resize live memmaps.
    log(f"Pre-allocating {args.max_train_tokens:,} uint32 slots (~{args.max_train_tokens * 4 / 1e9:.2f} GiB each)")
    tokens_mm = open_preallocated(train_tokens_path, args.max_train_tokens)
    doc_ids_mm = open_preallocated(train_doc_ids_path, args.max_train_tokens)
    positions_mm = open_preallocated(train_positions_path, args.max_train_tokens)

    val_doc_reservoir: list[dict] = []
    val_reservoir_size = 512

    doc_id = 0
    total_tokens = 0

    stream = open_stream(args.dataset)
    progress = tqdm(desc=f"tokenize {args.dataset}", unit="tok", total=args.max_train_tokens)

    for row in stream:
        text = row.get(spec["text_column"]) or ""
        if not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 2:
            continue

        doc_record = {"doc_id": doc_id, "token_ids": token_ids}
        if len(val_doc_reservoir) < val_reservoir_size:
            val_doc_reservoir.append(doc_record)
        else:
            replace_idx = rng.randint(0, doc_id)
            if replace_idx < val_reservoir_size:
                val_doc_reservoir[replace_idx] = doc_record

        end = min(total_tokens + len(token_ids), args.max_train_tokens)
        write_len = end - total_tokens
        if write_len <= 0:
            break

        tokens_mm[total_tokens:end] = np.asarray(token_ids[:write_len], dtype=np.uint32)
        doc_ids_mm[total_tokens:end] = np.uint32(doc_id)
        positions_mm[total_tokens:end] = np.arange(write_len, dtype=np.uint32)
        total_tokens = end
        progress.update(write_len)
        doc_id += 1

        if total_tokens >= args.max_train_tokens:
            break

    progress.close()

    for mm in (tokens_mm, doc_ids_mm, positions_mm):
        mm.flush()
        del mm

    # Shrink files to the tokens actually written.
    for path in (train_tokens_path, train_doc_ids_path, train_positions_path):
        truncate_bin(path, total_tokens)

    integrity = integrity_check(train_tokens_path, train_doc_ids_path, total_tokens)
    log(f"Integrity check passed: {json.dumps(integrity)}")

    build_val_split(
        val_dir=args.output_dir / "val",
        docs=val_doc_reservoir,
        seq_len=args.seq_len,
        val_tokens_target=args.val_tokens_target,
        seed=args.seed,
    )

    meta = {
        "dataset": args.dataset,
        "hf_path": spec["hf_path"],
        "hf_name": spec["hf_name"],
        "tokenizer": args.tokenizer,
        "num_tokens": int(total_tokens),
        "num_docs": int(doc_id),
        "seq_len": args.seq_len,
        "val_tokens_target": args.val_tokens_target,
        "integrity": integrity,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Wrote {total_tokens:,} training tokens ({doc_id:,} docs) to {args.output_dir}")


def build_val_split(
    val_dir: Path,
    docs: list[dict],
    seq_len: int,
    val_tokens_target: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    val_dir.mkdir(parents=True, exist_ok=True)

    sequences: list[list[int]] = []
    doc_id_rows: list[list[int]] = []
    position_rows: list[list[int]] = []
    loss_tokens = 0

    docs_shuffled = docs[:]
    rng.shuffle(docs_shuffled)

    buffer: list[tuple[int, int, int]] = []
    for doc in docs_shuffled:
        for pos, token in enumerate(doc["token_ids"]):
            buffer.append((token, doc["doc_id"], pos))
            if len(buffer) == seq_len:
                sequences.append([item[0] for item in buffer])
                doc_id_rows.append([item[1] for item in buffer])
                position_rows.append([item[2] for item in buffer])
                loss_tokens += seq_len - 1
                buffer = []
                if loss_tokens >= val_tokens_target:
                    break
        if loss_tokens >= val_tokens_target:
            break

    if not sequences:
        raise RuntimeError("Failed to build validation split; no validation sequences created.")

    np.save(val_dir / "sequences.npy", np.asarray(sequences, dtype=np.uint32))
    np.save(val_dir / "doc_ids.npy", np.asarray(doc_id_rows, dtype=np.uint32))
    np.save(val_dir / "positions.npy", np.asarray(position_rows, dtype=np.uint32))
    write_val_manifest(
        val_dir,
        seq_len=seq_len,
        num_sequences=len(sequences),
        num_loss_tokens=loss_tokens,
    )
    log(f"Wrote validation manifest with {loss_tokens:,} loss tokens across {len(sequences)} sequences")


if __name__ == "__main__":
    main()
