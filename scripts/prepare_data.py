from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.token_loss import write_val_manifest


DATASET_SPECS = {
    "fineweb_edu": {
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
    parser = argparse.ArgumentParser(description="Tokenize corpora into memmap shards.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        required=True,
        help="Which corpus to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory, e.g. data/slimpajama",
    )
    parser.add_argument(
        "--tokenizer",
        default="EleutherAI/pythia-70m",
        help="Tokenizer used for all runs.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for validation windows.",
    )
    parser.add_argument(
        "--max-train-tokens",
        type=int,
        default=None,
        help="Cap number of unique training tokens to write.",
    )
    parser.add_argument(
        "--val-tokens-target",
        type=int,
        default=320000,
        help="Target number of predicted tokens in the validation manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
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


def tokenize_text(tokenizer, text: str) -> list[int]:
    if not text or not text.strip():
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def main() -> None:
    args = parse_args()
    log(f"prepare_data: dataset={args.dataset} max_tokens={args.max_train_tokens} out={args.output_dir}")
    rng = random.Random(args.seed)
    spec = DATASET_SPECS[args.dataset]
    log(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    log("Tokenizer ready.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tokens_path = args.output_dir / "train_tokens.bin"
    train_doc_ids_path = args.output_dir / "train_doc_ids.bin"
    train_positions_path = args.output_dir / "train_positions.bin"

    # Reservoir sample of docs for validation manifest.
    val_doc_reservoir: list[dict] = []
    val_reservoir_size = 512

    doc_id = 0
    total_tokens = 0
    # Open memmaps lazily after first write.
    tokens_mm = None
    doc_ids_mm = None
    positions_mm = None
    capacity = 0

    def ensure_capacity(min_size: int) -> None:
        nonlocal capacity, tokens_mm, doc_ids_mm, positions_mm
        if capacity >= min_size:
            return
        new_capacity = max(min_size, max(1, capacity) * 2)
        for path in (train_tokens_path, train_doc_ids_path, train_positions_path):
            if not path.exists():
                np.memmap(path, dtype=np.uint32, mode="w+", shape=(new_capacity,))
            else:
                old = np.memmap(path, dtype=np.uint32, mode="r", shape=(capacity,))
                updated = np.memmap(path, dtype=np.uint32, mode="w+", shape=(new_capacity,))
                updated[:capacity] = old
        tokens_mm = np.memmap(train_tokens_path, dtype=np.uint32, mode="r+", shape=(new_capacity,))
        doc_ids_mm = np.memmap(train_doc_ids_path, dtype=np.uint32, mode="r+", shape=(new_capacity,))
        positions_mm = np.memmap(train_positions_path, dtype=np.uint32, mode="r+", shape=(new_capacity,))
        capacity = new_capacity

    stream = open_stream(args.dataset)
    progress = tqdm(desc=f"prepare {args.dataset}", unit="tok")

    for row in stream:
        text = row.get(spec["text_column"]) or ""
        token_ids = tokenize_text(tokenizer, text)
        if len(token_ids) < 2:
            continue

        doc_record = {
            "doc_id": doc_id,
            "token_ids": token_ids,
        }
        if len(val_doc_reservoir) < val_reservoir_size:
            val_doc_reservoir.append(doc_record)
        else:
            replace_idx = rng.randint(0, doc_id)
            if replace_idx < val_reservoir_size:
                val_doc_reservoir[replace_idx] = doc_record

        ensure_capacity(total_tokens + len(token_ids))
        end = total_tokens + len(token_ids)
        if args.max_train_tokens is not None:
            end = min(end, args.max_train_tokens)
            write_len = end - total_tokens
        else:
            write_len = len(token_ids)

        if write_len <= 0:
            break

        slice_tokens = np.asarray(token_ids[:write_len], dtype=np.uint32)
        tokens_mm[total_tokens:end] = slice_tokens
        doc_ids_mm[total_tokens:end] = np.uint32(doc_id)
        positions_mm[total_tokens:end] = np.arange(write_len, dtype=np.uint32)
        total_tokens = end
        progress.update(write_len)
        doc_id += 1

        if args.max_train_tokens is not None and total_tokens >= args.max_train_tokens:
            break

    progress.close()

    tokens_mm = np.memmap(train_tokens_path, dtype=np.uint32, mode="r+", shape=(capacity,))
    doc_ids_mm = np.memmap(train_doc_ids_path, dtype=np.uint32, mode="r+", shape=(capacity,))
    positions_mm = np.memmap(train_positions_path, dtype=np.uint32, mode="r+", shape=(capacity,))
    tokens_mm.flush()
    doc_ids_mm.flush()
    positions_mm.flush()

    final_tokens = np.memmap(train_tokens_path, dtype=np.uint32, mode="r+", shape=(total_tokens,))
    final_doc_ids = np.memmap(train_doc_ids_path, dtype=np.uint32, mode="r+", shape=(total_tokens,))
    final_positions = np.memmap(train_positions_path, dtype=np.uint32, mode="r+", shape=(total_tokens,))
    final_tokens.flush()
    final_doc_ids.flush()
    final_positions.flush()

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
        "num_tokens": total_tokens,
        "num_docs": doc_id,
        "seq_len": args.seq_len,
        "val_tokens_target": args.val_tokens_target,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {total_tokens:,} training tokens to {args.output_dir}")


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

    # Pack validation windows from held-out docs.
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

    seq_array = np.asarray(sequences, dtype=np.uint32)
    doc_array = np.asarray(doc_id_rows, dtype=np.uint32)
    pos_array = np.asarray(position_rows, dtype=np.uint32)
    np.save(val_dir / "sequences.npy", seq_array)
    np.save(val_dir / "doc_ids.npy", doc_array)
    np.save(val_dir / "positions.npy", pos_array)
    write_val_manifest(
        val_dir,
        seq_len=seq_len,
        num_sequences=len(sequences),
        num_loss_tokens=loss_tokens,
    )
    print(f"Wrote validation manifest with {loss_tokens:,} loss tokens across {len(sequences)} sequences")


if __name__ == "__main__":
    main()
