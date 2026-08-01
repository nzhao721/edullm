#!/usr/bin/env python3
"""Pack self-contained Co-LMLM annotations into SmolLM2 tokens and loss masks.

The text is not modified. A token's causal-LM target is masked when its
half-open character interval intersects any annotated fact interval, including
tokens that cross either fact boundary.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import zstandard
from transformers import AutoTokenizer


@dataclass(frozen=True)
class PackedShard:
    tokens: str
    mask: str
    sequences: int
    tokens_count: int
    masked_targets: int


def token_fact_mask(
    offsets: Iterable[tuple[int, int]], spans: Iterable[tuple[int, int]]
) -> np.ndarray:
    """Return true for every token interval intersecting a fact interval."""
    ordered = sorted((int(start), int(end)) for start, end in spans if end > start)
    result: list[bool] = []
    span_idx = 0
    for token_start, token_end in offsets:
        token_start, token_end = int(token_start), int(token_end)
        if token_end <= token_start:
            result.append(False)
            continue
        while span_idx < len(ordered) and ordered[span_idx][1] <= token_start:
            span_idx += 1
        hit = (
            span_idx < len(ordered)
            and token_end > ordered[span_idx][0]
            and token_start < ordered[span_idx][1]
        )
        result.append(hit)
    return np.asarray(result, dtype=np.uint8)


def _iter_jsonl_zst(path: Path) -> Iterator[dict]:
    with path.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line_no, line in enumerate(text, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                yield record


def annotation_files(root: Path, expected_shards: int | None) -> list[Path]:
    files = sorted(root.rglob("*.annotations.jsonl.zst"))
    if not files:
        raise FileNotFoundError(f"no annotation shards under {root}")
    if expected_shards is not None and len(files) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} annotation shards, found {len(files)} under {root}"
        )
    stems = [p.name.split(".annotations.", 1)[0] for p in files]
    if len(stems) != len(set(stems)):
        raise ValueError(f"duplicate annotation shard names: {stems}")
    return files


def _validated_spans(record: dict, *, source: Path) -> list[tuple[int, int]]:
    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{source}: record id={record.get('id')!r} has no text")
    spans: list[tuple[int, int]] = []
    for annotation in record.get("annotations", []):
        start = int(annotation["char_start"])
        end = int(annotation["char_end"])
        if start < 0 or end <= start or end > len(text):
            raise ValueError(
                f"{source}: invalid span [{start},{end}) for id={record.get('id')!r}"
            )
        if text[start:end] != annotation.get("span"):
            raise ValueError(
                f"{source}: unfaithful span [{start},{end}) for id={record.get('id')!r}"
            )
        spans.append((start, end))
    spans.sort()
    return spans


class ShardWriter:
    def __init__(self, out_dir: Path, *, seq_len: int, sequences_per_shard: int):
        self.out_dir = out_dir
        self.seq_len = seq_len
        self.sequences_per_shard = sequences_per_shard
        self.pending_tokens: list[int] = []
        self.pending_mask: list[int] = []
        self.token_rows = np.empty((sequences_per_shard, seq_len), dtype="<u4")
        self.mask_rows = np.empty((sequences_per_shard, seq_len), dtype=np.uint8)
        self.rows = 0
        self.shard_index = 0
        self.shards: list[PackedShard] = []

    def add_document(self, token_ids: list[int], mask: np.ndarray, eos_id: int) -> None:
        if len(token_ids) != len(mask):
            raise ValueError("token/mask length mismatch")
        self.pending_tokens.extend(int(v) for v in token_ids)
        self.pending_tokens.append(int(eos_id))
        self.pending_mask.extend(int(v) for v in mask)
        self.pending_mask.append(0)
        while len(self.pending_tokens) >= self.seq_len:
            self.token_rows[self.rows] = self.pending_tokens[: self.seq_len]
            self.mask_rows[self.rows] = self.pending_mask[: self.seq_len]
            del self.pending_tokens[: self.seq_len]
            del self.pending_mask[: self.seq_len]
            self.rows += 1
            if self.rows == self.sequences_per_shard:
                self.flush()

    def flush(self) -> None:
        if self.rows == 0:
            return
        token_name = f"train-{self.shard_index:05d}.tokens.u32le.bin"
        mask_name = f"train-{self.shard_index:05d}.mask.u8.bin"
        tokens = self.token_rows[: self.rows]
        mask = self.mask_rows[: self.rows]
        tokens.tofile(self.out_dir / token_name)
        mask.tofile(self.out_dir / mask_name)
        count = int(self.rows * self.seq_len)
        self.shards.append(
            PackedShard(
                tokens=token_name,
                mask=mask_name,
                sequences=self.rows,
                tokens_count=count,
                masked_targets=int(mask.sum()),
            )
        )
        self.rows = 0
        self.shard_index += 1


def _batched(records: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare(args: argparse.Namespace) -> dict:
    ready = args.output_dir / "_READY.json"
    if ready.exists() and not args.overwrite:
        return json.loads(ready.read_text(encoding="utf-8"))
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} exists without _READY.json; pass --overwrite"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    files = annotation_files(args.annotations_dir, args.expected_shards)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for offset mappings")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"{args.tokenizer} has no eos_token_id")

    writer = ShardWriter(
        args.output_dir,
        seq_len=args.seq_len,
        sequences_per_shard=args.sequences_per_shard,
    )
    docs = facts = input_chars = 0
    for file_index, path in enumerate(files, 1):
        file_docs = 0
        for batch in _batched(_iter_jsonl_zst(path), args.tokenizer_batch_size):
            texts = []
            span_batches = []
            for record in batch:
                text = record.get("text")
                spans = _validated_spans(record, source=path)
                texts.append(text)
                span_batches.append(spans)
                docs += 1
                file_docs += 1
                facts += len(spans)
                input_chars += len(text)
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            for ids, offsets, spans in zip(
                encoded["input_ids"], encoded["offset_mapping"], span_batches
            ):
                writer.add_document(ids, token_fact_mask(offsets, spans), eos_id)
        print(
            f"prepared {file_index}/{len(files)} {path.name}: {file_docs:,} docs",
            flush=True,
        )
    writer.flush()

    total_tokens = sum(s.tokens_count for s in writer.shards)
    masked = sum(s.masked_targets for s in writer.shards)
    manifest = {
        "schema_version": "smollm2-colmlm-packed/v1",
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "annotation_shards": [str(p.relative_to(args.annotations_dir)) for p in files],
        "documents": docs,
        "facts": facts,
        "input_chars": input_chars,
        "sequences": sum(s.sequences for s in writer.shards),
        "tokens": total_tokens,
        "masked_targets": masked,
        "masked_fraction": masked / total_tokens if total_tokens else 0.0,
        "dropped_tail_tokens": len(writer.pending_tokens),
        "shards": [asdict(s) for s in writer.shards],
    }
    tmp = ready.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(ready)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--expected-shards", type=int, default=19)
    parser.add_argument("--tokenizer-batch-size", type=int, default=64)
    parser.add_argument("--sequences-per-shard", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    manifest = prepare(parse_args())
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
