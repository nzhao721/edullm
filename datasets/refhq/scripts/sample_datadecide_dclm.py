#!/usr/bin/env python3
"""Sample ~1B tokens from DataDecide DCLM-Baseline QC 7% FW2 tokenized shards.

DataDecide stores uint16/uint32 token id streams as .npy parts under the
gpt-neox-olmo-dolma-v1_5 tokenizer. We decode with that source tokenizer, then
retokenize with the HQ budget tokenizer (OLMo-2) so realized_tokens match the
rest of the corpus. Processing is streamed to avoid holding the full budget
in memory.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

SOURCE_TOKENIZER_ID = "allenai/gpt-neox-olmo-dolma-v1_5"
TARGET_TOKENIZER_ID = "allenai/dolma2-tokenizer"


def iter_token_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("**/*.npy"))
    if not files:
        raise SystemExit(f"no .npy files under {raw_dir}")
    return files


def write_doc(handle, doc: dict) -> None:
    handle.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_token_memmap(path: Path) -> np.ndarray:
    """Load DataDecide/OLMo token shard.

    These ``.npy`` files are raw memmaps (usually uint16 for gpt-neox vocab),
    not NumPy ``.npy`` archives with headers.
    """
    size = path.stat().st_size
    if size % 2 == 0:
        try:
            mm = np.memmap(path, dtype=np.uint16, mode="r")
            # Sanity: token ids should fit NeoX/Dolma-v1.5 vocab (~50k).
            sample = np.asarray(mm[: min(4096, mm.size)])
            if sample.size and int(sample.max()) < 100_000:
                return mm
        except Exception:
            pass
    if size % 4 == 0:
        return np.memmap(path, dtype=np.uint32, mode="r")
    # Last resort: standard npy
    return np.load(path, mmap_mode="r")


def sample_stream(
    files: list[Path],
    target_tokens: int,
    seed: int,
    source_tokenizer_id: str,
    target_tokenizer_id: str,
    chunk_tokens: int,
    out_dir: Path,
    max_docs_per_shard: int = 50_000,
) -> tuple[list[Path], int, int, int]:
    from transformers import AutoTokenizer

    src_tok = AutoTokenizer.from_pretrained(source_tokenizer_id, use_fast=True)
    try:
        tgt_tok = AutoTokenizer.from_pretrained(target_tokenizer_id, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        print(f"fast target tokenizer failed ({exc!r}); retrying use_fast=False", flush=True)
        tgt_tok = AutoTokenizer.from_pretrained(target_tokenizer_id, use_fast=False)

    rng = np.random.default_rng(seed)
    order = list(range(len(files)))
    rng.shuffle(order)

    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    handle = None
    shard_idx = 0
    count_in_shard = 0
    doc_idx = 0
    realized = 0
    source_seen = 0

    try:
        for idx in order:
            if realized >= target_tokens:
                break
            path = files[idx]
            print(f"sampling {path.name} realized={realized}", flush=True)
            try:
                arr = load_token_memmap(path)
            except Exception as exc:  # noqa: BLE001
                print(f"skip bad npy {path.name}: {exc}", flush=True)
                continue
            flat = np.asarray(arr).reshape(-1)
            if flat.size == 0:
                continue
            start = int(rng.integers(0, max(1, flat.size)))
            # Walk a circular view without materializing the full concat.
            remaining = int(flat.size)
            pos = start
            while remaining > 0 and realized < target_tokens:
                take = min(chunk_tokens, remaining, max(chunk_tokens, target_tokens - realized + chunk_tokens))
                end = pos + take
                if end <= flat.size:
                    chunk = np.asarray(flat[pos:end], dtype=np.int64)
                else:
                    first = np.asarray(flat[pos:], dtype=np.int64)
                    second = np.asarray(flat[: end - flat.size], dtype=np.int64)
                    chunk = np.concatenate([first, second])
                pos = end % flat.size
                remaining -= int(chunk.size)
                source_seen += int(chunk.size)

                text = src_tok.decode(chunk.tolist(), skip_special_tokens=True)
                if not text.strip():
                    continue
                tgt_ids = tgt_tok.encode(text, add_special_tokens=False)
                n = len(tgt_ids)
                if n == 0:
                    continue
                if realized + n > target_tokens:
                    keep = target_tokens - realized
                    if keep <= 0:
                        break
                    text = tgt_tok.decode(tgt_ids[:keep], skip_special_tokens=True)
                    n = keep

                if handle is None or count_in_shard >= max_docs_per_shard:
                    if handle is not None:
                        handle.close()
                    shard = out_dir / f"documents-{shard_idx:05d}.json.gz"
                    handle = gzip.open(shard, "wt", encoding="utf-8")
                    shards.append(shard)
                    shard_idx += 1
                    count_in_shard = 0

                write_doc(
                    handle,
                    {
                        "id": f"datadecide-qc7p-fw2-{doc_idx:08d}",
                        "text": text,
                        "source": "allenai/DataDecide-data-recipes",
                        "recipe": "dclm-baseline-qc-7p-fw2",
                        "n_tokens": n,
                        "source_tokenizer_id": source_tokenizer_id,
                        "tokenizer_id": target_tokenizer_id,
                    },
                )
                doc_idx += 1
                count_in_shard += 1
                realized += n
                if doc_idx % 500 == 0:
                    print(f"docs={doc_idx} realized={realized}", flush=True)
    finally:
        if handle is not None:
            handle.close()

    return shards, realized, source_seen, doc_idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Downloaded DataDecide .npy dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--tokenizer", default=TARGET_TOKENIZER_ID)
    parser.add_argument("--source-tokenizer", default=SOURCE_TOKENIZER_ID)
    parser.add_argument("--stats-out", type=Path, default=None)
    args = parser.parse_args()

    files = iter_token_files(args.raw_dir)
    print(f"found {len(files)} npy shards under {args.raw_dir}", flush=True)

    shards, realized, source_seen, doc_count = sample_stream(
        files,
        args.target_tokens,
        args.seed,
        args.source_tokenizer,
        args.tokenizer,
        args.chunk_tokens,
        args.out_dir,
    )
    stats = {
        "domain": "dclm",
        "source": "allenai/DataDecide-data-recipes",
        "recipe": "v0_rep32_ft7percentile_fw2",
        "target_tokens": args.target_tokens,
        "realized_tokens": realized,
        "source_tokens_sampled": source_seen,
        "doc_count": doc_count,
        "shard_count": len(shards),
        "seed": args.seed,
        "source_tokenizer_id": args.source_tokenizer,
        "tokenizer_id": args.tokenizer,
        "shards": [str(p) for p in shards],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.stats_out or (args.out_dir / "stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    if realized < int(args.target_tokens * 0.98):
        raise SystemExit(
            f"dclm realized_tokens={realized} below 98% of target={args.target_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
