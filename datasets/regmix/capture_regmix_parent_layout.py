#!/usr/bin/env python3
"""Capture parent_layout.json for RegMix curriculum index builds.

Combines the published pretrain/regmix-10b train-shard layout (dataset_paths
order) with full-stream source_total_tokens from the local tokenized/*.json
metas (train+val = original packed npy). Val was carved from the stream tail,
so train shard offsets are a contiguous prefix of the full stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257
DEFAULT_DATASET_ID = "pretrain/regmix-10b"
DEFAULT_VERSION = "v1"
DEFAULT_MANIFEST_SHA256 = (
    "a24992f53dc4a900bacf8fa571d77e343fd28ffa9054c14b93d54204b0a38cb4"
)
SEQ_LEN = 2048

# Published train shards in dataset_paths() order (from tokens/manifest.json).
# Counts are train-only after the 0.15% val carve from each source tail.
PUBLISHED_TRAIN_SHARDS = [
    ("tokens/algebraic-stack/train-00000.u32le.bin", "algebraic-stack", 268435456),
    ("tokens/algebraic-stack/train-00001.u32le.bin", "algebraic-stack", 268435456),
    ("tokens/algebraic-stack/train-00002.u32le.bin", "algebraic-stack", 77445247),
    ("tokens/arxiv/train-00000.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00001.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00002.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00003.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00004.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00005.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00006.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00007.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00008.u32le.bin", "arxiv", 268435456),
    ("tokens/arxiv/train-00009.u32le.bin", "arxiv", 80493557),
    ("tokens/dclm/train-00000.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00001.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00002.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00003.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00004.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00005.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00006.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00007.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00008.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00009.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00010.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00011.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00012.u32le.bin", "dclm", 268435456),
    ("tokens/dclm/train-00013.u32le.bin", "dclm", 257511711),
    ("tokens/open-web-math/train-00000.u32le.bin", "open-web-math", 268435456),
    ("tokens/open-web-math/train-00001.u32le.bin", "open-web-math", 268435456),
    ("tokens/open-web-math/train-00002.u32le.bin", "open-web-math", 97275218),
    ("tokens/pes2o/train-00000.u32le.bin", "pes2o", 268435456),
    ("tokens/pes2o/train-00001.u32le.bin", "pes2o", 268435456),
    ("tokens/pes2o/train-00002.u32le.bin", "pes2o", 268435456),
    ("tokens/pes2o/train-00003.u32le.bin", "pes2o", 131443707),
    ("tokens/starcoder/train-00000.u32le.bin", "starcoder", 268435456),
    ("tokens/starcoder/train-00001.u32le.bin", "starcoder", 268435456),
    ("tokens/starcoder/train-00002.u32le.bin", "starcoder", 268435456),
    ("tokens/starcoder/train-00003.u32le.bin", "starcoder", 268435456),
    ("tokens/starcoder/train-00004.u32le.bin", "starcoder", 268435456),
    ("tokens/starcoder/train-00005.u32le.bin", "starcoder", 62698626),
    ("tokens/wiki/train-00000.u32le.bin", "wiki", 156126264),
]


def load_full_stream_totals(tokenized_root: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for meta_path in sorted(tokenized_root.glob("*/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        domain = meta.get("domain") or meta_path.parent.name
        tokens = int(meta["tokens_with_eos"])
        npy = meta_path.with_suffix(".npy")
        if not npy.is_file():
            npy = meta_path.parent / f"{domain}.npy"
        actual = npy.stat().st_size // 4
        if actual != tokens:
            raise SystemExit(
                f"{domain}: meta tokens_with_eos={tokens} != npy u32 count={actual}"
            )
        totals[str(domain)] = tokens
    if not totals:
        raise SystemExit(f"no tokenized metas under {tokenized_root}")
    return totals


def build_layout(*, tokenized_root: Path, dataset_id: str, version: str, manifest_sha256: str) -> dict:
    source_totals = load_full_stream_totals(tokenized_root)
    shards = []
    source_starts: dict[str, int] = {}
    for path, source, count in PUBLISHED_TRAIN_SHARDS:
        start = source_starts.get(source, 0)
        shards.append(
            {
                "path": path,
                "count": int(count),
                "source": source,
                "source_token_start": int(start),
            }
        )
        source_starts[source] = start + int(count)
    # Train must be a prefix of the full packed stream.
    for source, train_end in source_starts.items():
        full = source_totals.get(source)
        if full is None:
            raise SystemExit(f"missing full-stream total for {source}")
        if train_end > full:
            raise SystemExit(
                f"{source}: train end {train_end} exceeds full stream {full}"
            )
        val = full - train_end
        if val < 0:
            raise SystemExit(f"{source}: negative val carve")
    return {
        "dataset_id": dataset_id,
        "version": version,
        "manifest_sha256": manifest_sha256,
        "seq_len": SEQ_LEN,
        "tokenizer_id": TOKENIZER_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "source_total_tokens": source_totals,
        "source_train_tokens": source_starts,
        "shards": shards,
        "notes": (
            "source_total_tokens is the full packed stream (train+val). "
            "Train shards are a contiguous prefix after the 0.15% val carve from each source tail."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenized-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--version", default=DEFAULT_VERSION)
    ap.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    args = ap.parse_args()

    layout = build_layout(
        tokenized_root=args.tokenized_root,
        dataset_id=args.dataset_id,
        version=args.version,
        manifest_sha256=args.manifest_sha256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    train_total = sum(s["count"] for s in layout["shards"])
    full_total = sum(layout["source_total_tokens"].values())
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_shards": len(layout["shards"]),
                "train_tokens": train_total,
                "full_stream_tokens": full_total,
                "val_tokens": full_total - train_total,
                "sources": sorted(layout["source_total_tokens"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
