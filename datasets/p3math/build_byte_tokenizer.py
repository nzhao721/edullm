#!/usr/bin/env python3
"""Build a HF-format raw UTF-8 byte tokenizer (vocab 0..255) for edullm-data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_byte_tokenizer(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # HuggingFace tokenizers.json model.vocab: token_string -> id
    # Represent each byte as a single Latin-1 char so ids map 1:1 with UTF-8 bytes.
    vocab = {chr(i): i for i in range(256)}
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [],
        },
    }
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 1_000_000,
                "add_prefix_space": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "special_tokens_map.json").write_text("{}\n", encoding="utf-8")
    # Empty merges keeps HF-style riders present without introducing merges.
    (out_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return out_dir / "tokenizer.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    path = build_byte_tokenizer(args.out_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
