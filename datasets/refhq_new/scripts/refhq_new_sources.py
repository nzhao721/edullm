#!/usr/bin/env python3
"""Shared constants for the refhq-new instruct CE corpus build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from refhq_new.domain_map import DOMAINS, SOURCES
from refhq_new.exclusion import load_exclusion_rules

TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257
DEFAULT_SEED = 42
HOLDOUT_FRACTION = 0.0015  # 0.15% of documents per (source, domain)
DEFAULT_SCRATCH_ROOT = Path("/scratch/users/nzhao2/refhq-new-v1")
DEFAULT_S3_BUCKET = "edullm-datasets"
DEFAULT_S3_PREFIX = "refhq/refhq-new"
# Smaller shards → more Slurm array tasks for English filter + tokenize.
DOCS_PER_SHARD = 10_000
SPLITS: tuple[str, ...] = ("train", "val")


def hf_repo_for_source(source: str, rules: dict[str, Any] | None = None) -> str:
    bundle = rules if rules is not None else load_exclusion_rules()
    cfg = bundle["sources"][source]
    repo = cfg.get("hf_repo")
    if not repo:
        raise ValueError(f"exclusion rules missing hf_repo for {source}")
    return str(repo)


def source_specs(rules: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return per-source HF download/normalize specs."""
    bundle = rules if rules is not None else load_exclusion_rules()
    specs: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        cfg = bundle["sources"][source]
        specs[source] = {
            "hf_repo": str(cfg["hf_repo"]),
            "repo_type": "dataset",
            "split": "train",
            "multi_config": source == "smoltalk",
            "gated": source in {"hermes-3"},  # Hermes often needs HF auth
        }
    return specs


def scratch_layout(root: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    return {
        "root": root_path,
        "raw": root_path / "raw",
        "docs": root_path / "docs",
        "out": root_path / "out",
        "holdout": root_path / "holdout",
        "work": root_path / "work",
        "tokenized": root_path / "tokenized",
        "manifests": root_path / "manifests",
        "logs": root_path / "logs",
    }


def holdout_counts(n_docs: int, fraction: float = HOLDOUT_FRACTION) -> tuple[int, int]:
    """Return (n_train, n_val) for a document pool.

    Uses round(n * fraction). Keeps at least one train doc when n_docs > 1.
    Returns (0, 0) for empty pools; (n_docs, 0) when the carve rounds to 0.
    """
    if n_docs <= 0:
        return 0, 0
    if not (0.0 < fraction < 0.5):
        raise ValueError(f"holdout fraction must be in (0, 0.5); got {fraction}")
    n_val = int(round(n_docs * fraction))
    if n_docs > 1:
        n_val = min(n_val, n_docs - 1)
    else:
        n_val = 0
    return n_docs - n_val, n_val
