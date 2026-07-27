#!/usr/bin/env python3
"""Shared constants and helpers for the HQ reference corpus build."""

from __future__ import annotations

from typing import Any

TOKENIZER_ID = "allenai/dolma2-tokenizer"
DEFAULT_SEED = 42
BUDGET_TOLERANCE = 0.02  # ±2%

# HQ token budgets (exact targets for acceptance).
HQ_BUDGETS: dict[str, float] = {
    "dclm": 1.0e9,
    "starcoder": 0.5e9,
    "pes2o": 0.5e9,
    "arxiv": 0.5e9,
    "open-web-math": 0.5e9,
    "algebraic-stack": 0.5e9,
    "wiki": 0.5e9,
}

# RegMix-10B weights scaled to ~5.514B strict-HQ ceiling (OWM pool binds).
REGMIX_5P5_BUDGETS: dict[str, float] = {
    "dclm": 2_067_750_000,
    "arxiv": 1_378_500_000,
    "starcoder": 775_268_000,
    "pes2o": 517_213_000,
    "open-web-math": 350_139_000,
    "algebraic-stack": 339_111_000,
    "wiki": 86_018_400,
}
REGMIX_5P5_DCLM_MAX_FILES = 52  # ~2B source tokens from DataDecide QC shards

# Published / measured unfiltered pool sizes (tokens).
UNFILTERED_POOL_TOKENS: dict[str, float] = {
    "dclm": 100e9,  # DataDecide QC 7% FW2 recipe stream
    "starcoder": 83.0e9,
    "pes2o": 58.6e9,
    "arxiv": 20.8e9,
    "open-web-math": 12.2e9,
    "algebraic-stack": 11.8e9,
    "wiki": 3.66e9,
}

# Mini-batch size for Dolma code filtering (tag/mix is shard-oriented).
DOLMA_MINI_BATCH_DOCS = 4_000

HQ_DOMAINS: tuple[str, ...] = tuple(HQ_BUDGETS.keys())

# Hugging Face source specs per domain.
# kind: "datadecide_npy" | "hf_dataset" | "olmo_mix_domain"
HQ_SOURCES: dict[str, dict[str, Any]] = {
    "dclm": {
        "kind": "datadecide_npy",
        "repo_id": "allenai/DataDecide-data-recipes",
        "repo_type": "dataset",
        "prefix": "preprocessed/dclm/v0_rep32_ft7percentile_fw2/gpt-neox-olmo-dolma-v1_5/",
        # DataDecide shards are tokenized with this NeoX/Dolma-v1.5 vocab; decode
        # with it, then retokenize with TOKENIZER_ID for budget accounting.
        "source_tokenizer_id": "allenai/gpt-neox-olmo-dolma-v1_5",
        # ~2GB / ~0.9–1B source tokens per part; 24 parts >> 1B OLMo-2 budget.
        "max_files": 24,
        "filter": "DataDecide DCLM-Baseline QC 7% FW2",
        "gated": False,
    },
    "starcoder": {
        "kind": "hf_dataset",
        "repo_id": "bigcode/starcoderdata",
        "repo_type": "dataset",
        "config": None,
        "split": "train",
        "text_field": "content",
        "filter": "dolma-code-hq",
        "gated": True,
        "dolma_domain": "code-hq",
    },
    "pes2o": {
        "kind": "hf_dataset",
        "repo_id": "allenai/peS2o",
        "repo_type": "dataset",
        "config": None,
        "split": "train",
        "text_field": "text",
        "filter": "random",
        "gated": False,
    },
    "arxiv": {
        "kind": "olmo_mix_domain",
        "repo_id": "allenai/olmo-mix-1124",
        "repo_type": "dataset",
        "path_prefix": "data/arxiv/",
        "filter": "random",
        "gated": False,
    },
    "open-web-math": {
        "kind": "hf_dataset",
        "repo_id": "open-web-math/open-web-math",
        "repo_type": "dataset",
        "config": None,
        "split": "train",
        "text_field": "text",
        "filter": "openwebmath-hq",
        "gated": False,
    },
    "algebraic-stack": {
        "kind": "hf_dataset",
        "repo_id": "typeof/algebraic-stack",
        "repo_type": "dataset",
        "config": None,
        "split": "train",
        "text_field": "text",
        "filter": "algebraic-stack-heuristic",
        "gated": False,
    },
    "wiki": {
        "kind": "hf_dataset",
        "repo_id": "wikimedia/wikipedia",
        "repo_type": "dataset",
        "config": "20231101.en",
        "split": "train",
        "text_field": "text",
        "filter": "random",
        "gated": False,
    },
}


def scratch_layout(root: str | Any) -> dict[str, Any]:
    from pathlib import Path

    root_path = Path(root)
    return {
        "root": root_path,
        "raw": root_path / "raw",
        "work": root_path / "work",
        "out": root_path / "out",
        "manifests": root_path / "manifests",
        "logs": root_path / "logs",
    }


def within_budget(realized: float, budget: float, tolerance: float = BUDGET_TOLERANCE) -> bool:
    if budget <= 0:
        return realized == 0
    return abs(realized - budget) / budget <= tolerance
