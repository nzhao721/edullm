"""Shared config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from token_selection.scripts.train_data_resolve import (
    DATA_BUCKET,
    DEFAULT_TRAIN_DATASET_ID,
    resolve_tokens_s3,
    resolve_train_dataset,
    resolve_train_dataset_id,
    resolve_train_dataset_version,
    resolve_train_split,
)

__all__ = [
    "DATA_BUCKET",
    "DEFAULT_TRAIN_DATASET_ID",
    "derive_steps",
    "load_config",
    "resolve_output_dir",
    "resolve_tokens_s3",
    "resolve_train_dataset",
    "resolve_train_dataset_id",
    "resolve_train_dataset_version",
    "resolve_train_split",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return cfg


def resolve_output_dir(cfg: Dict[str, Any], root: Path | None = None) -> Path:
    root = root or Path.cwd()
    out = Path(cfg.get("output_dir", "token_selection/data/out"))
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def derive_steps(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Return ``(total_steps, t0_steps)`` from max_tokens / GBS / t0_frac.

    An explicit ``t0_steps`` in the config wins over ``t0_frac``. Pin it when a run
    may later extend ``max_tokens`` (e.g. 5B then continue to 10B) so the REL warmup
    boundary stays fixed across the resume.

    ``t0_steps=0`` / ``t0_frac=0`` means selection is active from step 0 (no masking
    warmup). Online scorer arms in the token-selection plan use this.
    """
    max_tokens = int(cfg["train"]["max_tokens"])
    gbs = int(cfg["train"]["global_batch_size"])
    if gbs <= 0:
        raise ValueError("train.global_batch_size must be > 0")
    total_steps = max(1, max_tokens // gbs)
    if "t0_steps" in cfg and cfg["t0_steps"] is not None:
        t0_steps = max(0, int(cfg["t0_steps"]))
    else:
        t0_frac = float(cfg.get("t0_frac", 0.02))
        if t0_frac <= 0:
            t0_steps = 0
        else:
            t0_steps = max(1, int(round(total_steps * t0_frac)))
    t0_steps = min(t0_steps, total_steps)
    return total_steps, t0_steps
