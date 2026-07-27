"""Shared config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


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
    """
    max_tokens = int(cfg["train"]["max_tokens"])
    gbs = int(cfg["train"]["global_batch_size"])
    if gbs <= 0:
        raise ValueError("train.global_batch_size must be > 0")
    total_steps = max(1, max_tokens // gbs)
    if cfg.get("t0_steps") is not None:
        t0_steps = max(1, int(cfg["t0_steps"]))
    else:
        t0_frac = float(cfg.get("t0_frac", 0.02))
        t0_steps = max(1, int(round(total_steps * t0_frac)))
    t0_steps = min(t0_steps, total_steps)
    return total_steps, t0_steps


def resolve_tokens_s3(cfg: Dict[str, Any]) -> str:
    """Return the pre-tokenized shard directory URI from ``data.tokens_s3``."""
    uri = str((cfg.get("data") or {}).get("tokens_s3") or "").strip().rstrip("/")
    if not uri.startswith("s3://"):
        raise ValueError(
            "data.tokens_s3 must be an s3:// URI to the pre-tokenized corpus directory "
            "(per-domain <domain>/<domain>.npy shards, their .json sidecars, and paths.txt)"
        )
    if "REPLACE_ME" in uri:
        raise ValueError(
            f"data.tokens_s3 is still the placeholder {uri!r}; point it at the real "
            "pre-tokenized shard directory before running anything."
        )
    return uri


def s3_uri(
    cfg: Dict[str, Any],
    *parts: str,
    bucket_key: str = "dataset_bucket",
    prefix_key: str = "prefix",
) -> str:
    """Build an S3 URI for per-run outputs (metrics / checkpoints).

    Tokenized training inputs are *not* composed here — use ``resolve_tokens_s3``.
    """
    s3 = cfg["s3"]
    bucket = s3.get(bucket_key) or s3["dataset_bucket"]
    prefix = str(s3.get(prefix_key) or s3["prefix"]).rstrip("/")
    rest = "/".join(p.strip("/") for p in parts if p)
    if rest:
        return f"s3://{bucket}/{prefix}/{rest}"
    return f"s3://{bucket}/{prefix}"
