"""Resolve published+validated train corpora from ``s3://edullm-data``.

Training arms must not assume FarmShare scratch, laptop-local trees, or the legacy
``s3://edullm-datasets/`` bucket. All trainable shard URIs come from
``edullm_data.read.resolve_latest`` / ``dataset_paths`` (require_validated=True).
Staging onto local/scratch is a separate step (``edullm_data_tokens.ensure_train_tokens``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple

DEFAULT_TRAIN_DATASET_ID = "pretrain/regmix-10b"
DATA_BUCKET = "edullm-data"

# Soft migration of legacy tokens_s3 URIs is intentionally omitted: callers must
# set data.dataset_id so resolution always goes through edullm_data.read.

_EDULLM_DATA_PREFIX_RE = re.compile(
    r"^s3://edullm-data/(?P<dataset_id>[^/]+/[^/]+)(?:/(?P<version>v\d+))?(?:/tokens)?/?$"
)


def _require_edullm_data() -> Tuple[Any, Any, Any]:
    try:
        from edullm_data.read import dataset_paths, resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The edullm-data package is required to resolve training corpora. "
            'Install with: uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0" '
            "(or pip install -e /path/to/edullm-data)."
        ) from exc
    return dataset_paths, resolve_latest, Boto3S3


def resolve_train_dataset_id(cfg: Mapping[str, Any]) -> str:
    """Return ``pretrain/<name>`` for the train corpus.

    Preference order:
      1. ``data.dataset_id`` / ``data.tokens_dataset_id``
      2. ``data.tokens_s3`` under ``s3://edullm-data/...`` (optional override)
    Legacy ``s3://edullm-datasets/`` URIs are refused.
    """
    data = cfg.get("data") or {}
    for key in ("dataset_id", "tokens_dataset_id"):
        raw = data.get(key)
        if raw is not None and str(raw).strip():
            dataset_id = str(raw).strip().strip("/")
            if dataset_id.count("/") != 1:
                raise ValueError(
                    f"data.{key}={dataset_id!r} must be '<family>/<name>' "
                    f"(e.g. {DEFAULT_TRAIN_DATASET_ID!r})"
                )
            if "edullm-datasets" in dataset_id:
                raise ValueError(
                    f"data.{key} must name an edullm-data dataset, not legacy edullm-datasets"
                )
            return dataset_id

    tokens_s3 = str(data.get("tokens_s3") or "").strip()
    if not tokens_s3:
        raise ValueError(
            "data.dataset_id is required (e.g. 'pretrain/regmix-10b'). "
            "Do not rely on FarmShare scratch, laptop-local paths, or "
            "s3://edullm-datasets/ as the source of truth."
        )
    if "REPLACE_ME" in tokens_s3:
        raise ValueError(
            f"data.tokens_s3 is still the placeholder {tokens_s3!r}; set data.dataset_id "
            f"to a published edullm-data id (e.g. {DEFAULT_TRAIN_DATASET_ID!r})."
        )

    match = _EDULLM_DATA_PREFIX_RE.match(tokens_s3.rstrip("/"))
    if match:
        return match.group("dataset_id")

    if "edullm-datasets" in tokens_s3:
        raise ValueError(
            f"data.tokens_s3={tokens_s3!r} points at legacy s3://edullm-datasets/. "
            f"Set data.dataset_id to a published edullm-data id "
            f"(default for this experiment: {DEFAULT_TRAIN_DATASET_ID!r})."
        )
    raise ValueError(
        f"Cannot derive an edullm-data dataset id from data.tokens_s3={tokens_s3!r}. "
        f"Set data.dataset_id (e.g. {DEFAULT_TRAIN_DATASET_ID!r})."
    )


def resolve_train_dataset_version(cfg: Mapping[str, Any], *, s3: Any = None) -> str:
    """Pin ``data.dataset_version`` when set; otherwise ``resolve_latest``."""
    data = cfg.get("data") or {}
    pinned = data.get("dataset_version")
    if pinned is not None and str(pinned).strip():
        ver = str(pinned).strip().strip("/")
        if not re.fullmatch(r"v\d+", ver):
            raise ValueError(f"data.dataset_version must look like 'v1', got {ver!r}")
        return ver

    tokens_s3 = str(data.get("tokens_s3") or "").strip()
    match = _EDULLM_DATA_PREFIX_RE.match(tokens_s3.rstrip("/")) if tokens_s3 else None
    if match and match.group("version"):
        return match.group("version")

    _, resolve_latest, Boto3S3 = _require_edullm_data()
    client = s3 if s3 is not None else Boto3S3.default()
    dataset_id = resolve_train_dataset_id(cfg)
    ver = resolve_latest(dataset_id, s3=client)
    if not ver:
        raise ValueError(
            f"No published versions found for {dataset_id!r} under "
            f"s3://{DATA_BUCKET}/_catalog/"
        )
    return str(ver)


def resolve_train_dataset(
    cfg: Mapping[str, Any],
    *,
    s3: Any = None,
    split: str = "train",
) -> Dict[str, Any]:
    """Resolve train corpus metadata from published+validated ``edullm-data``.

    Returns ``dataset_id``, ``version``, ``tokens_uri``, ``paths``, ``dtype``,
    ``numpy_dtype``, ``rows``, ``header_bytes``, ``byte_order``, and ``resolved``.
    """
    dataset_paths, _, Boto3S3 = _require_edullm_data()
    client = s3 if s3 is not None else Boto3S3.default()
    dataset_id = resolve_train_dataset_id(cfg)
    version = resolve_train_dataset_version(cfg, s3=client)
    resolved = dataset_paths(
        dataset_id,
        version,
        split=split,
        s3=client,
        group="tokens",
        require_validated=True,
    )
    paths = list(getattr(resolved, "paths", None) or [])
    tokens_uri = f"s3://{DATA_BUCKET}/{dataset_id}/{version}/tokens"
    if not paths:
        raise ValueError(
            f"{dataset_id}/{version} split={split!r} resolved to zero shard paths "
            f"under {tokens_uri}; refusing to train"
        )
    for uri in paths:
        if not (str(uri) == tokens_uri or str(uri).startswith(tokens_uri + "/")):
            raise ValueError(
                f"Resolved shard {uri!r} is outside the expected tokens root {tokens_uri!r}"
            )
    return {
        "dataset_id": dataset_id,
        "version": version,
        "tokens_uri": tokens_uri,
        "paths": paths,
        "dtype": getattr(resolved, "dtype", None),
        "numpy_dtype": getattr(resolved, "numpy_dtype", None),
        "rows": getattr(resolved, "rows", None),
        "header_bytes": int(getattr(resolved, "header_bytes", 0) or 0),
        "byte_order": getattr(resolved, "byte_order", None),
        "resolved": resolved,
    }


def resolve_tokens_s3(cfg: Mapping[str, Any], *, s3: Any = None) -> str:
    """Return the validated ``s3://edullm-data/<id>/<ver>/tokens`` prefix (no trailing /)."""
    return str(resolve_train_dataset(cfg, s3=s3)["tokens_uri"])


def resolve_train_split(cfg: Mapping[str, Any], *, s3: Any = None) -> Any:
    """Return the ``ResolvedSplit`` for the configured train corpus."""
    return resolve_train_dataset(cfg, s3=s3)["resolved"]
