#!/usr/bin/env python3
"""Derive ``tokens/manifest.json`` from the pre-tokenized corpus downloaded from S3.

The corpus is the ground truth and it does not ship a manifest. What it ships, under
``data.tokens_s3``, is one raw shard per domain plus a JSON sidecar describing it::

    paths.txt                                  # every shard, one relative path per line
    <domain>/<domain>.npy                      # raw headerless uint32 tokens
    <domain>/<domain>.json                     # {"tokenizer", "eos_token_id", "docs",
                                               #  "tokens_content", "tokens_with_eos",
                                               #  "bytes", "dtype", ...}

The rest of the pipeline wants the single manifest described by
``experiment_contract.TOKEN_MANIFEST_SCHEMA``, because the order contract fingerprints
that file to pin the training set. So this script translates one into the other and
verifies the corpus while it is at it: that ``paths.txt`` and the shards on disk are the
same set, that every shard's byte size matches the sidecar's ``bytes``, that the byte
size is a whole number of tokens in the declared dtype, and that all shards share a
dtype, tokenizer, and EOS id.

Output is written deterministically (sorted keys, shards sorted by path), so re-running
this on the same corpus reproduces the same bytes and therefore the same order-contract
fingerprint.

Note on ``tokens_with_eos`` vs ``tokens_content``: the shard on disk holds
``tokens_with_eos`` tokens, since each document is terminated by the EOS id. That is the
count the trainer sees, so it is the count the manifest records; ``tokens_content`` is
carried through per shard as provenance only.

Usage:
    python -m token_selection.scripts.build_token_manifest \
        --config token_selection/configs/run_rho_10b.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.olmo_ext.token_io import dtype_from_name, dtype_name
from token_selection.scripts import load_config, resolve_output_dir, resolve_tokens_s3

MANIFEST_NAME = "manifest.json"
PATHS_INDEX_NAME = "paths.txt"
SIDECAR_REQUIRED_KEYS = ("tokenizer", "eos_token_id", "tokens_with_eos", "bytes", "dtype")


def _rel(path: Path, tokens_dir: Path) -> str:
    return path.relative_to(tokens_dir).as_posix()


def _shards_on_disk(tokens_dir: Path) -> List[str]:
    return sorted(_rel(p, tokens_dir) for p in tokens_dir.rglob("*.npy"))


def read_paths_index(tokens_dir: Path) -> Optional[List[str]]:
    """Shard paths from ``paths.txt``, or None when the corpus has no index.

    The index in the bucket is written relative to the corpus root (``tokenized/dclm/
    dclm.npy``) while we sync the ``tokenized/`` prefix itself into ``tokens/``, so the
    leading component is dropped when that is what makes the path resolve.
    """
    index_path = tokens_dir / PATHS_INDEX_NAME
    if not index_path.exists():
        return None
    listed: List[str] = []
    for lineno, line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        entry = line.strip().replace("\\", "/").lstrip("/")
        if not entry:
            continue
        candidates = [entry]
        head, _, tail = entry.partition("/")
        if tail:
            candidates.append(tail)
        for candidate in candidates:
            if (tokens_dir / candidate).exists():
                listed.append(candidate)
                break
        else:
            raise SystemExit(
                f"{index_path}:{lineno}: lists {entry!r}, which is not present under "
                f"{tokens_dir} (tried {candidates}). Re-download the corpus with "
                "`sync_artifacts.py --direction download --what tokens`."
            )
    if not listed:
        raise SystemExit(f"{index_path} is empty; the corpus lists no shards.")
    return sorted(listed)


def _read_sidecar(shard: Path, tokens_dir: Path) -> Dict[str, Any]:
    sidecar = shard.with_suffix(".json")
    rel = _rel(shard, tokens_dir)
    if not sidecar.exists():
        raise SystemExit(
            f"{rel}: missing the sidecar {_rel(sidecar, tokens_dir)} that records its "
            "tokenizer, dtype and token count. Every shard in this corpus ships one; a "
            "shard without it is unverifiable, so it cannot enter the training set."
        )
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{_rel(sidecar, tokens_dir)}: not valid JSON ({exc})") from exc
    if not isinstance(meta, dict):
        raise SystemExit(f"{_rel(sidecar, tokens_dir)}: expected a JSON object")
    absent = [key for key in SIDECAR_REQUIRED_KEYS if key not in meta]
    if absent:
        raise SystemExit(
            f"{_rel(sidecar, tokens_dir)}: missing required key(s) {absent}. Expected the "
            f"corpus sidecar schema: {list(SIDECAR_REQUIRED_KEYS)}."
        )
    return meta


def build_manifest(
    tokens_dir: Path,
    *,
    source_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate the downloaded corpus and return the manifest to persist."""
    tokens_dir = Path(tokens_dir)
    if not tokens_dir.is_dir():
        raise SystemExit(
            f"{tokens_dir} does not exist. Download the corpus first:\n"
            "  python token_selection/scripts/sync_artifacts.py "
            "--direction download --what tokens"
        )

    on_disk = _shards_on_disk(tokens_dir)
    if not on_disk:
        raise SystemExit(f"No .npy shards found under {tokens_dir}.")
    indexed = read_paths_index(tokens_dir)
    if indexed is not None and indexed != on_disk:
        only_indexed = sorted(set(indexed) - set(on_disk))
        only_on_disk = sorted(set(on_disk) - set(indexed))
        raise SystemExit(
            f"{PATHS_INDEX_NAME} and the shards under {tokens_dir} disagree.\n"
            f"  listed but absent: {only_indexed}\n"
            f"  present but unlisted: {only_on_disk}\n"
            "The corpus index defines the training set; re-sync rather than guessing."
        )
    shard_paths = indexed if indexed is not None else on_disk

    shards: List[Dict[str, Any]] = []
    dtypes: Dict[str, str] = {}
    tokenizers: Dict[str, str] = {}
    eos_ids: Dict[str, int] = {}
    total_tokens = 0
    total_content_tokens = 0
    total_docs = 0

    for rel in shard_paths:
        shard = tokens_dir / rel
        meta = _read_sidecar(shard, tokens_dir)

        declared_dtype = str(meta["dtype"])
        try:
            dtype = dtype_from_name(declared_dtype)
        except ValueError as exc:
            raise SystemExit(f"{rel}: {exc}") from exc
        item_size = np.dtype(dtype).itemsize

        actual_bytes = os.path.getsize(shard)
        declared_bytes = int(meta["bytes"])
        if actual_bytes != declared_bytes:
            raise SystemExit(
                f"{rel}: sidecar says {declared_bytes} bytes but the file is "
                f"{actual_bytes}. The download is truncated or the shard was replaced; "
                "re-sync before training."
            )
        if actual_bytes % item_size:
            raise SystemExit(
                f"{rel}: {actual_bytes} bytes is not a whole number of "
                f"{dtype_name(dtype)} tokens. A shard written with np.save (128-byte "
                "header) is the usual cause; this corpus must be raw and headerless."
            )
        observed_tokens = actual_bytes // item_size
        declared_tokens = int(meta["tokens_with_eos"])
        if observed_tokens != declared_tokens:
            raise SystemExit(
                f"{rel}: sidecar claims {declared_tokens} tokens but the file holds "
                f"{observed_tokens} as {dtype_name(dtype)}."
            )

        dtypes[rel] = declared_dtype
        tokenizers[rel] = str(meta["tokenizer"])
        eos_ids[rel] = int(meta["eos_token_id"])
        total_tokens += observed_tokens
        total_content_tokens += int(meta.get("tokens_content", 0))
        total_docs += int(meta.get("docs", 0))
        shards.append(
            {
                "path": rel,
                "n_tokens": observed_tokens,
                "domain": str(meta.get("domain") or Path(rel).parent.name or Path(rel).stem),
                "docs": int(meta.get("docs", 0)),
                "tokens_content": int(meta.get("tokens_content", 0)),
            }
        )

    _require_uniform(dtypes, "dtype")
    _require_uniform(tokenizers, "tokenizer")
    _require_uniform(eos_ids, "eos_token_id")

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "n_tokens": total_tokens,
        "dtype": next(iter(dtypes.values())),
        "tokenizer": next(iter(tokenizers.values())),
        "eos_token_id": next(iter(eos_ids.values())),
        "n_docs": total_docs,
        "n_content_tokens": total_content_tokens,
        "shards": sorted(shards, key=lambda shard: shard["path"]),
    }
    if source_uri:
        manifest["source"] = source_uri
    return manifest


def _require_uniform(values: Mapping[str, Any], field: str) -> None:
    distinct = sorted({str(value) for value in values.values()})
    if len(distinct) > 1:
        detail = ", ".join(f"{path}={values[path]!r}" for path in sorted(values))
        raise SystemExit(
            f"Shards disagree on {field}: {distinct}. A corpus assembled from "
            f"inconsistently tokenized parts cannot be trained as one dataset. ({detail})"
        )


def write_manifest(tokens_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Persist the manifest deterministically so its fingerprint is reproducible."""
    manifest_path = Path(tokens_dir) / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_rho_10b.yaml")
    ap.add_argument(
        "--tokens-dir",
        type=Path,
        default=None,
        help="Override the corpus directory (defaults to <output_dir>/tokens).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    tokens_dir = args.tokens_dir or (out / "tokens")

    try:
        source_uri = resolve_tokens_s3(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    manifest = build_manifest(tokens_dir, source_uri=source_uri)
    manifest_path = write_manifest(tokens_dir, manifest)

    configured_tokenizer = str((cfg.get("data") or {}).get("tokenizer") or "")
    if configured_tokenizer and configured_tokenizer != manifest["tokenizer"]:
        raise SystemExit(
            f"Wrote {manifest_path}, but the corpus was tokenized with "
            f"{manifest['tokenizer']!r} while data.tokenizer is {configured_tokenizer!r}. "
            "The tokenizer determines the model's vocabulary size, so training with the "
            "configured value would index an embedding table the ids do not belong to. "
            "Set data.tokenizer to the corpus tokenizer."
        )

    print(
        json.dumps(
            {
                "status": "written",
                "manifest": str(manifest_path),
                "source": manifest.get("source"),
                "tokenizer": manifest["tokenizer"],
                "eos_token_id": manifest["eos_token_id"],
                "dtype": manifest["dtype"],
                "n_shards": len(manifest["shards"]),
                "n_tokens": manifest["n_tokens"],
                "domains": [shard["domain"] for shard in manifest["shards"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
