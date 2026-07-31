"""Stage published ``edullm-data`` train shards onto local/scratch for OLMo-core.

Resolution (dataset id / version / shard URIs) lives in
``token_selection.scripts.train_data_resolve``. This module only materializes
those URIs onto disk and writes the local train ``manifest.json`` /
``order/manifest.json`` that ``train_olmo_template`` validates. It never assumes
FarmShare scratch or laptop-local trees already exist.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from token_selection.scripts.train_data_resolve import resolve_train_dataset


def _rel_from_s3_uri(uri: str, *, tokens_prefix: str) -> str:
    prefix = tokens_prefix.rstrip("/") + "/"
    if uri.startswith(prefix):
        return uri[len(prefix) :]
    marker = "/tokens/"
    if marker not in uri:
        raise ValueError(f"Train shard URI {uri!r} is not under {tokens_prefix!r}")
    return uri.split(marker, 1)[1]


def _aws_sync(remote: str, local: Path, *, profile: str, excludes: Tuple[str, ...] = ()) -> None:
    remote = remote if remote.endswith("/") else remote + "/"
    local.mkdir(parents=True, exist_ok=True)
    cmd = ["aws"]
    if profile and str(profile).strip().lower() not in {"none", "null", "-"}:
        cmd.extend(["--profile", profile])
    cmd.extend(["s3", "sync", remote, str(local), "--only-show-errors"])
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _fill_shard_token_counts(manifest: Dict[str, Any], tokens_dir: Path) -> Dict[str, Any]:
    from token_selection.olmo_ext.token_io import TOKEN_DTYPE, count_tokens, dtype_from_name

    dtype = dtype_from_name(manifest["dtype"]) if manifest.get("dtype") else TOKEN_DTYPE
    total = 0
    shards = []
    for shard in manifest["shards"]:
        rel = str(shard["path"])
        path = tokens_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing staged train shard: {path}")
        n_tokens = count_tokens(path, dtype=dtype)
        shards.append({"path": rel, "n_tokens": n_tokens})
        total += n_tokens
    out = dict(manifest)
    out["shards"] = shards
    out["n_tokens"] = total
    return out


def ensure_train_tokens(
    cfg: Mapping[str, Any],
    tokens_dir: Path,
    *,
    profile: Optional[str] = None,
    force: bool = False,
    s3: Any = None,
) -> Dict[str, Any]:
    """Stage train shards from edullm-data into ``tokens_dir`` and write ``manifest.json``.

    Idempotent when a complete local train manifest already matches the resolved
    ``dataset_id`` / ``version`` and every listed shard is present.
    """
    tokens_dir = Path(tokens_dir)
    tokens_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_train_dataset(cfg, s3=s3, split="train")
    dataset_id = str(resolved["dataset_id"])
    version = str(resolved["version"])
    tokens_prefix = str(resolved["tokens_uri"]).rstrip("/")
    tokenizer = str((cfg.get("data") or {}).get("tokenizer") or "")
    if not tokenizer:
        raise ValueError("data.tokenizer is required")

    paths = [str(p) for p in (resolved["paths"] or [])]
    if not paths:
        raise ValueError(
            f"{dataset_id}/{version} train split resolved to zero shard paths; refusing to train"
        )

    manifest_path = tokens_dir / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        if (
            str(prior.get("dataset_id") or "") == dataset_id
            and str(prior.get("dataset_version") or "") == version
            and prior.get("shards")
        ):
            missing = [
                s["path"]
                for s in prior["shards"]
                if not (tokens_dir / str(s["path"])).is_file()
            ]
            if not missing:
                return prior

    s3_block = cfg.get("s3") or {}
    aws_profile = (
        profile
        if profile is not None
        else str(s3_block.get("profile") or os.environ.get("AWS_PROFILE") or "sbsandbox")
    )
    # Train shards only — exclude held-out val and the edullm group manifest so we can
    # write the train-local manifest shape experiment_contract expects.
    _aws_sync(
        tokens_prefix,
        tokens_dir,
        profile=aws_profile,
        excludes=("*/val-*", "val-*", "manifest.json"),
    )

    shards = []
    for uri in paths:
        rel = _rel_from_s3_uri(uri, tokens_prefix=tokens_prefix)
        if not (tokens_dir / rel).is_file():
            raise FileNotFoundError(
                f"Synced {tokens_prefix} into {tokens_dir} but missing train shard {rel}. "
                "Check AWS credentials and that the dataset is published under edullm-data."
            )
        shards.append({"path": rel})

    draft: Dict[str, Any] = {
        "n_tokens": 0,
        "dtype": str(resolved.get("dtype") or "uint32"),
        "tokenizer": tokenizer,
        "dataset_id": dataset_id,
        "dataset_version": version,
        "source_uri": tokens_prefix,
        "shards": shards,
    }
    manifest = _fill_shard_token_counts(draft, tokens_dir)
    published_rows = resolved.get("rows")
    if published_rows is not None and int(manifest["n_tokens"]) != int(published_rows):
        raise ValueError(
            f"Staged train token count {manifest['n_tokens']} != published rows "
            f"{published_rows} for {dataset_id}/{version}. Refusing to train on a "
            "partial/corrupt staging directory; clear tokens/ or pass force=True."
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def ensure_order_contract(cfg: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Build ``order/manifest.json`` when missing (clean-machine path)."""
    from token_selection.scripts.experiment_contract import (
        build_order_contract,
        validate_token_budget,
        validate_token_manifest,
    )

    output_dir = Path(output_dir)
    tokens_dir = output_dir / "tokens"
    order_dir = output_dir / "order"
    order_manifest_path = order_dir / "manifest.json"
    tokenizer = (cfg.get("data") or {}).get("tokenizer")
    manifest = validate_token_manifest(tokens_dir, expected_tokenizer=tokenizer)
    budget = validate_token_budget(cfg, manifest)
    if order_manifest_path.exists():
        return json.loads(order_manifest_path.read_text(encoding="utf-8"))

    order_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    seq_len = int(cfg["data"]["sequence_length"])
    n_tokens = int(manifest["n_tokens"])
    n_seqs = n_tokens // seq_len
    if n_seqs <= 0:
        raise ValueError(f"Not enough tokens ({n_tokens}) for sequence_length={seq_len}")
    order_contract = build_order_contract(cfg, output_dir=output_dir, token_manifest=manifest)
    meta = {
        "schema_version": 2,
        "seed": seed,
        "sequence_length": seq_len,
        "n_tokens": n_tokens,
        "n_sequences": n_seqs,
        "remainder_tokens": n_tokens - n_seqs * seq_len,
        "token_budget": budget,
        "order_contract": order_contract,
        "note": "Production OLMo-core uses the seeded global-index order in order_contract.",
    }
    order_manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
