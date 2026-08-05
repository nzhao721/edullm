#!/usr/bin/env python3
"""Stage RegMix train + RefHQ validation memmaps for BLADE 370M from edullm-data.

Resolves published datasets via ``edullm_data.read.resolve_latest`` /
``dataset_paths`` and downloads train-split ``.u32le.bin`` shards into
``--work`` (idempotent: skips objects whose local size already matches S3).

Intended for **ephemeral** job scratch: ``--work`` starts empty (or is created),
is filled from ``s3://edullm-data``, and must not be assumed to survive the job.
Never reads ``s3://edullm-datasets/``, FarmShare persistent trees, or laptop paths.

Defaults (override with flags)::

  train → pretrain/regmix-10b   (latest validated)
  ref   → pretrain/refhq-instruct/v3  (pinned validated release)

Writes::

  <work>/train_tokenized/paths_train.txt
  <work>/ref_tokenized/paths_refhq.txt
  <work>/length_tokens.txt
  <work>/blade_data_summary.json

Requires ``edullm-data`` installed and AWS credentials that can read
``s3://edullm-data`` (and ``aws`` CLI for multipart downloads).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_LENGTH_TOKENS = 9_900_000_000  # shared one-epoch matrix → 2360 steps
DEFAULT_TRAIN_DATASET_ID = "pretrain/regmix-10b"
DEFAULT_REF_DATASET_ID = "pretrain/refhq-instruct"
DEFAULT_REF_VERSION = "v3"
DEFAULT_SPLIT = "train"
LEGACY_DATA_BUCKET = "edullm-datasets"
DATA_BUCKET = "edullm-data"


def _ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "edullm-data package is required. Install with:\n"
            '  uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@main"\n'
            "or: pip install --upgrade 'edullm-data @ git+https://github.com/edu-llm/edullm-data@main'"
        ) from exc


def _resolve_split(dataset_id: str, *, version: Optional[str], split: str):
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(
            f"No published version in edullm-data catalog for {dataset_id!r}. "
            f"Do not use s3://{LEGACY_DATA_BUCKET}/ or pre-staged FarmShare/laptop paths."
        )
    resolved = dataset_paths(dataset_id, ver, split=split, s3=s3)
    if not resolved.paths:
        raise SystemExit(f"No objects for {dataset_id}/{ver} split={split!r}")
    for uri in resolved.paths:
        if LEGACY_DATA_BUCKET in uri:
            raise SystemExit(
                f"Refusing legacy bucket URI from catalog: {uri}\n"
                f"Expected objects under s3://{DATA_BUCKET}/"
            )
        if not uri.startswith(f"s3://{DATA_BUCKET}/"):
            raise SystemExit(
                f"Expected s3://{DATA_BUCKET}/ URI from dataset_paths, got {uri!r}"
            )
    if resolved.dtype and resolved.dtype != "uint32":
        raise SystemExit(
            f"{dataset_id}/{ver} dtype={resolved.dtype!r}; BLADE memmap loader expects uint32"
        )
    return resolved


def _s3_uri_parts(uri: str) -> Tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise SystemExit(f"Expected s3:// URI, got {uri!r}")
    if p.netloc == LEGACY_DATA_BUCKET:
        raise SystemExit(
            f"Refusing legacy bucket {LEGACY_DATA_BUCKET!r}; use {DATA_BUCKET!r}"
        )
    return p.netloc, p.path.lstrip("/")


def _head_size(bucket: str, key: str) -> int:
    from edullm_data.s3 import Boto3S3

    return int(Boto3S3.default().head(bucket, key)["size"])


def _download_one(uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = _s3_uri_parts(uri)
    remote_size = _head_size(bucket, key)
    if dest.is_file() and dest.stat().st_size == remote_size:
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if shutil.which("aws"):
        subprocess.run(
            ["aws", "s3", "cp", uri, str(tmp), "--only-show-errors"],
            check=True,
        )
    else:
        # Streaming download via boto3 (aws CLI preferred for large shards).
        import boto3

        boto3.client("s3").download_file(bucket, key, str(tmp))
    got = tmp.stat().st_size
    if got != remote_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Download size mismatch for {uri}: got {got}, expected {remote_size}"
        )
    tmp.replace(dest)


def stage_resolved(
    resolved: Any,
    stage_root: Path,
    *,
    label: str,
) -> dict:
    """Download ``resolved.paths`` under ``stage_root``; return local path info."""
    stage_root.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    prefix = f"{resolved.dataset_id}/{resolved.version}/"
    for uri in resolved.paths:
        bucket, key = _s3_uri_parts(uri)
        if not key.startswith(prefix):
            raise SystemExit(f"Object key {key!r} not under {prefix!r}")
        rel = key[len(prefix) :]  # e.g. tokens/dclm/train-00000.u32le.bin
        dest = stage_root / rel
        print(f"[{label}] stage {uri} -> {dest}", flush=True)
        _download_one(uri, dest)
        local_paths.append(dest)
    if not local_paths:
        raise SystemExit(f"No local shards staged for {label}")
    return {
        "n_files": len(local_paths),
        "total_tokens_on_disk": sum(p.stat().st_size // 4 for p in local_paths),
        "local_paths": [str(p.resolve()) for p in local_paths],
        "s3_paths": list(resolved.paths),
        "dataset_id": resolved.dataset_id,
        "version": resolved.version,
        "split": resolved.split,
        "dtype": resolved.dtype or "uint32",
        "rows": resolved.rows,
        "uri": f"s3://edullm-data/{resolved.dataset_id}/{resolved.version}/",
    }


def write_paths(local_paths: List[str], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(local_paths) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True, help="Staging + path-list output root")
    ap.add_argument(
        "--train-dataset-id",
        default=DEFAULT_TRAIN_DATASET_ID,
        help=f"edullm-data dataset id for proxy train (default: {DEFAULT_TRAIN_DATASET_ID})",
    )
    ap.add_argument(
        "--ref-dataset-id",
        default=DEFAULT_REF_DATASET_ID,
        help=f"edullm-data dataset id for RefHQ K-updates (default: {DEFAULT_REF_DATASET_ID})",
    )
    ap.add_argument("--train-version", default=None, help="Pin train version (default: resolve_latest)")
    ap.add_argument(
        "--ref-version",
        default=DEFAULT_REF_VERSION,
        help=f"Pin reference-update version (default: {DEFAULT_REF_VERSION})",
    )
    ap.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Partition name to stage for both corpora (default: train)",
    )
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=None,
        help=(
            "Proxy training token budget (default: "
            f"{DEFAULT_LENGTH_TOKENS} one-epoch matrix; "
            "pass published train rows to use resolve_latest rows instead)"
        ),
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Same-job only: rewrite path lists if shards are already under --work "
        "(after prepare in this job). Does not assume persistent scratch.",
    )
    args = ap.parse_args()

    _ensure_edullm_data()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)

    train_resolved = _resolve_split(
        args.train_dataset_id, version=args.train_version, split=args.split
    )
    ref_resolved = _resolve_split(
        args.ref_dataset_id, version=args.ref_version, split=args.split
    )

    train_stage = work / "train_tokenized" / "shards"
    ref_stage = work / "ref_tokenized" / "shards"

    if args.skip_download:
        # Re-map S3 URIs to expected local destinations; require all present.
        def _map(resolved: Any, root: Path) -> dict:
            prefix = f"{resolved.dataset_id}/{resolved.version}/"
            locals_: List[str] = []
            for uri in resolved.paths:
                _, key = _s3_uri_parts(uri)
                dest = root / key[len(prefix) :]
                if not dest.is_file():
                    raise SystemExit(
                        f"--skip-download but missing {dest}; re-run without --skip-download"
                    )
                locals_.append(str(dest.resolve()))
            return {
                "n_files": len(locals_),
                "total_tokens_on_disk": sum(Path(p).stat().st_size // 4 for p in locals_),
                "local_paths": locals_,
                "s3_paths": list(resolved.paths),
                "dataset_id": resolved.dataset_id,
                "version": resolved.version,
                "split": resolved.split,
                "dtype": resolved.dtype or "uint32",
                "rows": resolved.rows,
                "uri": f"s3://edullm-data/{resolved.dataset_id}/{resolved.version}/",
            }

        train_info = _map(train_resolved, train_stage)
        ref_info = _map(ref_resolved, ref_stage)
    else:
        train_info = stage_resolved(train_resolved, train_stage, label="train")
        ref_info = stage_resolved(ref_resolved, ref_stage, label="refhq")

    train_paths_file = work / "train_tokenized" / "paths_train.txt"
    ref_paths_file = work / "ref_tokenized" / "paths_refhq.txt"
    write_paths(train_info["local_paths"], train_paths_file)
    write_paths(ref_info["local_paths"], ref_paths_file)
    train_info["paths_file"] = str(train_paths_file.resolve())
    ref_info["paths_file"] = str(ref_paths_file.resolve())

    length = int(args.length_tokens) if args.length_tokens is not None else DEFAULT_LENGTH_TOKENS
    published_rows = train_info.get("rows") or train_info.get("total_tokens_on_disk")
    if published_rows is not None and length > int(published_rows):
        raise SystemExit(
            f"--length-tokens {length} exceeds published train rows {published_rows}; "
            "refusing to wrap past one epoch of pretrain/regmix-10b"
        )
    (work / "length_tokens.txt").write_text(str(length) + "\n", encoding="utf-8")

    summary = {
        "train": train_info,
        "refhq": ref_info,
        "length_tokens": length,
        "blade_schedule": {
            "tau": 375,
            "K": 75,
            "gamma": 0.6,
            "lambda_pen": 1.0,
            "blade_start": 500,
            "sync_steps": [500, 875, 1250, 1625, 2000],
        },
        "source": "edullm-data",
    }
    (work / "blade_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
