#!/usr/bin/env python3
"""Stage RegMix 10B train memmaps for the Control CE arm from edullm-data.

Resolves published ``pretrain/regmix-10b`` via ``edullm_data.read.resolve_latest`` /
``dataset_paths`` (validated) and downloads train-split ``.u32le.bin`` shards into
``--work`` (idempotent: skips objects whose local size already matches S3).

Writes::

  <work>/train_tokenized/paths_train.txt
  <work>/length_tokens.txt
  <work>/control_data_summary.json

Requires ``edullm-data`` installed and AWS credentials that can read
``s3://edullm-data``. Does **not** assume FarmShare scratch or laptop-local
corpora already exist, and never reads ``s3://edullm-datasets/``.

The trainer can also stage itself via ``--stage-dir``; this script is an optional
pre-stage for reuse across launches on the same ephemeral job directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_TRAIN_DATASET_ID = "pretrain/regmix-10b"
DEFAULT_SPLIT = "train"
DEFAULT_LENGTH_TOKENS = 9_900_000_000  # shared one-epoch matrix budget → 2360 steps
DATA_BUCKET = "edullm-data"
LEGACY_DATA_BUCKET = "edullm-datasets"


def _ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "edullm-data package is required. Install with:\n"
            '  uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"\n'
            "or: pip install 'edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0'"
        ) from exc


def _reject_legacy_uri(uri: str) -> None:
    if f"s3://{LEGACY_DATA_BUCKET}/" in uri or uri.startswith(f"{LEGACY_DATA_BUCKET}/"):
        raise SystemExit(
            f"Refusing legacy URI under s3://{LEGACY_DATA_BUCKET}/: {uri}. "
            f"Use published s3://{DATA_BUCKET}/ via --train-dataset-id."
        )


def _resolve_split(dataset_id: str, *, version: Optional[str], split: str):
    from edullm_data.read import NotValidated, ReadError, dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3, data_bucket=DATA_BUCKET)
    if not ver:
        raise SystemExit(
            f"No published version in s3://{DATA_BUCKET}/_catalog/ for {dataset_id!r}. "
            f"Do not use s3://{LEGACY_DATA_BUCKET}/."
        )
    try:
        resolved = dataset_paths(
            dataset_id,
            ver,
            split=split,
            s3=s3,
            data_bucket=DATA_BUCKET,
            require_validated=True,
        )
    except NotValidated as exc:
        raise SystemExit(
            f"{dataset_id}/{ver} has no _VALIDATED.json — refusing unvalidated data: {exc}"
        ) from exc
    except ReadError as exc:
        raise SystemExit(f"Cannot resolve {dataset_id}/{ver} split={split}: {exc}") from exc
    if not resolved.paths:
        raise SystemExit(f"No objects for {dataset_id}/{ver} split={split!r}")
    for uri in resolved.paths:
        _reject_legacy_uri(uri)
    if resolved.dtype and resolved.dtype != "uint32":
        raise SystemExit(
            f"{dataset_id}/{ver} dtype={resolved.dtype!r}; Control memmap loader expects uint32"
        )
    return resolved


def _s3_uri_parts(uri: str) -> Tuple[str, str]:
    _reject_legacy_uri(uri)
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise SystemExit(f"Expected s3:// URI, got {uri!r}")
    if p.netloc != DATA_BUCKET:
        raise SystemExit(f"Only s3://{DATA_BUCKET}/ staging is allowed, got: {uri}")
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
    if tmp.exists():
        tmp.unlink()
    if shutil.which("aws"):
        subprocess.run(
            ["aws", "s3", "cp", uri, str(tmp), "--only-show-errors"],
            check=True,
        )
    else:
        import boto3

        boto3.client("s3").download_file(bucket, key, str(tmp))
    got = tmp.stat().st_size
    if got != remote_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Download size mismatch for {uri}: got {got}, expected {remote_size}"
        )
    tmp.replace(dest)


def stage_resolved(resolved: Any, stage_root: Path, *, label: str) -> dict:
    """Download ``resolved.paths`` under ``stage_root``; return local path info."""
    stage_root.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    prefix = f"{resolved.dataset_id}/{resolved.version}/"
    for uri in resolved.paths:
        _, key = _s3_uri_parts(uri)
        if not key.startswith(prefix):
            raise SystemExit(f"Object key {key!r} not under {prefix!r}")
        rel = key[len(prefix) :]
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
        "uri": f"s3://{DATA_BUCKET}/{resolved.dataset_id}/{resolved.version}/",
    }


def write_paths(local_paths: List[str], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(local_paths) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Ephemeral staging + path-list output root (job scratch OK; not durable)",
    )
    ap.add_argument(
        "--train-dataset-id",
        default=DEFAULT_TRAIN_DATASET_ID,
        help=f"edullm-data dataset id (default: {DEFAULT_TRAIN_DATASET_ID})",
    )
    ap.add_argument(
        "--train-version",
        default=None,
        help="Pin version (default: resolve_latest)",
    )
    ap.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Partition to stage (default: {DEFAULT_SPLIT})",
    )
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=DEFAULT_LENGTH_TOKENS,
        help="Training token budget (default 9.9B → 2360 steps; one-epoch matrix)",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Only resolve catalog paths; require shards already staged under --work",
    )
    args = ap.parse_args()

    _ensure_edullm_data()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)

    train_resolved = _resolve_split(
        args.train_dataset_id, version=args.train_version, split=args.split
    )
    train_stage = work / "train_tokenized" / "shards"

    if args.skip_download:
        prefix = f"{train_resolved.dataset_id}/{train_resolved.version}/"
        locals_: List[str] = []
        for uri in train_resolved.paths:
            _, key = _s3_uri_parts(uri)
            dest = train_stage / key[len(prefix) :]
            if not dest.is_file():
                raise SystemExit(
                    f"--skip-download but missing {dest}; re-run without --skip-download"
                )
            locals_.append(str(dest.resolve()))
        train_info = {
            "n_files": len(locals_),
            "total_tokens_on_disk": sum(Path(p).stat().st_size // 4 for p in locals_),
            "local_paths": locals_,
            "s3_paths": list(train_resolved.paths),
            "dataset_id": train_resolved.dataset_id,
            "version": train_resolved.version,
            "split": train_resolved.split,
            "dtype": train_resolved.dtype or "uint32",
            "rows": train_resolved.rows,
            "uri": f"s3://{DATA_BUCKET}/{train_resolved.dataset_id}/{train_resolved.version}/",
        }
    else:
        train_info = stage_resolved(train_resolved, train_stage, label="train")

    train_paths_file = work / "train_tokenized" / "paths_train.txt"
    write_paths(train_info["local_paths"], train_paths_file)
    train_info["paths_file"] = str(train_paths_file.resolve())

    length = int(args.length_tokens)
    published_rows = train_info.get("rows") or train_info.get("total_tokens_on_disk")
    if published_rows is not None and length > int(published_rows):
        raise SystemExit(
            f"--length-tokens {length} exceeds published train rows {published_rows}; "
            "refusing to wrap past one epoch of pretrain/regmix-10b"
        )
    (work / "length_tokens.txt").write_text(str(length) + "\n", encoding="utf-8")

    summary = {
        "arm": "control",
        "source": "edullm-data",
        "bucket": DATA_BUCKET,
        "train": train_info,
        "length_tokens": length,
        "expected_steps": length // 4_194_304,
        "note": (
            "Control arm: uniform random 60% token keep (method=random). Staging under "
            "--work is ephemeral; durable checkpoints remain on scratch and upload "
            "to W&B via the trainer."
        ),
    }
    (work / "control_data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
