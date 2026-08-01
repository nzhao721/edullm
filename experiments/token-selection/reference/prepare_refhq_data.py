#!/usr/bin/env python3
"""Stage RefHQ train memmaps for the reference arm from ``s3://edullm-data``.

Resolves ``pretrain/refhq-regmix-5p5b`` via ``edullm_data.read.resolve_latest`` /
``dataset_paths`` and downloads train-split ``.u32le.bin`` shards into ``--work``
(idempotent: skips objects whose local size already matches S3).

Writes::

  <work>/tokenized/paths_train.txt
  <work>/length_tokens.txt
  <work>/refhq_data_summary.json

Requires ``edullm-data`` installed and AWS credentials that can read
``s3://edullm-data`` (``aws`` CLI preferred for multipart downloads). Does **not**
assume FarmShare scratch or laptop-local corpora already exist, and never reads
``s3://edullm-datasets/``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_DATASET_ID = "pretrain/refhq-regmix-5p5b"
DEFAULT_SPLIT = "train"
LEGACY_BUCKET = "edullm-datasets"


def _ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "edullm-data package is required. Install with:\n"
            '  uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"\n'
            "or: pip install 'edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0'"
        ) from exc


def _resolve_split(dataset_id: str, *, version: Optional[str], split: str):
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(f"No published version in edullm-data catalog for {dataset_id!r}")
    resolved = dataset_paths(dataset_id, ver, split=split, s3=s3, require_validated=True)
    if not resolved.paths:
        raise SystemExit(f"No objects for {dataset_id}/{ver} split={split!r}")
    return resolved


def _s3_uri_parts(uri: str) -> Tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise SystemExit(f"Expected s3:// URI, got {uri!r}")
    if p.netloc == LEGACY_BUCKET:
        raise SystemExit(
            f"Refusing legacy bucket {LEGACY_BUCKET!r} in {uri!r}; use s3://edullm-data/"
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
        import boto3

        boto3.client("s3").download_file(bucket, key, str(tmp))
    got = tmp.stat().st_size
    if got != remote_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Download size mismatch for {uri}: got {got}, expected {remote_size}"
        )
    tmp.replace(dest)


def stage_resolved(resolved: Any, stage_root: Path) -> dict:
    """Download ``resolved.paths`` under ``stage_root``; return local path info."""
    stage_root.mkdir(parents=True, exist_ok=True)
    local_paths: List[Path] = []
    prefix = f"{resolved.dataset_id}/{resolved.version}/"
    for uri in resolved.paths:
        _bucket, key = _s3_uri_parts(uri)
        if not key.startswith(prefix):
            raise SystemExit(f"Object key {key!r} not under {prefix!r}")
        rel = key[len(prefix) :]
        dest = stage_root / rel
        print(f"[refhq] stage {uri} -> {dest}", flush=True)
        _download_one(uri, dest)
        local_paths.append(dest)
    if not local_paths:
        raise SystemExit("No local shards staged for refhq")
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
    ap.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Ephemeral staging root (must be empty or a prior stage of this script)",
    )
    ap.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"edullm-data dataset id (default: {DEFAULT_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin version (default: resolve_latest)",
    )
    ap.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Partition to stage (default: train)",
    )
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=None,
        help="Token budget written to length_tokens.txt (default: published rows)",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Only rematerialize path lists if shards already staged under --work",
    )
    args = ap.parse_args()

    if LEGACY_BUCKET in str(args.work).replace("\\", "/"):
        raise SystemExit(
            f"Refusing --work under legacy {LEGACY_BUCKET!r}; use a clean scratch dir"
        )

    _ensure_edullm_data()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_split(
        args.dataset_id, version=args.dataset_version, split=args.split
    )
    stage_root = work / "tokenized" / "shards"

    if args.skip_download:
        prefix = f"{resolved.dataset_id}/{resolved.version}/"
        locals_: List[str] = []
        for uri in resolved.paths:
            _, key = _s3_uri_parts(uri)
            dest = stage_root / key[len(prefix) :]
            if not dest.is_file():
                raise SystemExit(
                    f"--skip-download but missing {dest}; re-run without --skip-download"
                )
            locals_.append(str(dest.resolve()))
        info = {
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
    else:
        info = stage_resolved(resolved, stage_root)

    paths_file = work / "tokenized" / "paths_train.txt"
    write_paths(info["local_paths"], paths_file)
    info["paths_file"] = str(paths_file.resolve())

    length = (
        int(args.length_tokens)
        if args.length_tokens is not None
        else int(info["rows"] or info["total_tokens_on_disk"])
    )
    (work / "length_tokens.txt").write_text(str(length) + "\n", encoding="utf-8")

    summary = {
        "refhq": info,
        "length_tokens": length,
        "note": "Ephemeral input stage only; durable checkpoints upload to W&B",
    }
    (work / "refhq_data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
