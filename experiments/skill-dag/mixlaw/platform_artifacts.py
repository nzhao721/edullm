#!/usr/bin/env python3
"""Fail-closed S3 publication helpers for the MixLaw platform runtime.

This module is imported only when the platform entrypoint supplies durable S3
prefixes. FarmShare and local launches continue to use their existing paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


class PlatformArtifactError(RuntimeError):
    """A platform artifact prefix or upload is invalid."""


def parse_s3_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(str(uri).strip())
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise PlatformArtifactError(f"expected non-root s3 prefix, got {uri!r}")
    return parsed.netloc, parsed.path.strip("/") + "/"


def join_s3_prefix(uri: str, *parts: str) -> str:
    bucket, prefix = parse_s3_prefix(uri)
    suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
    return f"s3://{bucket}/{prefix}{suffix}{'/' if suffix else ''}"


def _iter_files(root: Path, *, excluded_dirs: Iterable[str] = ()) -> list[Path]:
    excluded = set(excluded_dirs)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.endswith((".tmp", ".partial"))
        and not any(part in excluded for part in path.relative_to(root).parts[:-1])
    )


def _client(client: Any | None = None) -> Any:
    if client is not None:
        return client
    import boto3

    return boto3.client("s3")


def upload_tree(
    local_root: Path,
    destination: str,
    *,
    client: Any | None = None,
    excluded_dirs: Iterable[str] = (),
) -> list[str]:
    """Upload a local tree in sorted order and return the written S3 keys."""
    root = Path(local_root)
    if not root.is_dir():
        raise PlatformArtifactError(f"artifact directory does not exist: {root}")
    bucket, prefix = parse_s3_prefix(destination)
    s3 = _client(client)
    written: list[str] = []
    for path in _iter_files(root, excluded_dirs=excluded_dirs):
        relative = path.relative_to(root).as_posix()
        key = f"{prefix}{relative}"
        s3.upload_file(str(path), bucket, key)
        written.append(key)
    return written


def upload_checkpoint(
    checkpoint_dir: Path,
    destination: str,
    *,
    step: int,
    mix_name: str,
    client: Any | None = None,
) -> str:
    """Publish checkpoint contents first and a completion sentinel last."""
    root = Path(checkpoint_dir)
    s3 = _client(client)
    bucket, prefix = parse_s3_prefix(destination)
    written = upload_tree(root, destination, client=s3)
    if not written:
        raise PlatformArtifactError(f"checkpoint is empty: {root}")
    sentinel_key = f"{prefix}_COMPLETE.json"
    payload = {
        "schema_version": 1,
        "mix_name": mix_name,
        "step": int(step),
        "files": [key.removeprefix(prefix) for key in written],
    }
    s3.put_object(
        Bucket=bucket,
        Key=sentinel_key,
        Body=(json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{prefix}"


def upload_run_outputs(
    progress_dir: Path,
    task_loss_dir: Path,
    destination: str,
    *,
    client: Any | None = None,
) -> list[str]:
    """Publish progress and task-loss trees below one arm-specific prefix."""
    s3 = _client(client)
    written: list[str] = []
    progress = Path(progress_dir)
    if progress.is_dir():
        written.extend(
            upload_tree(
                progress,
                join_s3_prefix(destination, "progress"),
                client=s3,
                excluded_dirs=("wandb",),
            )
        )
    task_loss = Path(task_loss_dir)
    if task_loss.is_dir():
        written.extend(
            upload_tree(
                task_loss,
                join_s3_prefix(destination, "task-loss"),
                client=s3,
            )
        )
    return written
