"""Best-effort ``aws s3 sync`` for arm checkpoint / result exports.

Used by standalone trainers after permanent saves. Never raises into the train
loop — logs failures and returns. Does not call AWS from developer workstations
unless the train host has credentials; disable with ``S3_EXPORT=0``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

from token_selection.olmo_ext.s3_layout import arm_uri

log = logging.getLogger("token_selection.s3_export")


_FALSEY = frozenset({"0", "false", "no", "off"})


def s3_export_enabled(explicit: Optional[bool] = None) -> bool:
    """Return whether live S3 uploads are allowed.

    Disabled when ``explicit`` is False, or when either env var is truthy-false::

        S3_EXPORT=0|false|no|off
        SKIP_S3_UPLOAD=1|true|yes|on
    """
    if explicit is not None:
        return bool(explicit)
    if os.environ.get("S3_EXPORT", "1").strip().lower() in _FALSEY:
        return False
    if os.environ.get("SKIP_S3_UPLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return True


def sync_to_s3(
    local: Union[str, Path],
    remote: str,
    *,
    enabled: Optional[bool] = None,
) -> bool:
    """Sync ``local`` → ``remote`` (trailing slash). Returns True on success."""
    if not s3_export_enabled(enabled):
        log.info("S3 export disabled (S3_EXPORT=0 or SKIP_S3_UPLOAD=1)")
        return False
    if shutil.which("aws") is None:
        log.warning("aws CLI not on PATH; skip S3 export to %s", remote)
        return False
    local_p = Path(local)
    if not local_p.exists():
        log.warning("S3 export skip: missing local path %s", local_p)
        return False
    remote = remote if remote.endswith("/") else remote + "/"
    cmd = [
        "aws",
        "s3",
        "sync",
        str(local_p),
        remote,
        "--only-show-errors",
    ]
    try:
        log.info("S3 export: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # noqa: BLE001 — never kill training
        log.warning("S3 export failed (%s → %s): %s", local_p, remote, exc)
        return False


def export_arm_checkpoint(
    arm: str,
    checkpoint_dir: Union[str, Path],
    *,
    method: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload one step directory under ``token-sel/<arm>/checkpoints[/method]/<step>/``."""
    ckpt = Path(checkpoint_dir)
    parts: list[str] = ["checkpoints"]
    if method:
        parts.append(str(method).strip("/"))
    parts.append(ckpt.name)
    remote = arm_uri(arm, *parts)
    return sync_to_s3(ckpt, remote, enabled=enabled)


def export_arm_task_loss_dir(
    arm: str,
    results_dir: Union[str, Path],
    *,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload task_loss JSON directory under ``token-sel/<arm>/task_loss_results/``."""
    return sync_to_s3(results_dir, arm_uri(arm, "task_loss_results"), enabled=enabled)


def export_arm_tree(
    arm: str,
    *,
    checkpoints_root: Optional[Union[str, Path]] = None,
    task_loss_dir: Optional[Union[str, Path]] = None,
    progress_dir: Optional[Union[str, Path]] = None,
    enabled: Optional[bool] = None,
) -> None:
    """Best-effort full arm export (checkpoints + task_loss + optional progress)."""
    if checkpoints_root is not None:
        sync_to_s3(checkpoints_root, arm_uri(arm, "checkpoints"), enabled=enabled)
    if task_loss_dir is not None:
        export_arm_task_loss_dir(arm, task_loss_dir, enabled=enabled)
    if progress_dir is not None:
        sync_to_s3(progress_dir, arm_uri(arm, "progress"), enabled=enabled)
