"""Best-effort ``aws s3 sync`` / ``cp`` for arm checkpoint / result exports.

Used by the YAML spine and standalone trainers after permanent saves. Upload
helpers never raise into the train loop — they log failures and return. Download
helpers used for ``--resume`` on ephemeral scratch *do* surface failures so a
resume cannot silently start from an empty save folder.

Disable live export with ``S3_EXPORT=0`` / ``SKIP_S3_UPLOAD=1``. Resume fetch
still runs unless ``S3_EXPORT=0`` (there is nothing durable to pull from).
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


def _aws_cmd(*args: str) -> list[str]:
    return ["aws", *args]


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
    cmd = _aws_cmd(
        "s3",
        "sync",
        str(local_p),
        remote,
        "--only-show-errors",
    )
    try:
        log.info("S3 export: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # noqa: BLE001 — never kill training
        log.warning("S3 export failed (%s → %s): %s", local_p, remote, exc)
        return False


def sync_from_s3(
    remote: str,
    local: Union[str, Path],
    *,
    enabled: Optional[bool] = None,
    raise_on_error: bool = False,
) -> bool:
    """Sync ``remote`` → ``local`` (trailing slash). Returns True on success.

    When ``raise_on_error`` is True (resume path), failures propagate so the
    caller can fail closed instead of training from an empty save folder.
    """
    if not s3_export_enabled(enabled):
        log.info("S3 download skipped (S3_EXPORT=0 or SKIP_S3_UPLOAD=1)")
        return False
    if shutil.which("aws") is None:
        msg = f"aws CLI not on PATH; cannot sync {remote} → {local}"
        if raise_on_error:
            raise RuntimeError(msg)
        log.warning("%s", msg)
        return False
    local_p = Path(local)
    local_p.mkdir(parents=True, exist_ok=True)
    remote = remote if remote.endswith("/") else remote + "/"
    cmd = _aws_cmd(
        "s3",
        "sync",
        remote,
        str(local_p),
        "--only-show-errors",
    )
    try:
        log.info("S3 download: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # noqa: BLE001
        if raise_on_error:
            raise RuntimeError(f"S3 download failed ({remote} → {local_p}): {exc}") from exc
        log.warning("S3 download failed (%s → %s): %s", remote, local_p, exc)
        return False


def cp_to_s3(
    local: Union[str, Path],
    remote: str,
    *,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload one file to an exact S3 object key. Returns True on success."""
    if not s3_export_enabled(enabled):
        return False
    if shutil.which("aws") is None:
        log.warning("aws CLI not on PATH; skip S3 cp to %s", remote)
        return False
    local_p = Path(local)
    if not local_p.is_file():
        log.warning("S3 cp skip: missing file %s", local_p)
        return False
    cmd = _aws_cmd("s3", "cp", str(local_p), remote, "--only-show-errors")
    try:
        log.info("S3 cp: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # noqa: BLE001 — never kill training
        log.warning("S3 cp failed (%s → %s): %s", local_p, remote, exc)
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


def export_arm_run_fingerprint(
    arm: str,
    fingerprint_path: Union[str, Path],
    *,
    method: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload ``run_fingerprint.json`` next to method checkpoints (durable resume id)."""
    parts: list[str] = ["checkpoints"]
    if method:
        parts.append(str(method).strip("/"))
    parts.append("run_fingerprint.json")
    return cp_to_s3(fingerprint_path, arm_uri(arm, *parts), enabled=enabled)


def export_arm_metrics_dir(
    arm: str,
    metrics_dir: Union[str, Path],
    *,
    method: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload metrics under ``token-sel/<arm>/metrics[/method]/``."""
    parts: list[str] = ["metrics"]
    if method:
        parts.append(str(method).strip("/"))
    return sync_to_s3(metrics_dir, arm_uri(arm, *parts), enabled=enabled)


def export_arm_task_loss_dir(
    arm: str,
    results_dir: Union[str, Path],
    *,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload task_loss JSON directory under ``token-sel/<arm>/task_loss_results/``."""
    return sync_to_s3(results_dir, arm_uri(arm, "task_loss_results"), enabled=enabled)


def fetch_arm_method_checkpoints(
    arm: str,
    local_method_dir: Union[str, Path],
    *,
    method: Optional[str] = None,
    enabled: Optional[bool] = None,
    raise_on_error: bool = True,
) -> bool:
    """Download durable method checkpoints (+ fingerprint) into ``local_method_dir``.

    Required for ``--resume`` on ephemeral FarmShare/AWS scratch that does not
    retain a prior job's save folder.
    """
    parts: list[str] = ["checkpoints"]
    if method:
        parts.append(str(method).strip("/"))
    return sync_from_s3(
        arm_uri(arm, *parts),
        local_method_dir,
        enabled=enabled,
        raise_on_error=raise_on_error,
    )


def fetch_arm_method_metrics(
    arm: str,
    local_metrics_dir: Union[str, Path],
    *,
    method: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Best-effort download of method metrics for ledger continuity on resume."""
    parts: list[str] = ["metrics"]
    if method:
        parts.append(str(method).strip("/"))
    return sync_from_s3(
        arm_uri(arm, *parts),
        local_metrics_dir,
        enabled=enabled,
        raise_on_error=False,
    )


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
