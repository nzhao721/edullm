"""Shared permanent-checkpoint contract for token-selection trainers.

Production ordering is deliberately strict:

1. the complete checkpoint already exists locally;
2. the full task-loss suite finishes and is validated;
3. checkpoint, progress, and eval artifacts are uploaded to W&B; and
4. ``last_durable_step.json`` advances locally only after required uploads succeed.

S3 is input-only. Local smoke runs may disable W&B and/or task-loss evaluation;
production online checkpoint uploads are fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

RUN_FINGERPRINT_FILENAME = "run_fingerprint.json"
FINGERPRINT_SCHEMA_VERSION = 2


class CheckpointContractError(RuntimeError):
    """A permanent checkpoint failed evaluation, export, or resume validation."""


def make_run_fingerprint(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical, self-checking scientific resume fingerprint."""
    canonical = json.loads(json.dumps(dict(identity), sort_keys=True))
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "identity": canonical,
        "identity_sha256": digest,
    }


def write_run_fingerprint(
    save_folder: str | Path, identity: Mapping[str, Any]
) -> Path:
    """Atomically persist the current run identity beside permanent steps."""
    root = Path(save_folder)
    root.mkdir(parents=True, exist_ok=True)
    target = root / RUN_FINGERPRINT_FILENAME
    payload = make_run_fingerprint(identity)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def read_run_fingerprint(path: str | Path) -> dict[str, Any]:
    """Read and verify a schema-v2 fingerprint file."""
    p = Path(path)
    if p.is_dir():
        p = p / RUN_FINGERPRINT_FILENAME
    if not p.is_file():
        raise CheckpointContractError(f"missing {RUN_FINGERPRINT_FILENAME}: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise CheckpointContractError(
            f"out-of-contract legacy fingerprint (expected schema 2): {p}"
        )
    expected = make_run_fingerprint(payload.get("identity") or {})
    if payload != expected:
        raise CheckpointContractError(f"invalid or corrupted run fingerprint: {p}")
    return payload


def assert_resume_fingerprint(
    checkpoint_dir: str | Path, identity: Mapping[str, Any]
) -> None:
    """Refuse legacy or scientifically mismatched standalone checkpoints."""
    checkpoint = Path(checkpoint_dir)
    candidates = (
        checkpoint / RUN_FINGERPRINT_FILENAME,
        checkpoint.parent / RUN_FINGERPRINT_FILENAME,
    )
    prior_path = next((p for p in candidates if p.is_file()), None)
    if prior_path is None:
        raise CheckpointContractError(
            f"{checkpoint} is an out-of-contract legacy artifact: no "
            f"{RUN_FINGERPRINT_FILENAME}. Start fresh; do not silently import it."
        )
    prior = read_run_fingerprint(prior_path)
    current = make_run_fingerprint(identity)
    if prior != current:
        old = prior.get("identity") or {}
        new = current["identity"]
        diffs = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        raise CheckpointContractError(
            "refusing resume with changed scientific identity "
            f"(differing fields: {diffs})"
        )


def copy_fingerprint_into_checkpoint(
    fingerprint_path: str | Path, checkpoint_dir: str | Path
) -> Path:
    """Copy the validated root fingerprint into a self-contained step directory."""
    src = Path(fingerprint_path)
    read_run_fingerprint(src)
    dst = Path(checkpoint_dir) / RUN_FINGERPRINT_FILENAME
    dst.write_bytes(src.read_bytes())
    return dst


def _assert_materialized(checkpoint_dir: Path) -> None:
    ready = (
        (checkpoint_dir / "state.pt").is_file()
        or (checkpoint_dir / "model_eval.pt").is_file()
        or (checkpoint_dir / "model_and_optim" / ".metadata").is_file()
    )
    if not ready:
        raise CheckpointContractError(
            f"permanent checkpoint is not fully materialized: {checkpoint_dir}"
        )


def validate_task_loss_result(path: str | Path) -> dict[str, Any]:
    """Require the current 20-label task-loss contract, rejecting partial/legacy JSON."""
    from token_selection.olmo_ext.wandb_logging import task_loss_payload_complete

    result = Path(path)
    if not result.is_file():
        raise CheckpointContractError(f"task-loss result was not written: {result}")
    payload = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not task_loss_payload_complete(payload):
        raise CheckpointContractError(
            f"out-of-contract partial/legacy task-loss result: {result}"
        )
    return payload


def finalize_permanent_checkpoint(
    *,
    arm: str,
    checkpoint_dir: str | Path,
    step: int,
    run_name: str,
    task_loss_dir: str | Path,
    task_loss_enabled: bool,
    task_loss_eval_script: Optional[str | Path] = None,
    task_loss_nproc: Optional[int] = None,
    progress_dir: Optional[str | Path] = None,
    fingerprint_path: Optional[str | Path] = None,
    method: Optional[str] = None,
    wandb_run: Any | None = None,
    wandb_mode: Optional[str] = None,
    production: bool = False,
) -> Optional[dict[str, Any]]:
    """Evaluate and upload one already-materialized permanent checkpoint.

    This function is rank-0 work. Distributed callers must keep every other rank
    paused and broadcast any exception before resuming training.
    """
    from token_selection.olmo_ext.durability import write_last_durable_step
    from token_selection.olmo_ext.task_loss_hook import trigger_task_loss_eval
    from token_selection.olmo_ext.wandb_logging import (
        checkpoint_artifact_ref,
        production_online,
        require_wandb_for_production,
        wandb_log_checkpoint,
        wandb_log_directory_artifact,
        wandb_log_eval,
    )

    checkpoint = Path(checkpoint_dir)
    _assert_materialized(checkpoint)
    out = Path(task_loss_dir) / f"step{int(step)}_task_loss.json"
    payload: Optional[dict[str, Any]] = None
    strict_upload = production_online(production=production, mode=wandb_mode)
    require_wandb_for_production(
        wandb_run, production=production, mode=wandb_mode
    )
    if strict_upload and not task_loss_enabled:
        raise CheckpointContractError(
            "refusing to mark a production checkpoint durable without the complete "
            "task-loss suite"
        )
    if task_loss_enabled:
        trigger_task_loss_eval(
            checkpoint,
            run_name=run_name,
            out_path=out,
            eval_script=task_loss_eval_script,
            async_=False,
            nproc=task_loss_nproc,
            strict=True,
        )
        payload = validate_task_loss_result(out)

    if fingerprint_path is not None:
        copy_fingerprint_into_checkpoint(fingerprint_path, checkpoint)

    artifact_name = checkpoint_artifact_ref(
        run_name=run_name,
        project=str(getattr(wandb_run, "project", "") or "token-selection"),
        entity=str(getattr(wandb_run, "entity", "") or "") or None,
        alias=f"step-{int(step):07d}",
    )
    wandb_log_checkpoint(
        wandb_run,
        checkpoint,
        step=int(step),
        extra_meta={"arm": arm, "method": method},
        strict=strict_upload,
        run_name=run_name,
    )
    if payload is not None:
        wandb_log_eval(
            wandb_run,
            payload,
            step=int(step),
            eval_path=out,
            strict=strict_upload,
        )
    if progress_dir is not None:
        wandb_log_directory_artifact(
            wandb_run,
            progress_dir,
            name=f"{run_name}-progress",
            artifact_type="metrics",
            strict=strict_upload,
        )
    if task_loss_enabled:
        wandb_log_directory_artifact(
            wandb_run,
            task_loss_dir,
            name=f"{run_name}-task-loss",
            artifact_type="eval",
            strict=strict_upload,
        )

    marker_dir = Path(progress_dir) if progress_dir is not None else checkpoint.parent
    write_last_durable_step(
        marker_dir,
        int(step),
        checkpoint_artifact=artifact_name if wandb_run is not None else None,
        extra={
            "run_name": str(run_name),
            "task_loss_complete": bool(task_loss_enabled),
            "task_loss_result": str(out) if task_loss_enabled else None,
            "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        },
    )
    return payload
