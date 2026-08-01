#!/usr/bin/env python3
"""Shared Weights & Biases helpers for mixlaw trainers (SmolLM2 protocol).

Enablement mirrors ``scripts/farmshare/train_smollm2_135m_ddp.py``:
  * ``--wandb-mode online|offline|disabled`` (default online when API key present
    via launcher; trainers default to ``online`` CLI but no-op without key)
  * Requires ``wandb`` package + ``WANDB_API_KEY`` (FarmShare: ``wandb-session.env``)
  * Project default: ``mixlaw``
  * Rank-0 only; resume via ``wandb_run_id.txt`` under the progress/save root

Does not replace fail-closed local + W&B durable exports — W&B artifacts are the
off-scratch durability layer; checkpoints stay on runtime scratch during the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

log = logging.getLogger("mixlaw_wandb")

DEFAULT_WANDB_PROJECT = "mixlaw"

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

# Env keys curriculum / older trainers may wipe at import time.
_WANDB_ENV_KEYS = (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_NAME",
    "WANDB_GROUP",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_ENABLE",
    "WANDB_MODE",
    "WANDB_DISABLED",
    "WANDB_START_METHOD",
)


def snapshot_wandb_env() -> dict[str, str]:
    """Capture W&B-related env before imports that hard-disable / pop keys."""
    return {k: os.environ[k] for k in _WANDB_ENV_KEYS if k in os.environ}


def restore_wandb_env(snapshot: Mapping[str, str]) -> None:
    """Restore a prior snapshot (clears keys that were absent in the snapshot)."""
    for k in _WANDB_ENV_KEYS:
        if k in snapshot:
            os.environ[k] = snapshot[k]
        else:
            os.environ.pop(k, None)


def add_wandb_args(
    parser: argparse.ArgumentParser,
    *,
    default_project: str = DEFAULT_WANDB_PROJECT,
    default_mode: Optional[str] = None,
) -> None:
    mode = default_mode or os.environ.get("WANDB_MODE", "online")
    if mode not in ("online", "offline", "disabled"):
        mode = "online"
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", default_project))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_NAME") or None)
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP") or None)
    parser.add_argument(
        "--wandb-mode",
        default=mode,
        choices=("online", "offline", "disabled"),
        help="W&B logging/artifacts; online needs WANDB_API_KEY (wandb-session.env).",
    )
    parser.add_argument(
        "--wandb-upload-existing",
        action="store_true",
        help="On start, upload existing local checkpoints/evals as W&B artifacts.",
    )


def wandb_wanted(args: argparse.Namespace) -> bool:
    return getattr(args, "wandb_mode", "disabled") != "disabled"


def wandb_enabled(args: argparse.Namespace, *, is_main: bool = True) -> bool:
    return (
        is_main
        and wandb_wanted(args)
        and wandb is not None
        and bool(os.environ.get("WANDB_API_KEY"))
    )


def init_wandb(
    args: argparse.Namespace,
    run_meta: dict[str, Any],
    *,
    id_dir: Path,
    is_main: bool = True,
    tags: Optional[list[str]] = None,
    alert_title: str = "mixlaw train job started",
) -> Any:
    """Init a W&B run on rank 0; return the run handle or None."""
    if not wandb_wanted(args):
        return None
    if not is_main:
        return None
    if wandb is None:
        log.warning("wandb package missing; continuing without W&B")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY unset; continuing without W&B")
        return None

    os.environ.pop("WANDB_DISABLED", None)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_START_METHOD", "thread")

    id_dir = Path(id_dir)
    id_dir.mkdir(parents=True, exist_ok=True)
    id_path = id_dir / "wandb_run_id.txt"
    run_id = (
        id_path.read_text(encoding="utf-8").strip()
        if id_path.is_file()
        else (os.environ.get("WANDB_RUN_ID") or None)
    )
    run_name = args.wandb_run_name or getattr(args, "name", None) or getattr(args, "mix_name", None)

    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **run_meta,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=getattr(args, "wandb_group", None) or None,
        name=run_name,
        id=run_id,
        resume="allow" if run_id else None,
        config=config,
        dir=str(id_dir / "wandb"),
        tags=tags,
    )
    id_path.write_text(str(run.id), encoding="utf-8")
    log.info("wandb run=%s url=%s", run.id, run.url)
    try:
        run.alert(
            title=alert_title,
            text=(
                f"run={run.name} id={run.id} "
                f"slurm_job={os.environ.get('SLURM_JOB_ID', 'n/a')} "
                f"host={os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', 'n/a'))}"
            ),
            level=wandb.AlertLevel.INFO,
        )
    except Exception as exc:  # pragma: no cover - alert is best-effort
        log.debug("wandb alert skipped: %s", exc)
    return run


def wandb_log(run: Any, metrics: dict[str, Any], *, step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def wandb_log_eval(
    run: Any,
    payload: Mapping[str, Any],
    *,
    step: int,
    eval_path: Optional[Path] = None,
) -> None:
    """Log task-loss / ladder eval metrics (+ optional JSON artifact)."""
    if run is None:
        return
    # Shared ingestion recomputes macro from exactly the 20 raw BPB labels and
    # never blesses a partial/legacy suite as contract-complete.
    from token_selection.olmo_ext.wandb_logging import task_loss_metrics

    metrics = task_loss_metrics(payload)
    if metrics:
        wandb_log(run, metrics, step=step)
    if eval_path is not None and Path(eval_path).is_file():
        art = wandb.Artifact(name=f"eval-step{int(step):07d}", type="eval")
        art.add_file(str(eval_path), name=Path(eval_path).name)
        run.log_artifact(art)


def wandb_log_checkpoint(
    run: Any,
    ckpt_dir: Path,
    *,
    step: int,
    tokens_seen: int = 0,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> bool:
    if run is None:
        return False
    ckpt = Path(ckpt_dir)
    if not ckpt.is_dir():
        return False
    payload = {
        "checkpoint/step": step,
        "checkpoint/tokens_seen": tokens_seen,
    }
    if extra_meta:
        for k, v in extra_meta.items():
            if isinstance(v, (int, float)):
                payload[f"checkpoint/{k}"] = v
    wandb_log(run, payload, step=step)
    meta = {"step": step, "tokens_seen": tokens_seen}
    if extra_meta:
        meta.update(dict(extra_meta))
    art = wandb.Artifact(
        name=f"checkpoint-step{int(step):07d}",
        type="model",
        metadata=meta,
    )
    art.add_dir(str(ckpt))
    logged = run.log_artifact(art)
    wait = getattr(logged, "wait", None)
    if callable(wait):
        wait()
    log.info("wandb uploaded checkpoint artifact %s", art.name)
    return True


def wandb_upload_existing(
    run: Any,
    *,
    checkpoints_root: Optional[Path] = None,
    task_loss_dir: Optional[Path] = None,
    progress_dir: Optional[Path] = None,
    tokens_per_step: int = 0,
) -> None:
    """Best-effort upload of already-written local artifacts (resume / catch-up)."""
    if run is None:
        return
    if checkpoints_root is not None and Path(checkpoints_root).is_dir():
        for ckpt_dir in sorted(Path(checkpoints_root).glob("step*")):
            if not ckpt_dir.is_dir():
                continue
            try:
                step = int(ckpt_dir.name.replace("step", "").split("-")[0])
            except ValueError:
                continue
            tokens_seen = step * int(tokens_per_step) if tokens_per_step else 0
            wandb_log_checkpoint(run, ckpt_dir, step=step, tokens_seen=tokens_seen)
    if task_loss_dir is not None and Path(task_loss_dir).is_dir():
        for eval_path in sorted(Path(task_loss_dir).glob("step*_task_loss.json")):
            try:
                step = int(eval_path.name.split("_")[0].replace("step", ""))
            except ValueError:
                continue
            try:
                payload = json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            wandb_log_eval(run, payload, step=step, eval_path=eval_path)
    if progress_dir is not None:
        progress = Path(progress_dir)
        for name, art_type in (
            ("task_loss.jsonl", "metrics"),
            ("task_loss_final.json", "metrics"),
            ("train_loss.jsonl", "metrics"),
            ("run_meta.json", "config"),
        ):
            path = progress / name
            if path.is_file():
                art = wandb.Artifact(name=path.stem.replace("_", "-"), type=art_type)
                art.add_file(str(path), name=path.name)
                run.log_artifact(art)


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover
        log.warning("wandb.finish failed: %s", exc)


def build_olmo_wandb_config(args: argparse.Namespace):
    """Return ai2-olmo ``WandbConfig`` when enabled, else None."""
    if not wandb_enabled(args, is_main=True):
        if wandb_wanted(args):
            if wandb is None:
                log.warning("wandb package missing; TrainConfig.wandb=None")
            elif not os.environ.get("WANDB_API_KEY"):
                log.warning("WANDB_API_KEY unset; TrainConfig.wandb=None")
        return None
    from olmo import WandbConfig

    os.environ.pop("WANDB_DISABLED", None)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    name = args.wandb_run_name or getattr(args, "name", None)
    return WandbConfig(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=getattr(args, "wandb_group", None) or name,
        name=name,
        tags=["mixlaw", "datadecide-60m"],
        log_artifacts=True,
        rank_zero_only=True,
        log_interval=1,
    )
