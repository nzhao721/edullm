"""Shared W&B helpers for Skill-It 370M train + probe final-eval logging.

Mirrors ``scripts/farmshare/train_smollm2_135m_ddp.py`` enablement:
  - online/offline/disabled via ``--wandb-mode`` / ``WANDB_MODE``
  - requires ``WANDB_API_KEY`` (typically from ``wandb-session.env`` on FarmShare)
  - soft-skip when package or key is missing
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

log = logging.getLogger("skillit.wandb")

DEFAULT_WANDB_PROJECT = "skillit"


def wandb_enabled(args: argparse.Namespace, *, is_main: bool) -> bool:
    return (
        is_main
        and getattr(args, "wandb_mode", "disabled") != "disabled"
        and wandb is not None
        and bool(os.environ.get("WANDB_API_KEY"))
    )


def init_wandb(
    args: argparse.Namespace,
    run_meta: Mapping[str, Any],
    *,
    is_main: bool,
    output_dir: Path,
    default_name: str,
) -> object | None:
    """Init a W&B run on the main process; persist run id for resume."""
    if not wandb_enabled(args, is_main=is_main):
        if is_main and getattr(args, "wandb_mode", "disabled") != "disabled" and wandb is None:
            log.warning("wandb package missing; continuing without W&B")
        elif (
            is_main
            and getattr(args, "wandb_mode", "disabled") != "disabled"
            and not os.environ.get("WANDB_API_KEY")
        ):
            log.warning("WANDB_API_KEY unset; continuing without W&B")
        return None
    assert wandb is not None
    os.environ.setdefault("WANDB_MODE", str(args.wandb_mode))
    id_path = output_dir / "wandb_run_id.txt"
    run_id = id_path.read_text(encoding="utf-8").strip() if id_path.is_file() else None
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in run_meta.items()},
    }
    run = wandb.init(
        project=str(args.wandb_project),
        entity=getattr(args, "wandb_entity", None) or None,
        name=getattr(args, "wandb_run_name", None) or default_name,
        group=getattr(args, "wandb_group", None) or None,
        id=run_id,
        resume="allow" if run_id else None,
        config=config,
        dir=str(wandb_dir),
    )
    id_path.write_text(str(run.id), encoding="utf-8")
    log.info("wandb run=%s url=%s", run.id, run.url)
    run.alert(
        title="skillit job started",
        text=(
            f"run={run.name} id={run.id} "
            f"slurm_job={os.environ.get('SLURM_JOB_ID', 'n/a')} "
            f"host={os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', 'n/a'))}"
        ),
        level=wandb.AlertLevel.INFO,
    )
    return run


def wandb_log(run: object | None, metrics: Mapping[str, Any], *, step: int) -> None:
    if run is None:
        return
    run.log(dict(metrics), step=int(step))


def wandb_log_train(
    run: object | None,
    *,
    step: int,
    tok_per_s: float,
    tok_per_s_avg: float,
    tokens_seen: int,
    weights: Mapping[str, float],
    a_mode: str,
) -> None:
    metrics: dict[str, Any] = {
        "train/tok_per_s": float(tok_per_s),
        "train/tok_per_s_avg": float(tok_per_s_avg),
        "train/tokens_seen": int(tokens_seen),
        "train/a_mode": a_mode,
    }
    for domain, value in weights.items():
        metrics[f"train/weight/{domain}"] = float(value)
    wandb_log(run, metrics, step=step)


def wandb_log_eval(
    run: object | None,
    payload: Mapping[str, Any],
    *,
    step: int,
    eval_path: Path,
    prefix: str = "eval",
) -> None:
    if run is None:
        return
    metrics: dict[str, float] = {}
    if "macro_mean" in payload:
        metrics[f"{prefix}/macro_bpb"] = float(payload["macro_mean"])
    if "macro_mean_accuracy" in payload:
        metrics[f"{prefix}/macro_acc"] = float(payload["macro_mean_accuracy"])
    label_src = (
        payload.get("labels")
        or payload.get("task_loss_bpb")
        or payload.get("task_loss_labels")
        or {}
    )
    for key, value in label_src.items():
        metrics[f"{prefix}/bpb/{key}"] = float(value)
    for key, value in (payload.get("accuracy_labels") or {}).items():
        metrics[f"{prefix}/acc/{key}"] = float(value)
    fam_src = payload.get("task_families") or payload.get("task_loss_families") or {}
    for key, value in fam_src.items():
        metrics[f"{prefix}/family_bpb/{key}"] = float(value)
    for key, value in (payload.get("accuracy_families") or {}).items():
        metrics[f"{prefix}/family_acc/{key}"] = float(value)
    if metrics:
        wandb_log(run, metrics, step=step)
    assert wandb is not None
    art = wandb.Artifact(name=f"eval-step{int(step):07d}", type="eval")
    art.add_file(str(eval_path), name=eval_path.name)
    run.log_artifact(art)


def _flatten_skillit_A(record: Mapping[str, Any]) -> dict[str, float]:
    """Expand the Skill-It adjacency matrix into per-entry W&B metrics.

    Prefers named ``rows_by_domain`` when present; otherwise uses the 2-D ``A``
    list with ``domain_order`` × ``family_order``.
    """
    out: dict[str, float] = {}
    rows = record.get("rows_by_domain")
    if isinstance(rows, Mapping):
        for domain, fam_map in rows.items():
            if not isinstance(fam_map, Mapping):
                continue
            for fam, value in fam_map.items():
                out[f"skillit/A/{domain}/{fam}"] = float(value)
        return out
    A = record.get("A")
    domains = list(record.get("domain_order") or [])
    families = list(record.get("family_order") or [])
    if not isinstance(A, list) or not domains or not families:
        return out
    for i, domain in enumerate(domains):
        if i >= len(A):
            break
        row = A[i]
        if not isinstance(row, list):
            continue
        for j, fam in enumerate(families):
            if j >= len(row):
                break
            out[f"skillit/A/{domain}/{fam}"] = float(row[j])
    return out


def wandb_log_skillit_update(
    run: object | None,
    record: Mapping[str, Any],
    *,
    step: int,
    snapshot_path: Optional[Path] = None,
    a_snapshot_path: Optional[Path] = None,
) -> None:
    if run is None:
        return
    metrics: dict[str, Any] = {
        "skillit/eta": float(record.get("eta", 0.0)),
        "skillit/w": float(record.get("w", 1.0)),
        "skillit/a_mode": record.get("a_mode"),
    }
    for domain, value in (record.get("p_before") or {}).items():
        metrics[f"skillit/weight_before/{domain}"] = float(value)
    for domain, value in (record.get("p_after") or {}).items():
        metrics[f"skillit/weight/{domain}"] = float(value)
    for fam, value in (record.get("losses") or {}).items():
        metrics[f"skillit/curve_loss/{fam}"] = float(value)
    metrics.update(_flatten_skillit_A(record))
    wandb_log(run, metrics, step=step)
    artifact_paths: list[Path] = []
    if snapshot_path is not None and snapshot_path.is_file():
        artifact_paths.append(Path(snapshot_path))
    if a_snapshot_path is not None and a_snapshot_path.is_file():
        artifact_paths.append(Path(a_snapshot_path))
    elif snapshot_path is not None:
        # Sibling written by write_skillit_snapshot: step{N}_A.json
        sibling = Path(snapshot_path).with_name(
            Path(snapshot_path).name.replace("_weights.json", "_A.json")
        )
        if sibling.is_file():
            artifact_paths.append(sibling)
    if artifact_paths:
        assert wandb is not None
        art = wandb.Artifact(name=f"skillit-update-step{int(step):07d}", type="skillit-update")
        for path in artifact_paths:
            art.add_file(str(path), name=path.name)
        run.log_artifact(art)


def wandb_log_checkpoint(
    run: object | None,
    ckpt_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    arm_id: str,
) -> None:
    if run is None:
        return
    wandb_log(
        run,
        {
            "checkpoint/step": int(step),
            "checkpoint/tokens_seen": int(tokens_seen),
            "checkpoint/arm_id": arm_id,
        },
        step=step,
    )
    assert wandb is not None
    art = wandb.Artifact(
        name=f"checkpoint-step{int(step):07d}",
        type="model",
        metadata={"step": int(step), "tokens_seen": int(tokens_seen), "arm_id": arm_id},
    )
    art.add_dir(str(ckpt_dir))
    run.log_artifact(art)
    log.info("wandb uploaded checkpoint artifact %s", art.name)


def wandb_upload_existing(
    run: object | None,
    *,
    save_folder: Path,
    progress_dir: Path,
    task_loss_results_dir: Path,
) -> None:
    """Best-effort upload of local checkpoints / evals / skillit snapshots."""
    if run is None:
        return
    for ckpt_dir in sorted(save_folder.glob("step*")):
        if not ckpt_dir.is_dir():
            continue
        if not (ckpt_dir / "state.pt").is_file() and not any(ckpt_dir.iterdir()):
            continue
        try:
            step = int(ckpt_dir.name.replace("step", "").split("-")[0])
        except ValueError:
            continue
        wandb_log_checkpoint(
            run,
            ckpt_dir,
            step=step,
            tokens_seen=0,
            arm_id=str(getattr(run, "name", "skillit")),
        )
    for eval_path in sorted(task_loss_results_dir.glob("step*_task_loss.json")):
        try:
            step = int(eval_path.name.split("_")[0].replace("step", ""))
        except ValueError:
            continue
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        wandb_log_eval(run, payload, step=step, eval_path=eval_path)
    updates = progress_dir / "skillit_updates.jsonl"
    if updates.is_file():
        assert wandb is not None
        art = wandb.Artifact(name="skillit-updates", type="metrics")
        art.add_file(str(updates), name=updates.name)
        run.log_artifact(art)
    meta = progress_dir / "run_meta.json"
    if meta.is_file():
        assert wandb is not None
        art = wandb.Artifact(name="run-meta", type="config")
        art.add_file(str(meta), name=meta.name)
        run.log_artifact(art)


def log_probe_final_eval(
    *,
    eval_path: Path,
    probe_id: str,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: Optional[str] = None,
    mode: str = "online",
    step: int = 0,
) -> Optional[str]:
    """One-shot W&B log for a probe final eval (no mixlaw trainer edits)."""
    if mode == "disabled":
        return None
    if wandb is None:
        log.warning("wandb package missing; skip probe W&B log")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY unset; skip probe W&B log")
        return None
    if not eval_path.is_file():
        log.warning("missing probe eval %s; skip W&B log", eval_path)
        return None
    os.environ.setdefault("WANDB_MODE", mode)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if step <= 0:
        # Prefer step parsed from checkpoint path if present.
        ckpt = str(payload.get("checkpoint") or "")
        for part in reversed(Path(ckpt).parts):
            if part.startswith("step"):
                digits = "".join(ch for ch in part if ch.isdigit())
                if digits:
                    step = int(digits)
                    break
        if step <= 0:
            step = 1451
    run = wandb.init(
        project=project,
        entity=entity or None,
        name=probe_id,
        job_type="probe-eval",
        config={"probe_id": probe_id, "eval_path": str(eval_path)},
        reinit=True,
    )
    try:
        wandb_log_eval(run, payload, step=step, eval_path=eval_path, prefix="probe/eval")
        return getattr(run, "url", None)
    finally:
        run.finish()


def add_wandb_args(parser: argparse.ArgumentParser, *, default_project: str = DEFAULT_WANDB_PROJECT) -> None:
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", default_project))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME") or None)
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP") or None)
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument(
        "--wandb-upload-existing",
        action="store_true",
        help="On start, upload existing local checkpoints/evals as W&B artifacts.",
    )
