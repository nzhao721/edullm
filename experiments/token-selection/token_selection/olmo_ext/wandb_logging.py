"""SmolLM2-style W&B helpers for token-selection trainers.

Protocol (mirrors ``scripts/farmshare/train_smollm2_135m_ddp.py``):

* Enable when ``wandb_mode != disabled``, ``wandb`` is importable, and
  ``WANDB_API_KEY`` is set; otherwise continue without W&B (warn on rank 0).
* One project for all arms: ``token-selection``.
* Run names distinguish arms (``control-…``, ``rho-1-…``, ``blade-…``, …).
* Persist ``wandb_run_id.txt`` under the progress/output dir for resume.
* Log train metrics, task-loss eval metrics (+ artifact), and checkpoints
  (+ model artifact).

Does **not** replace durable S3 export (``edullm-checkpoints``) or weaken
``edullm-data`` fail-closed staging.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

log = logging.getLogger("token_selection.wandb")

DEFAULT_WANDB_PROJECT = "token-selection"
DEFAULT_WANDB_MODE = "online"

try:
    import wandb as _wandb
except ImportError:  # pragma: no cover
    _wandb = None  # type: ignore[assignment]


def add_wandb_argparse_options(
    parser: argparse.ArgumentParser,
    *,
    default_project: str = DEFAULT_WANDB_PROJECT,
    default_run_name: Optional[str] = None,
) -> None:
    """Add SmolLM2-parity ``--wandb-*`` flags to a trainer ArgumentParser."""
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", default_project),
        help=f"W&B project (default: {default_project})",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY") or None,
        help="Optional W&B entity/team",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME") or default_run_name,
        help="W&B run name (defaults to arm run id / --name)",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP") or None,
        help="Optional W&B group (defaults to arm name when set by caller)",
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE),
        choices=("online", "offline", "disabled"),
        help="W&B mode (default: online; soft-skip without API key)",
    )
    parser.add_argument(
        "--wandb-upload-existing",
        action="store_true",
        default=os.environ.get("WANDB_UPLOAD_EXISTING", "").strip().lower()
        in {"1", "true", "yes", "on"},
        help="On start, upload existing local checkpoints/evals as W&B artifacts",
    )


def wandb_mode_from_args(args: Any) -> str:
    mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE)
    return str(mode).strip().lower() or DEFAULT_WANDB_MODE


def wandb_enabled(
    *,
    mode: Optional[str] = None,
    is_main: bool = True,
) -> bool:
    """True when this rank should open/log a W&B run (SmolLM2 gate)."""
    if not is_main:
        return False
    resolved = (mode or os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE)).strip().lower()
    if resolved == "disabled":
        return False
    if _wandb is None:
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def resolve_run_name(
    *,
    explicit: Optional[str] = None,
    arm: Optional[str] = None,
    run_id: Optional[str] = None,
    method: Optional[str] = None,
) -> str:
    """Prefer explicit name; else ``run_id``; else ``{arm}-{method}``."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    if run_id and str(run_id).strip():
        return str(run_id).strip()
    parts = [p for p in (arm, method) if p and str(p).strip()]
    if parts:
        return "-".join(str(p).strip() for p in parts)
    return "token-selection"


def init_wandb(
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: Optional[str] = None,
    run_name: str,
    mode: str = DEFAULT_WANDB_MODE,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    config: Optional[Mapping[str, Any]] = None,
    dir: Optional[Path | str] = None,
    id_path: Optional[Path | str] = None,
    is_main: bool = True,
    alert_title: Optional[str] = None,
) -> Any | None:
    """Initialize (or resume) a W&B run. Returns the run or None."""
    mode = (mode or DEFAULT_WANDB_MODE).strip().lower()
    if mode == "disabled":
        return None
    if not is_main:
        return None
    if _wandb is None:
        log.warning("wandb package missing; continuing without W&B")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY unset; continuing without W&B")
        return None

    os.environ.setdefault("WANDB_MODE", mode)
    if dir is not None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(dir))

    run_id: Optional[str] = None
    id_file = Path(id_path) if id_path is not None else None
    if id_file is not None and id_file.is_file():
        run_id = id_file.read_text(encoding="utf-8").strip() or None
    if run_id is None:
        run_id = os.environ.get("WANDB_RUN_ID") or None

    init_kwargs: dict[str, Any] = {
        "project": project or DEFAULT_WANDB_PROJECT,
        "entity": entity or None,
        "name": run_name,
        "config": dict(config or {}),
        "reinit": True,
    }
    if group:
        init_kwargs["group"] = group
    if tags:
        init_kwargs["tags"] = list(tags)
    if dir is not None:
        init_kwargs["dir"] = str(dir)
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    run = _wandb.init(**init_kwargs)
    os.environ["WANDB_RUN_ID"] = str(run.id)
    if id_file is not None:
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(str(run.id), encoding="utf-8")
    log.info("wandb run=%s url=%s project=%s name=%s", run.id, run.url, project, run_name)
    title = alert_title or "token-selection train job started"
    try:
        run.alert(
            title=title,
            text=(
                f"run={run.name} id={run.id} "
                f"slurm_job={os.environ.get('SLURM_JOB_ID', 'n/a')} "
                f"host={os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', 'n/a'))}"
            ),
            level=_wandb.AlertLevel.INFO,
        )
    except Exception as exc:  # pragma: no cover - alert is best-effort
        log.debug("wandb alert skipped: %s", exc)
    return run


def init_wandb_from_args(
    args: Any,
    *,
    run_name: str,
    config: Optional[Mapping[str, Any]] = None,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    dir: Optional[Path | str] = None,
    id_path: Optional[Path | str] = None,
    is_main: bool = True,
    alert_title: Optional[str] = None,
) -> Any | None:
    """Convenience wrapper using argparse Namespace from ``add_wandb_argparse_options``."""
    return init_wandb(
        project=getattr(args, "wandb_project", None) or DEFAULT_WANDB_PROJECT,
        entity=getattr(args, "wandb_entity", None),
        run_name=getattr(args, "wandb_run_name", None) or run_name,
        mode=wandb_mode_from_args(args),
        group=getattr(args, "wandb_group", None) or group,
        tags=tags,
        config=config,
        dir=dir,
        id_path=id_path,
        is_main=is_main,
        alert_title=alert_title,
    )


def wandb_log(run: Any | None, metrics: Mapping[str, Any], *, step: int) -> None:
    if run is None:
        return
    clean = {k: v for k, v in metrics.items() if v is not None}
    if not clean:
        return
    run.log(dict(clean), step=int(step))


def wandb_log_train(
    run: Any | None,
    *,
    step: int,
    train_loss: Optional[float] = None,
    tokens_seen: Optional[int] = None,
    tok_per_s: Optional[float] = None,
    tok_per_s_avg: Optional[float] = None,
    lr: Optional[float] = None,
    selected_frac: Optional[float] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Log train-loop scalars under ``train/`` (SmolLM2-style namespaces)."""
    metrics: dict[str, Any] = {}
    if train_loss is not None:
        metrics["train/loss"] = float(train_loss)
    if tokens_seen is not None:
        metrics["train/tokens_seen"] = int(tokens_seen)
    if tok_per_s is not None:
        metrics["train/tok_per_s"] = float(tok_per_s)
    if tok_per_s_avg is not None:
        metrics["train/tok_per_s_avg"] = float(tok_per_s_avg)
    if lr is not None:
        metrics["train/lr"] = float(lr)
    if selected_frac is not None:
        metrics["train/selected_frac"] = float(selected_frac)
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            key = k if "/" in str(k) else f"train/{k}"
            metrics[key] = v
    wandb_log(run, metrics, step=step)


def task_loss_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Flatten a ``step*_task_loss.json`` payload into W&B eval scalars."""
    metrics: dict[str, float] = {}
    if "macro_mean" in payload:
        metrics["eval/macro_bpb"] = float(payload["macro_mean"])
    elif "macro_bpb" in payload:
        metrics["eval/macro_bpb"] = float(payload["macro_bpb"])
    if "macro_mean_accuracy" in payload:
        metrics["eval/macro_acc"] = float(payload["macro_mean_accuracy"])
    for k, v in (payload.get("labels") or {}).items():
        metrics[f"eval/bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_labels") or {}).items():
        metrics[f"eval/acc/{k}"] = float(v)
    for k, v in (payload.get("task_families") or {}).items():
        metrics[f"eval/family_bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_families") or {}).items():
        metrics[f"eval/family_acc/{k}"] = float(v)
    return metrics


def wandb_log_eval(
    run: Any | None,
    payload: Mapping[str, Any],
    *,
    step: int,
    eval_path: Optional[Path | str] = None,
) -> None:
    if run is None:
        return
    metrics = task_loss_metrics(payload)
    wandb_log(run, metrics, step=step)
    if eval_path is None or _wandb is None:
        return
    path = Path(eval_path)
    if not path.is_file():
        return
    art = _wandb.Artifact(name=f"eval-step{int(step):07d}", type="eval")
    art.add_file(str(path), name=path.name)
    run.log_artifact(art)


def wandb_log_checkpoint(
    run: Any | None,
    ckpt_dir: Path | str,
    *,
    step: int,
    tokens_seen: Optional[int] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
    upload_artifact: bool = True,
) -> None:
    if run is None:
        return
    metrics: dict[str, Any] = {"checkpoint/step": int(step)}
    if tokens_seen is not None:
        metrics["checkpoint/tokens_seen"] = int(tokens_seen)
    if extra_meta:
        for k, v in extra_meta.items():
            if v is None:
                continue
            metrics[f"checkpoint/{k}"] = v
    wandb_log(run, metrics, step=step)
    if not upload_artifact or _wandb is None:
        return
    path = Path(ckpt_dir)
    if not path.exists():
        return
    meta = {"step": int(step)}
    if tokens_seen is not None:
        meta["tokens_seen"] = int(tokens_seen)
    if extra_meta:
        meta.update({k: v for k, v in extra_meta.items() if v is not None})
    art = _wandb.Artifact(
        name=f"checkpoint-step{int(step):07d}",
        type="model",
        metadata=meta,
    )
    # Prefer compact state files over full DistCP trees when present.
    state = path / "state.pt"
    model_eval = path / "model_eval.pt"
    if state.is_file() or model_eval.is_file():
        for f in (state, model_eval, path / "trainer_state.pt", path / "run_meta.json"):
            if f.is_file():
                art.add_file(str(f), name=f.name)
    else:
        art.add_dir(str(path))
    run.log_artifact(art)
    log.info("wandb uploaded checkpoint artifact %s", art.name)


class WandbEvalPoller:
    """Poll ``step*_task_loss.json`` and log newly finished evals to W&B.

    Async task-loss workers write JSON after training continues; call
    ``poll()`` from the train loop / callback so evals reach W&B without
    blocking the train PG.
    """

    def __init__(self, results_dir: Path | str, run: Any | None = None) -> None:
        self.results_dir = Path(results_dir)
        self.run = run
        self._logged: set[int] = set()

    def bind(self, run: Any | None) -> None:
        self.run = run

    def mark_logged(self, step: int) -> None:
        self._logged.add(int(step))

    def poll(self) -> list[int]:
        if self.run is None or not self.results_dir.is_dir():
            return []
        logged_now: list[int] = []
        for path in sorted(self.results_dir.glob("step*_task_loss.json")):
            name = path.name
            try:
                step = int(name.split("_", 1)[0].replace("step", ""))
            except ValueError:
                continue
            if step in self._logged:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            # Require at least one recognized scalar so partial writes are skipped.
            if "macro_mean" not in payload and "macro_bpb" not in payload and "labels" not in payload:
                continue
            wandb_log_eval(self.run, payload, step=step, eval_path=path)
            self._logged.add(step)
            logged_now.append(step)
        return logged_now


def wandb_upload_existing(
    run: Any | None,
    *,
    checkpoint_dir: Optional[Path | str] = None,
    task_loss_dir: Optional[Path | str] = None,
    progress_dir: Optional[Path | str] = None,
    tokens_per_step: Optional[int] = None,
) -> None:
    """Upload existing local checkpoints/evals (``--wandb-upload-existing``)."""
    if run is None:
        return
    ckpt_root = Path(checkpoint_dir) if checkpoint_dir else None
    if ckpt_root is not None and ckpt_root.exists():
        for ckpt_dir in sorted(ckpt_root.glob("step*")):
            if not ckpt_dir.is_dir():
                continue
            try:
                step = int(ckpt_dir.name.replace("step", ""))
            except ValueError:
                continue
            has_payload = (
                (ckpt_dir / "state.pt").is_file()
                or (ckpt_dir / "model_eval.pt").is_file()
                or (ckpt_dir / "model_and_optim" / ".metadata").is_file()
                or (ckpt_dir / "trainer_state.pt").is_file()
            )
            if not has_payload:
                continue
            tokens = step * tokens_per_step if tokens_per_step else None
            wandb_log_checkpoint(run, ckpt_dir, step=step, tokens_seen=tokens)
    tl_root = Path(task_loss_dir) if task_loss_dir else None
    if tl_root is not None and tl_root.exists():
        poller = WandbEvalPoller(tl_root, run)
        poller.poll()
    if progress_dir is not None:
        progress = Path(progress_dir)
        for name, art_type in (
            ("train_loss.jsonl", "metrics"),
            ("run_meta.json", "config"),
            ("checkpoint_ladder.json", "config"),
        ):
            path = progress / name
            if path.is_file() and _wandb is not None:
                art = _wandb.Artifact(name=path.stem.replace("_", "-"), type=art_type)
                art.add_file(str(path), name=path.name)
                run.log_artifact(art)


def finish_wandb(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover
        log.warning("wandb.finish failed: %s", exc)


def wandb_callback_kwargs_from_env(
    *,
    run_name: str,
    arm: Optional[str] = None,
    method: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    enabled: Optional[bool] = None,
) -> dict[str, Any]:
    """Build kwargs for ``olmo_core.train.callbacks.WandBCallback``.

    Soft-disable when API key / package missing so OLMo does not raise
    ``OLMoEnvironmentError`` on FarmShare smoke without a session file.
    """
    mode = os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE).strip().lower()
    if enabled is None:
        enabled = wandb_enabled(mode=mode, is_main=True)
    tags = [t for t in (arm, method) if t]
    kwargs: dict[str, Any] = {
        "enabled": bool(enabled),
        "name": resolve_run_name(
            explicit=os.environ.get("WANDB_RUN_NAME") or os.environ.get("WANDB_NAME"),
            arm=arm,
            run_id=run_name,
            method=method,
        ),
        "project": os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        "config": dict(config or {}),
    }
    entity = os.environ.get("WANDB_ENTITY") or None
    if entity:
        kwargs["entity"] = entity
    group = os.environ.get("WANDB_GROUP") or arm
    if group:
        kwargs["group"] = group
    if tags:
        kwargs["tags"] = tags
    return kwargs


def ensure_wandb_not_hard_disabled() -> None:
    """Clear import-time hard disables when a session wants W&B online/offline.

    Call early in trainers that previously forced ``WANDB_DISABLED=1``. Does
    nothing when ``WANDB_MODE=disabled``.
    """
    mode = os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE).strip().lower()
    if mode == "disabled":
        os.environ["WANDB_DISABLED"] = "1"
        return
    os.environ.pop("WANDB_DISABLED", None)
    # Keep MODE as requested; do not invent an API key.
    if mode in {"online", "offline"}:
        os.environ["WANDB_MODE"] = mode


def apply_wandb_env_defaults(
    *,
    project: str = DEFAULT_WANDB_PROJECT,
    run_name: Optional[str] = None,
    group: Optional[str] = None,
) -> None:
    """Set project/run defaults without overriding caller exports."""
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_RUN_NAME", run_name)
        os.environ.setdefault("WANDB_NAME", run_name)
    if group:
        os.environ.setdefault("WANDB_GROUP", group)
    os.environ.setdefault("WANDB_MODE", DEFAULT_WANDB_MODE)


def namespace_path_config(args: Any) -> MutableMapping[str, Any]:
    """JSON-safe config dict from argparse (Paths → str)."""
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _wandb_run_from_trainer(trainer: Any) -> Any | None:
    """Best-effort extract of an active W&B run from olmo_core callbacks."""
    callbacks = getattr(trainer, "callbacks", None) or {}
    for key in ("wandb", "wandb_ts", "token_selection_wandb"):
        cb = callbacks.get(key) if isinstance(callbacks, Mapping) else getattr(callbacks, key, None)
        if cb is None:
            continue
        for attr in ("run", "_run", "wandb_run"):
            run = getattr(cb, attr, None)
            if run is not None:
                return run
    return None


class WandbArtifactsCallback:  # subclassed onto olmo Callback when available
    """Log checkpoint artifacts + poll async task-loss JSON into W&B.

    Pair with ``olmo_core.train.callbacks.WandBCallback`` (train scalars) on the
    YAML spine / reference Trainer path. Priority 0 so it sees post-save steps.
    """

    priority = 0

    def __init__(
        self,
        *,
        results_dir: Path | str,
        save_folder: Path | str,
        total_steps: int,
        interval: int = 125,
        tokens_per_step: Optional[int] = None,
        upload_checkpoint_artifacts: bool = True,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.save_folder = Path(save_folder)
        self.total_steps = int(total_steps)
        self.interval = int(interval)
        self.tokens_per_step = tokens_per_step
        self.upload_checkpoint_artifacts = bool(upload_checkpoint_artifacts)
        self.poller = WandbEvalPoller(self.results_dir)
        self._ckpt_logged: set[int] = set()
        self.trainer: Any = None

    def post_attach(self) -> None:  # pragma: no cover - requires olmo_core
        pass

    def _bind_run(self) -> None:
        if self.poller.run is not None:
            return
        run = _wandb_run_from_trainer(self.trainer)
        if run is not None:
            self.poller.bind(run)

    def _maybe_log_checkpoint(self, step: int) -> None:
        from .checkpoint_ladder import is_permanent_checkpoint_step

        step = int(step)
        if step in self._ckpt_logged:
            return
        if not is_permanent_checkpoint_step(step, self.total_steps, self.interval):
            return
        self._bind_run()
        run = self.poller.run
        if run is None:
            return
        step_dir = self.save_folder / f"step{step}"
        if not step_dir.exists():
            return
        tokens = step * self.tokens_per_step if self.tokens_per_step else None
        wandb_log_checkpoint(
            run,
            step_dir,
            step=step,
            tokens_seen=tokens,
            upload_artifact=self.upload_checkpoint_artifacts,
        )
        self._ckpt_logged.add(step)

    def pre_train(self) -> None:  # pragma: no cover
        self._bind_run()
        self._maybe_log_checkpoint(0)
        self.poller.poll()

    def post_step(self) -> None:  # pragma: no cover
        step = int(getattr(self.trainer, "global_step", 0) or getattr(self, "step", 0) or 0)
        self._bind_run()
        self._maybe_log_checkpoint(step)
        self.poller.poll()

    def post_train(self) -> None:  # pragma: no cover
        self.post_step()


def make_wandb_artifacts_callback(**kwargs: Any) -> Any:
    """Return an olmo_core Callback subclass instance when available."""
    try:
        from olmo_core.train.callbacks import Callback  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("olmo_core is required for WandbArtifactsCallback") from exc

    class _CB(WandbArtifactsCallback, Callback):  # type: ignore[misc,valid-type]
        def __init__(self, **kw: Any) -> None:
            Callback.__init__(self)  # type: ignore[misc]
            WandbArtifactsCallback.__init__(self, **kw)

    return _CB(**kwargs)


def mirror_raw_step_to_wandb(run: Any | None, row: Mapping[str, Any]) -> None:
    """Mirror one RawComputeCallback / MetricLogger step row to W&B."""
    if run is None:
        return
    step = int(row.get("step", 0))
    extra = {
        "selected_frac": row.get("selected_frac"),
        "k": row.get("k"),
        "alpha": row.get("alpha"),
        "warmup": 1.0 if row.get("warmup") else 0.0,
        "mean_rel_kept": row.get("mean_rel_kept"),
        "mean_rel_dropped": row.get("mean_rel_dropped"),
        "selected_tokens": row.get("selected_tokens"),
        "forward_tokens_train": row.get("forward_tokens_train"),
        "forward_tokens_history": row.get("forward_tokens_history"),
        "method": row.get("method"),
    }
    wandb_log_train(
        run,
        step=step,
        train_loss=row.get("train_loss"),
        tokens_seen=row.get("tokens_seen"),
        extra={k: v for k, v in extra.items() if v is not None},
    )
