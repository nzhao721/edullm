"""Post-save hook: launch OLMo-ladder ``task_loss_bpb`` eval on a checkpoint.

Used by standalone arm trainers (Control / BLADE / …) that do not go through
``Trainer.fit`` callbacks. Rank-0 only; other ranks should not call this.

Eval runs asynchronously by default so training can continue. Disable with
``TASK_LOSS_EVAL=0`` or ``enabled=False``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

log = logging.getLogger("token_selection.task_loss_hook")

# Repo-relative default when ``eval_script`` is not passed.
_DEFAULT_EVAL_REL = Path("scripts") / "farmshare" / "task_loss" / "eval_task_loss_olmo_core.py"


def _repo_root_from_here() -> Path:
    # …/experiments/token-selection/token_selection/olmo_ext/task_loss_hook.py
    return Path(__file__).resolve().parents[4]


def resolve_eval_script(explicit: Optional[Union[str, Path]] = None) -> Optional[Path]:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get("TASK_LOSS_EVAL_SCRIPT", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    candidate = _repo_root_from_here() / _DEFAULT_EVAL_REL
    return candidate if candidate.is_file() else None


def trigger_task_loss_eval(
    checkpoint_dir: Union[str, Path],
    *,
    run_name: str,
    out_path: Union[str, Path],
    eval_script: Optional[Union[str, Path]] = None,
    extra_args: Optional[Sequence[str]] = None,
    async_: bool = True,
    enabled: Optional[bool] = None,
    python_executable: Optional[str] = None,
) -> Optional[subprocess.Popen]:
    """Spawn ``eval_task_loss_olmo_core.py`` against ``checkpoint_dir`` (proxy weights).

    Returns the ``Popen`` handle when ``async_=True`` and the process started,
    otherwise ``None``. Never raises on launch failure — logs and returns None
    so a missing eval stack cannot kill training.
    """
    if enabled is None:
        enabled = os.environ.get("TASK_LOSS_EVAL", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    if not enabled:
        log.info("task_loss eval disabled (TASK_LOSS_EVAL / enabled=False)")
        return None

    ckpt = Path(checkpoint_dir)
    out = Path(out_path)
    script = resolve_eval_script(eval_script)
    if script is None:
        log.warning(
            "task_loss eval script not found; skip eval for %s (set TASK_LOSS_EVAL_SCRIPT)",
            ckpt,
        )
        return None
    has_state = (ckpt / "state.pt").is_file() or (ckpt / "model_eval.pt").is_file()
    has_distcp = (ckpt / "model_and_optim" / ".metadata").is_file()
    if not has_state and not has_distcp:
        log.warning(
            "No state.pt / model_eval.pt / DistCP metadata under %s; skip task_loss eval",
            ckpt,
        )
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    py = python_executable or sys.executable
    cmd: List[str] = [
        py,
        str(script),
        "--checkpoint",
        str(ckpt),
        "--out",
        str(out),
        "--run-name",
        str(run_name),
        "--format",
        "auto",
    ]
    if extra_args:
        cmd.extend(str(a) for a in extra_args)

    log.info("Triggering task_loss eval (async=%s): %s", async_, " ".join(cmd))
    try:
        if async_:
            # Detach from training's CUDA / PG; eval script inits its own process group.
            env = os.environ.copy()
            for k in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_NAME",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCHELASTIC_RUN_ID",
            ):
                env.pop(k, None)
            # Prefer a free GPU if the caller set TASK_LOSS_CUDA_VISIBLE_DEVICES.
            tlc = os.environ.get("TASK_LOSS_CUDA_VISIBLE_DEVICES")
            if tlc is not None:
                env["CUDA_VISIBLE_DEVICES"] = tlc
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            log.info("task_loss eval spawned pid=%s → %s", proc.pid, out)
            return proc
        subprocess.run(cmd, check=False)
        return None
    except Exception as e:  # pragma: no cover - launch environment varies
        log.warning("Failed to launch task_loss eval: %s", e)
        return None
