"""Post-save hook: launch OLMo-ladder ``task_loss_bpb`` eval on a checkpoint.

Used by standalone arm trainers (Control / BLADE / MixLaw / …) that do not go
through ``Trainer.fit`` callbacks. Rank-0 only; other ranks should not call this.

Eval runs asynchronously by default so training can continue. Multi-GPU eval
(``nproc`` / ``TASK_LOSS_NPROC`` > 1) is always synchronous so training ranks
can idle while eval claims all devices. Disable with ``TASK_LOSS_EVAL=0`` or
``enabled=False``.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Union

log = logging.getLogger("token_selection.task_loss_hook")

# Repo-relative default when ``eval_script`` is not passed.
_DEFAULT_EVAL_REL = Path("scripts") / "farmshare" / "task_loss" / "eval_task_loss_olmo_core.py"

_DIST_ENV_KEYS = (
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
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "PET_NPROC_PER_NODE",
    "PET_NNODES",
    "PET_NODE_RANK",
    "PET_MASTER_ADDR",
    "PET_MASTER_PORT",
)


class TaskLossLaunchError(RuntimeError):
    """Task-loss evaluation could not be launched or completed."""


def _strict_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("TASK_LOSS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _launch_problem(message: str, *, strict: bool) -> None:
    if strict:
        raise TaskLossLaunchError(message)
    log.warning("%s", message)


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


def _resolve_nproc(explicit: Optional[int]) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    for key in ("TASK_LOSS_NPROC", "NPROC"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                log.warning("Ignoring non-integer %s=%r", key, raw)
    return 1


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _eval_env() -> dict:
    env = os.environ.copy()
    for k in _DIST_ENV_KEYS:
        env.pop(k, None)
    tlc = os.environ.get("TASK_LOSS_CUDA_VISIBLE_DEVICES")
    if tlc is not None:
        env["CUDA_VISIBLE_DEVICES"] = tlc
    return env


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
    nproc: Optional[int] = None,
    strict: Optional[bool] = None,
) -> Optional[subprocess.Popen]:
    """Spawn ``eval_task_loss_olmo_core.py`` against ``checkpoint_dir`` (proxy weights).

    When ``nproc`` (or ``TASK_LOSS_NPROC`` / ``NPROC``) is > 1, launches via
    ``torch.distributed.run`` so the eval script can shard labels across GPUs.
    Multi-GPU eval forces ``async_=False``.

    Returns the ``Popen`` handle when ``async_=True`` and the process started,
    otherwise ``None``. ``strict=True`` (or ``TASK_LOSS_STRICT=1``) forces a
    synchronous launch and raises on missing inputs, non-zero exit, or missing
    output. Non-strict smoke runs preserve warning-only behavior.
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

    strict_eff = _strict_enabled(strict)
    nproc_eff = _resolve_nproc(nproc)
    if nproc_eff > 1 or strict_eff:
        async_ = False

    ckpt = Path(checkpoint_dir)
    out = Path(out_path)
    script = resolve_eval_script(eval_script)
    if script is None:
        _launch_problem(
            f"task_loss eval script not found for {ckpt} "
            "(set TASK_LOSS_EVAL_SCRIPT)",
            strict=strict_eff,
        )
        return None
    has_state = (ckpt / "state.pt").is_file() or (ckpt / "model_eval.pt").is_file()
    has_distcp = (ckpt / "model_and_optim" / ".metadata").is_file()
    if not has_state and not has_distcp:
        _launch_problem(
            f"No state.pt / model_eval.pt / DistCP metadata under {ckpt}",
            strict=strict_eff,
        )
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    py = python_executable or sys.executable
    script_args: List[str] = [
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
        script_args.extend(str(a) for a in extra_args)

    if nproc_eff > 1:
        port = _free_port()
        cmd: List[str] = [
            py,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_eff}",
            f"--master_port={port}",
            *script_args,
        ]
    else:
        cmd = [py, *script_args]

    log.info(
        "Triggering task_loss eval (async=%s nproc=%s): %s",
        async_,
        nproc_eff,
        " ".join(cmd),
    )
    try:
        env = _eval_env()
        if async_:
            # Detach from training's CUDA / PG; eval script inits its own process group.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            log.info("task_loss eval spawned pid=%s → %s", proc.pid, out)
            return proc
        completed = subprocess.run(cmd, check=False, env=env)
        if completed.returncode != 0:
            _launch_problem(
                f"task_loss eval exited {completed.returncode} for {ckpt}",
                strict=strict_eff,
            )
            return None
        if strict_eff and not out.is_file():
            raise TaskLossLaunchError(
                f"task_loss eval exited successfully but did not write {out}"
            )
        return None
    except TaskLossLaunchError:
        raise
    except Exception as e:  # pragma: no cover - launch environment varies
        if strict_eff:
            raise TaskLossLaunchError(f"Failed to launch task_loss eval: {e}") from e
        log.warning("Failed to launch task_loss eval: %s", e)
        return None
