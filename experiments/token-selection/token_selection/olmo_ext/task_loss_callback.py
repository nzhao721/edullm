"""OLMo-core callback: launch full-suite ``task_loss_bpb`` on each permanent ckpt.

Wired from ``train_olmo_template`` for YAML arms (middle_ppl, rho, rel, …).
Rank 0 only; evals run asynchronously so training can continue.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Set

from .checkpoint_ladder import (
    DEFAULT_CHECKPOINT_INTERVAL,
    is_permanent_checkpoint_step,
)
from .s3_export import export_arm_checkpoint
from .s3_layout import arm_from_prefix

try:  # pragma: no cover - OLMo-core is intentionally absent from local CI.
    from olmo_core.train.callbacks import Callback  # type: ignore

    _HAS_OLMO = True
except Exception:  # pragma: no cover
    Callback = object  # type: ignore
    _HAS_OLMO = False


def _default_eval_script() -> Path:
    # …/experiments/token-selection/token_selection/olmo_ext/this_file → repo root
    return (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "farmshare"
        / "task_loss"
        / "eval_task_loss_olmo_core.py"
    )


class TaskLossEvalCallback(Callback if _HAS_OLMO else object):  # type: ignore[misc]
    """Spawn background task-loss eval when a permanent ladder step is saved.

    Priority 0 so ``CheckpointerCallback`` (priority 1) starts the save first.
    The child polls until DistCP metadata (or ``state.pt`` / ``model_eval.pt``)
    exists, then runs the shared evaluator.
    """

    priority = 0

    def __init__(
        self,
        *,
        total_steps: int,
        save_folder: str | Path,
        run_id: str,
        results_dir: str | Path,
        interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        enabled: bool = True,
        command_template: Optional[str] = None,
        eval_script: Optional[str | Path] = None,
        device_eval_batch_size: int = 4,
        arm: Optional[str] = None,
        s3_prefix: Optional[str] = None,
        s3_export: bool = True,
    ) -> None:
        if not _HAS_OLMO:
            raise ImportError("olmo_core is required for TaskLossEvalCallback")
        super().__init__()  # type: ignore[misc]
        self.total_steps = int(total_steps)
        self.save_folder = Path(save_folder)
        self.run_id = str(run_id)
        self.results_dir = Path(results_dir)
        self.interval = int(interval)
        self.enabled = bool(enabled)
        self.command_template = command_template
        self.eval_script = Path(eval_script) if eval_script else _default_eval_script()
        self.device_eval_batch_size = int(device_eval_batch_size)
        self._launched: Set[int] = set()
        if arm:
            self.arm = str(arm)
        elif s3_prefix:
            self.arm = arm_from_prefix(str(s3_prefix))
        else:
            self.arm = None
        self.s3_export = bool(s3_export)

    @property
    def _rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _step_dir(self, step: int) -> Path:
        return self.save_folder / f"step{int(step)}"

    def _out_path(self, step: int) -> Path:
        return self.results_dir / f"step{int(step)}_task_loss.json"

    def _build_cmd(self, step: int, step_dir: Path, out_path: Path) -> list[str]:
        if self.command_template:
            rendered = self.command_template.format(
                checkpoint=str(step_dir),
                out=str(out_path),
                run_id=self.run_id,
                run_name=self.run_id,
                step=int(step),
                eval_script=str(self.eval_script),
                python=sys.executable,
            )
            return shlex.split(rendered)
        return [
            sys.executable,
            str(self.eval_script),
            "--checkpoint",
            str(step_dir),
            "--out",
            str(out_path),
            "--run-name",
            self.run_id,
            "--device-eval-batch-size",
            str(self.device_eval_batch_size),
        ]

    def _maybe_launch(self, step: int) -> None:
        if not self.enabled or self._rank != 0:
            return
        if os.environ.get("TASK_LOSS_EVAL", "1").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        step = int(step)
        if step in self._launched:
            return
        if not is_permanent_checkpoint_step(step, self.total_steps, self.interval):
            return
        self._launched.add(step)
        step_dir = self._step_dir(step)
        out_path = self._out_path(step)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.s3_export and self.arm:
            # Best-effort: upload the step dir once DistCP/state.pt lands (waiter
            # also re-exports after ready). Immediate call covers state.pt trainers.
            export_arm_checkpoint(self.arm, step_dir, method=self.save_folder.name)
        cmd = self._build_cmd(step, step_dir, out_path)
        # Wait for async DistCP finalize, then run eval (detached from train PG).
        arm = self.arm
        method = self.save_folder.name
        s3_export = self.s3_export and bool(arm)
        results_dir = self.results_dir
        waiter = f"""
import json, os, subprocess, sys, time
from pathlib import Path
step_dir = Path({str(step_dir)!r})
out_path = Path({str(out_path)!r})
results_dir = Path({str(results_dir)!r})
cmd = {cmd!r}
arm = {arm!r}
method = {method!r}
s3_export = {s3_export!r}
deadline = time.time() + 3600
ready = lambda: (
    (step_dir / "model_and_optim" / ".metadata").is_file()
    or (step_dir / "model_eval.pt").is_file()
    or (step_dir / "state.pt").is_file()
)
while time.time() < deadline and not ready():
    time.sleep(5)
if not ready():
    print(json.dumps({{"status": "timeout_waiting_ckpt", "step_dir": str(step_dir)}}), flush=True)
    sys.exit(2)
if s3_export and arm:
    try:
        from token_selection.olmo_ext.s3_export import export_arm_checkpoint, export_arm_task_loss_dir
        export_arm_checkpoint(arm, step_dir, method=method)
    except Exception as exc:
        print(json.dumps({{"status": "s3_ckpt_export_warn", "error": str(exc)}}), flush=True)
if out_path.is_file():
    print(json.dumps({{"status": "skip_exists", "out": str(out_path)}}), flush=True)
    if s3_export and arm:
        try:
            from token_selection.olmo_ext.s3_export import export_arm_task_loss_dir
            export_arm_task_loss_dir(arm, results_dir)
        except Exception:
            pass
    sys.exit(0)
env = os.environ.copy()
for k in (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
):
    env.pop(k, None)
tlc = os.environ.get("TASK_LOSS_CUDA_VISIBLE_DEVICES")
if tlc is not None:
    env["CUDA_VISIBLE_DEVICES"] = tlc
print(json.dumps({{"status": "launching_task_loss", "cmd": cmd}}), flush=True)
rc = subprocess.call(cmd, env=env)
if s3_export and arm:
    try:
        from token_selection.olmo_ext.s3_export import export_arm_task_loss_dir
        export_arm_task_loss_dir(arm, results_dir)
    except Exception as exc:
        print(json.dumps({{"status": "s3_task_loss_export_warn", "error": str(exc)}}), flush=True)
raise SystemExit(rc)
"""
        log_path = self.results_dir / f"step{step}_task_loss_launch.log"
        with open(log_path, "a", encoding="utf-8") as log_f:
            subprocess.Popen(  # noqa: S603 — intentional async eval handoff
                [sys.executable, "-c", waiter],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        print(
            json.dumps(
                {
                    "status": "task_loss_spawned",
                    "step": step,
                    "checkpoint": str(step_dir),
                    "out": str(out_path),
                    "log": str(log_path),
                    "arm": self.arm,
                    "s3_export": self.s3_export and bool(self.arm),
                }
            ),
            flush=True,
        )

    def pre_train(self) -> None:  # pragma: no cover - requires olmo_core
        self._maybe_launch(0)

    def post_step(self) -> None:  # pragma: no cover - requires olmo_core
        self._maybe_launch(int(self.step))

    def post_train(self) -> None:  # pragma: no cover - requires olmo_core
        self._maybe_launch(int(self.step))
