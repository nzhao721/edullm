"""OLMo-core callback: complete the permanent-checkpoint durability contract.

Wired from ``train_olmo_template`` for YAML arms (middle_ppl, rho, rel, …).
Every rank pauses while rank 0 waits for materialization, runs strict synchronous
task-loss evaluation, uploads artifacts to W&B, and advances the local marker.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Set

from .checkpoint_ladder import (
    DEFAULT_CHECKPOINT_INTERVAL,
    is_permanent_checkpoint_step,
)
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
    """Finalize each permanent ladder step before training continues.

    Priority 0 lets ``CheckpointerCallback`` (priority 1) finish its post-step
    work first. ``save_async=False`` is required by the YAML spine.
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
        progress_dir: Optional[str | Path] = None,
        method: Optional[str] = None,
        task_loss_nproc: Optional[int] = None,
        strict: bool = True,
        production: bool = False,
        wandb_mode: Optional[str] = None,
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
        self.arm = str(arm or method or self.save_folder.name)
        self.progress_dir = Path(progress_dir) if progress_dir else None
        self.method = str(method) if method else self.save_folder.name
        self.task_loss_nproc = (
            int(task_loss_nproc) if task_loss_nproc is not None else None
        )
        self.strict = bool(strict)
        self.production = bool(production)
        self.wandb_mode = wandb_mode

    @property
    def _rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _step_dir(self, step: int) -> Path:
        return self.save_folder / f"step{int(step)}"

    def _out_path(self, step: int) -> Path:
        return self.results_dir / f"step{int(step)}_task_loss.json"

    def _task_loss_eval_wanted(self) -> bool:
        if not self.enabled:
            return False
        return os.environ.get("TASK_LOSS_EVAL", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _maybe_launch(self, step: int) -> None:
        """Complete checkpoint → eval → export → marker in lockstep."""
        step = int(step)
        if step in self._launched:
            return
        if not is_permanent_checkpoint_step(step, self.total_steps, self.interval):
            return

        eval_wanted = self._task_loss_eval_wanted()
        self._launched.add(step)
        step_dir = self._step_dir(step)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        error: Optional[BaseException] = None
        if self._rank == 0:
            try:
                deadline = time.time() + 3600
                while time.time() < deadline:
                    ready = (
                        (step_dir / "model_and_optim" / ".metadata").is_file()
                        or (step_dir / "model_eval.pt").is_file()
                        or (step_dir / "state.pt").is_file()
                    )
                    if ready:
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"timed out waiting for permanent checkpoint {step_dir}"
                    )
                if self.command_template:
                    raise RuntimeError(
                        "eval.task_loss.command_template is out of contract; "
                        "use eval_script/resource fields so strict validation applies"
                    )
                from .permanent_checkpoint import finalize_permanent_checkpoint
                from .wandb_logging import wandb_run_from_trainer

                finalize_permanent_checkpoint(
                    arm=str(self.arm),
                    checkpoint_dir=step_dir,
                    step=step,
                    run_name=self.run_id,
                    task_loss_dir=self.results_dir,
                    task_loss_enabled=eval_wanted,
                    task_loss_eval_script=self.eval_script,
                    task_loss_nproc=self.task_loss_nproc,
                    progress_dir=self.progress_dir,
                    fingerprint_path=self.save_folder / "run_fingerprint.json",
                    method=self.method,
                    wandb_run=wandb_run_from_trainer(self.trainer),
                    wandb_mode=self.wandb_mode,
                    production=self.production,
                )
            except BaseException as exc:  # noqa: BLE001
                error = exc
        if self._rank == 0 and error is not None:
            # torchrun propagates the rank-0 failure and terminates ranks waiting
            # at the barrier; never continue past a partial permanent step.
            raise error
        self._barrier()
        if self._rank == 0:
            print(
                json.dumps(
                    {
                        "status": "permanent_checkpoint_complete",
                        "step": step,
                        "checkpoint": str(step_dir),
                        "arm": self.arm,
                        "task_loss_eval": eval_wanted,
                        "artifact_store": "wandb",
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
