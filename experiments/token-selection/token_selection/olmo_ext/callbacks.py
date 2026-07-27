"""OLMo-core callbacks for persisted raw train compute (no held-out evaluation).

This module is import-safe without OLMo-core. The production callback uses the
train module's compute-delta seam so FSDP ranks contribute to one ledger.
Test-loss / benchmark evaluation is intentionally deferred to a later pass on
saved checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist

from .metrics import MetricLogger, StepMetrics, empty_metrics_payload

try:  # pragma: no cover - OLMo-core is intentionally absent from local CI.
    from olmo_core.train.callbacks import Callback  # type: ignore

    _HAS_OLMO = True
except Exception:  # pragma: no cover
    Callback = object  # type: ignore
    _HAS_OLMO = False


def _all_reduce_counters(counters: Mapping[str, int], device: torch.device) -> Dict[str, int]:
    """Sum rank-local raw counters, preserving a single global ledger."""
    names = sorted(counters)
    values = torch.tensor([int(counters[name]) for name in names], device=device, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: int(value) for name, value in zip(names, values.cpu().tolist())}


def _all_reduce_sums(sums: Mapping[str, float], device: torch.device) -> Dict[str, float]:
    """Sum rank-local float accumulators (REL score sums, batch CE) across ranks."""
    names = sorted(sums)
    values = torch.tensor(
        [float(sums[name]) for name in names], device=device, dtype=torch.float64
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: float(value) for name, value in zip(names, values.cpu().tolist())}


def _safe_mean(total: float, count: float) -> Optional[float]:
    return total / count if count else None


class RawComputeCallback(Callback if _HAS_OLMO else object):  # type: ignore[misc]
    """Persist raw training counters and per-step REL selection diagnostics.

    Rank zero writes the canonical ledger. Alongside the compute counters it records
    one row per optimizer step with the batch CE, the fraction of valid tokens kept,
    and the mean REL score of kept vs dropped tokens. Those last two are the run's
    only online evidence that selection is behaving: if the kept and dropped means
    converge, or the kept mean sits below the dropped mean, REL is not separating
    tokens and the run is not testing what it claims to.

    Held-out evaluation is not performed here; checkpoints are the handoff for later
    test-loss evaluation.
    """

    # Run before the priority-1 CheckpointerCallback so the callback state and
    # flushed ledger agree with every saved optimizer step.
    priority = 2
    STATE_VERSION = 1

    def __init__(
        self,
        *,
        metrics_path: str | Path,
        payload: Mapping[str, Any],
        resume: bool = False,
    ) -> None:
        if not _HAS_OLMO:
            raise ImportError("olmo_core is required for RawComputeCallback")
        super().__init__()  # type: ignore[misc]
        self.metrics_path = Path(metrics_path)
        self.payload = dict(payload)
        self.resume = bool(resume)
        self._logger: Optional[MetricLogger] = None
        self._trimmed_to_checkpoint = not self.resume

    @property
    def _rank(self) -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def post_attach(self) -> None:  # pragma: no cover - requires olmo_core
        module = self.trainer.train_module
        n_params = _all_reduce_counters(
            {"n_params": int(sum(parameter.numel() for parameter in module.model.parameters()))},
            module.device,
        )["n_params"]
        if self._rank == 0:
            payload = self.payload
            if self.resume and self.metrics_path.exists():
                payload = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            elif self.metrics_path.exists():
                self.metrics_path.unlink()
                jsonl = self.metrics_path.with_suffix(".jsonl")
                if jsonl.exists():
                    jsonl.unlink()
            self._logger = MetricLogger(self.metrics_path, payload)
            if self._logger.payload["model"].get("n_params") is None:
                self._logger.payload["model"]["n_params"] = n_params

    def _current_step(self) -> int:
        step = getattr(self.trainer, "global_step", None)
        if step is not None:
            return int(step)
        assert self._logger is not None
        return len(self._logger.payload["steps"])

    def _drop_rows_after_checkpoint(self, step: int) -> None:
        """Discard ledger rows for batches the restored checkpoint predates.

        A crash can leave rows written after the last checkpoint. Those steps are
        replayed on resume, so keeping them would double-count the curves.
        """
        if self._trimmed_to_checkpoint:
            return
        self._trimmed_to_checkpoint = True
        assert self._logger is not None
        payload = self._logger.payload
        for key in ("steps", "train_loss_curve", "selection_curve"):
            rows = payload.get(key) or []
            payload[key] = [row for row in rows if int(row.get("step", -1)) < step]

    def _consume_train_compute(self) -> None:
        module = self.trainer.train_module
        if not hasattr(module, "consume_token_selection_compute_delta"):
            raise RuntimeError(
                "RawComputeCallback requires TokenSelectTrainModule; "
                "the attached module has no raw compute delta."
            )
        local = module.consume_token_selection_compute_delta()
        global_counters = _all_reduce_counters(local, module.device)
        local_selection = module.consume_token_selection_selection_delta()
        selection = _all_reduce_sums(local_selection, module.device)
        progress = module.token_selection_progress()
        if self._rank != 0:
            return

        assert self._logger is not None
        self._logger.add_compute(
            train_tokens=global_counters["forward_tokens_train"],
            selected_tokens=global_counters["selected_tokens"],
            forward_tokens_train=global_counters["forward_tokens_train"],
            forward_tokens_history=global_counters["forward_tokens_history"],
            forward_tokens_current=global_counters["forward_tokens_current"],
            fwd_passes_train=global_counters["fwd_passes_train"],
            fwd_passes_history=global_counters["fwd_passes_history"],
            fwd_passes_current=global_counters["fwd_passes_current"],
        )
        step = self._current_step()
        self._drop_rows_after_checkpoint(step)
        n_kept = selection["n_kept"]
        n_valid = selection["n_valid"]
        self._logger.log_step(
            StepMetrics(
                step=step,
                tokens_seen=int(self._logger.payload["compute"]["forward_tokens_train"]),
                k=float(progress["k"]),
                alpha=float(progress["alpha"]),
                warmup=bool(progress["warmup"]),
                selected_frac=(n_kept / n_valid) if n_valid else 1.0,
                mean_rel_kept=_safe_mean(selection["rel_score_sum_kept"], n_kept),
                mean_rel_dropped=_safe_mean(selection["rel_score_sum_dropped"], selection["n_dropped"]),
                train_loss=_safe_mean(selection["ce_loss_sum"], selection["n_batches"]),
                selected_tokens=global_counters["selected_tokens"],
                forward_tokens_train=global_counters["forward_tokens_train"],
                forward_tokens_history=global_counters["forward_tokens_history"],
                forward_tokens_current=global_counters["forward_tokens_current"],
                fwd_passes_train=global_counters["fwd_passes_train"],
                fwd_passes_history=global_counters["fwd_passes_history"],
                fwd_passes_current=global_counters["fwd_passes_current"],
                method=str(progress["method"]),
            )
        )
        self._logger.flush()

    def post_train_batch(self) -> None:  # pragma: no cover - requires olmo_core
        self._consume_train_compute()

    def state_dict(self) -> Dict[str, Any]:
        return {"version": self.STATE_VERSION}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("version", self.STATE_VERSION) != self.STATE_VERSION:
            raise ValueError("unsupported raw metrics callback state version")


def build_metrics_payload(
    *,
    run_id: str,
    method: str,
    seed: int,
    ts_config: Mapping[str, Any],
    t0_tokens: int,
    order_id: str,
    init_id: str,
    spec_id: Optional[str] = None,
    n_params: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the v2 ledger identity that must match across both arms."""
    return empty_metrics_payload(
        run_id=run_id,
        method=method,
        seed=seed,
        k=float(ts_config["k"]),
        t0_tokens=int(t0_tokens),
        alpha_start=float(ts_config["alpha_start"]),
        alpha_end=float(ts_config["alpha_end"]),
        n_params=n_params,
        experiment_id=run_id,
        order_id=order_id,
        init_id=init_id,
        spec_id=spec_id,
    )

