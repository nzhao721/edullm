"""Versioned metrics payload + JSONL logger for REL+EMA runs (train-only)."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


METRICS_SCHEMA_VERSION = 2
"""Latest persisted metrics schema version."""

# ``selected_tokens`` records loss-bearing tokens. It is intentionally distinct
# from the forward token counters because token masking does not remove a
# training forward/backward pass.
RAW_COMPUTE_COUNTERS = (
    "selected_tokens",
    "forward_tokens_train",
    "forward_tokens_history",
    "forward_tokens_current",
)

_FORWARD_PASS_COUNTERS = (
    "fwd_passes_train",
    "fwd_passes_history",
    "fwd_passes_current",
)


class MetricsSchemaError(ValueError):
    """Raised when metrics cannot support a fair-compute analysis."""


def metrics_schema_version(payload: Mapping[str, Any]) -> int:
    """Return the schema version, treating unversioned data as legacy v1."""
    if not isinstance(payload, Mapping):
        raise MetricsSchemaError("metrics payload must be a mapping")
    version = payload.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise MetricsSchemaError("schema_version must be an integer")
    if version not in (1, METRICS_SCHEMA_VERSION):
        raise MetricsSchemaError(
            f"unsupported metrics schema_version={version}; "
            f"expected 1 or {METRICS_SCHEMA_VERSION}"
        )
    return version


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MetricsSchemaError(f"{field_name} must be a non-negative integer")
    return value


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricsSchemaError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MetricsSchemaError(f"{field_name} must be a finite number")
    return number


def raw_compute_snapshot(compute: Mapping[str, Any]) -> Dict[str, int]:
    """Copy and validate observed v2 raw token counters.

    Legacy estimates such as ``total_compute_tokens`` are deliberately excluded.
    A v2 comparison can only use recorded counters.
    """
    if not isinstance(compute, Mapping):
        raise MetricsSchemaError("compute must be a mapping")
    snapshot: Dict[str, int] = {}
    for key in RAW_COMPUTE_COUNTERS:
        if key not in compute:
            raise MetricsSchemaError(f"v2 compute is missing required counter {key!r}")
        snapshot[key] = _non_negative_int(compute[key], field_name=f"compute.{key}")
    if snapshot["selected_tokens"] > snapshot["forward_tokens_train"]:
        raise MetricsSchemaError(
            "compute.selected_tokens cannot exceed compute.forward_tokens_train"
        )
    return snapshot


def _validate_contract_string(contract: Mapping[str, Any], key: str) -> None:
    value = contract.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MetricsSchemaError(f"comparison-ready v2 metrics require {key!r}")


def validate_metrics_payload(
    payload: Mapping[str, Any],
    *,
    comparison_ready: bool = False,
) -> int:
    """Validate metrics and return its schema version.

    Train-only v2 payloads bind experiment/order/init identity and raw train
    compute. Scientific arm comparison (test loss) is deferred to a later eval
    pass on checkpoints; ``comparison_ready`` still requires experiment identity
    so arms can be matched before that later eval.
    """
    version = metrics_schema_version(payload)
    if version == 1:
        if comparison_ready:
            raise MetricsSchemaError(
                "legacy v1 metrics cannot satisfy v2 comparison contracts; "
                "use the legacy compatibility path instead"
            )
        return version

    compute = payload.get("compute")
    raw_compute_snapshot(compute if isinstance(compute, Mapping) else {})
    experiment = payload.get("experiment")
    if not isinstance(experiment, Mapping):
        raise MetricsSchemaError("v2 metrics require an experiment contract")

    if comparison_ready:
        for key in ("experiment_id", "order_id", "init_id"):
            _validate_contract_string(experiment, key)
    return version


def migrate_v1_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a v2-shaped v1 migration without inventing scientific contracts."""
    if metrics_schema_version(payload) != 1:
        raise MetricsSchemaError("migrate_v1_payload only accepts legacy schema v1 payloads")

    migrated = copy.deepcopy(dict(payload))
    legacy_compute = migrated.get("compute") or {}
    if not isinstance(legacy_compute, Mapping):
        raise MetricsSchemaError("legacy compute must be a mapping")
    compute = dict(legacy_compute)
    train_tokens = compute.get("forward_tokens_train", compute.get("train_tokens", 0))
    _non_negative_int(train_tokens, field_name="compute.train_tokens")
    scoring_tokens = compute.get("scoring_overhead_tokens", 0)
    _non_negative_int(scoring_tokens, field_name="compute.scoring_overhead_tokens")
    selected_tokens = compute.get("selected_tokens", train_tokens)
    _non_negative_int(selected_tokens, field_name="compute.selected_tokens")
    if selected_tokens > train_tokens:
        raise MetricsSchemaError("legacy selected_tokens cannot exceed training tokens")
    compute.setdefault("selected_tokens", selected_tokens)
    compute.setdefault("forward_tokens_train", train_tokens)
    compute.setdefault("forward_tokens_history", scoring_tokens)
    compute.setdefault("forward_tokens_current", 0)
    for key in _FORWARD_PASS_COUNTERS:
        compute.setdefault(key, 0)
    # Drop obsolete validation counters if present on legacy payloads.
    compute.pop("forward_tokens_validation", None)
    compute.pop("fwd_passes_validation", None)

    migrated["schema_version"] = METRICS_SCHEMA_VERSION
    migrated["compute"] = compute
    migrated["experiment"] = {
        "experiment_id": migrated.get("run_id"),
        "order_id": None,
        "init_id": None,
    }
    migrated.pop("validation", None)
    migrated.pop("val_loss_curve", None)
    migrated["migration"] = {
        "source_schema_version": 1,
        "raw_counter_attribution": (
            "Legacy scoring_overhead_tokens was assigned to forward_tokens_history; "
            "order/init contracts remain unknown; held-out validation was removed."
        ),
    }
    validate_metrics_payload(migrated)
    return migrated


@dataclass
class StepMetrics:
    step: int
    tokens_seen: int
    k: float
    alpha: float
    warmup: bool
    selected_frac: float
    mean_rel_kept: Optional[float] = None
    mean_rel_dropped: Optional[float] = None
    train_loss: Optional[float] = None
    wall_time_s: Optional[float] = None
    selected_tokens: Optional[int] = None
    forward_tokens_train: Optional[int] = None
    forward_tokens_history: Optional[int] = None
    forward_tokens_current: Optional[int] = None
    fwd_passes_train: Optional[int] = None
    fwd_passes_history: Optional[int] = None
    fwd_passes_current: Optional[int] = None
    method: str = "rel_ema"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra", {}) or {}
        d.update(extra)
        return d


def empty_metrics_payload(
    *,
    run_id: str,
    method: str,
    seed: int,
    k: float,
    t0_tokens: int,
    alpha_start: float,
    alpha_end: float,
    n_params: Optional[int] = None,
    experiment_id: Optional[str] = None,
    order_id: Optional[str] = None,
    init_id: Optional[str] = None,
    spec_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return an empty v2 train-only payload.

    Callers must bind frozen data order and initialization before arms can be
    matched. Test-loss evaluation is deferred to a later checkpoint pass.
    """
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "seed": seed,
        "k": k,
        "t0_tokens": t0_tokens,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"n_params": n_params},
        "experiment": {
            "experiment_id": experiment_id or run_id,
            "order_id": order_id,
            "init_id": init_id,
            "spec_id": spec_id,
        },
        "train_loss_curve": [],
        "selection_curve": [],
        "benchmarks": {},
        "compute": {
            "scoring_overhead_tokens": 0,
            "train_tokens": 0,
            "total_compute_tokens": 0,
            "selected_tokens": 0,
            "forward_tokens_train": 0,
            "forward_tokens_history": 0,
            "forward_tokens_current": 0,
            "fwd_passes_train": 0,
            "fwd_passes_history": 0,
            "fwd_passes_current": 0,
            "wall_time_s": 0.0,
        },
        "bootstrap_ci": None,
        "steps": [],
    }


class MetricLogger:
    def __init__(self, path: Path, payload: Dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if metrics_schema_version(payload) == 1:
            self.payload = migrate_v1_payload(payload)
        else:
            validate_metrics_payload(payload)
            self.payload = payload
        # Hard-remove residue: drop any leftover held-out fields from older ledgers.
        self.payload.pop("validation", None)
        self.payload.pop("val_loss_curve", None)
        self.payload.pop("final_val_loss", None)
        self.payload.pop("val_loss_samples", None)
        compute = self.payload.get("compute")
        if isinstance(compute, dict):
            compute.pop("forward_tokens_validation", None)
            compute.pop("fwd_passes_validation", None)
        self._jsonl = self.path.with_suffix(".jsonl")

    def log_step(self, metrics: StepMetrics) -> None:
        row = metrics.to_dict()
        self.payload["steps"].append(row)
        self.payload["selection_curve"].append(
            {
                "step": metrics.step,
                "tokens": metrics.tokens_seen,
                "selected_frac": metrics.selected_frac,
                "alpha": metrics.alpha,
                "warmup": metrics.warmup,
                "mean_rel_kept": metrics.mean_rel_kept,
                "mean_rel_dropped": metrics.mean_rel_dropped,
            }
        )
        if metrics.train_loss is not None:
            self.payload["train_loss_curve"].append(
                {"step": metrics.step, "tokens": metrics.tokens_seen, "loss": metrics.train_loss}
            )
        with self._jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def add_compute(
        self,
        *,
        train_tokens: int = 0,
        scoring_overhead_tokens: int = 0,
        wall_time_s: float = 0.0,
        selected_tokens: int = 0,
        forward_tokens_train: int = 0,
        forward_tokens_history: int = 0,
        forward_tokens_current: int = 0,
        fwd_passes_train: int = 0,
        fwd_passes_history: int = 0,
        fwd_passes_current: int = 0,
    ) -> None:
        c = self.payload.setdefault("compute", {})
        increments = {
            "train_tokens": train_tokens,
            "scoring_overhead_tokens": scoring_overhead_tokens,
            "selected_tokens": selected_tokens,
            "forward_tokens_train": forward_tokens_train,
            "forward_tokens_history": forward_tokens_history,
            "forward_tokens_current": forward_tokens_current,
            "fwd_passes_train": fwd_passes_train,
            "fwd_passes_history": fwd_passes_history,
            "fwd_passes_current": fwd_passes_current,
        }
        for key, val in increments.items():
            increment = _non_negative_int(val, field_name=f"add_compute.{key}")
            c[key] = int(c.get(key, 0)) + increment
        c["wall_time_s"] = float(c.get("wall_time_s", 0.0)) + float(wall_time_s)
        c["total_compute_tokens"] = int(c.get("train_tokens", 0)) + int(
            c.get("scoring_overhead_tokens", 0)
        )

    def flush(self) -> None:
        validate_metrics_payload(self.payload)
        self.path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
