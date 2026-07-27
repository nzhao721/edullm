"""Tests for versioned fair-compute metrics telemetry (train-only)."""

from __future__ import annotations

import json

import pytest

from token_selection.olmo_ext.metrics import (
    METRICS_SCHEMA_VERSION,
    MetricLogger,
    MetricsSchemaError,
    StepMetrics,
    empty_metrics_payload,
    migrate_v1_payload,
    raw_compute_snapshot,
    validate_metrics_payload,
)


def _payload(**kwargs):
    return empty_metrics_payload(
        run_id="fair-rel-smoke",
        method="full",
        seed=7,
        k=0.6,
        t0_tokens=0,
        alpha_start=0.999,
        alpha_end=0.995,
        n_params=10,
        **kwargs,
    )


def test_v2_payload_distinguishes_selected_and_forward_tokens():
    payload = _payload()

    assert payload["schema_version"] == METRICS_SCHEMA_VERSION
    compute = payload["compute"]
    assert compute["selected_tokens"] == 0
    assert compute["forward_tokens_train"] == 0
    assert "forward_tokens_validation" not in compute
    assert "validation" not in payload
    validate_metrics_payload(payload)


def test_logger_records_train_steps_and_cumulative_compute(tmp_path):
    payload = _payload(
        order_id="frozen-order-v1",
        init_id="checkpoint-sha",
    )
    logger = MetricLogger(tmp_path / "metrics.json", payload)
    logger.add_compute(
        train_tokens=12,
        selected_tokens=8,
        forward_tokens_train=12,
        forward_tokens_history=3,
    )
    logger.log_step(
        StepMetrics(
            step=0,
            tokens_seen=12,
            k=0.6,
            alpha=0.999,
            warmup=True,
            selected_frac=0.5,
            train_loss=2.5,
            selected_tokens=8,
            forward_tokens_train=12,
            forward_tokens_history=3,
            forward_tokens_current=0,
            method="full",
        )
    )
    logger.flush()

    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert persisted["compute"]["forward_tokens_train"] == 12
    assert persisted["compute"]["forward_tokens_history"] == 3
    assert persisted["train_loss_curve"][0]["loss"] == 2.5
    assert "validation" not in persisted


def test_comparison_ready_requires_experiment_identity():
    payload = _payload()
    with pytest.raises(MetricsSchemaError, match="order_id"):
        validate_metrics_payload(payload, comparison_ready=True)

    ready = _payload(order_id="o", init_id="i", experiment_id="e")
    assert validate_metrics_payload(ready, comparison_ready=True) == METRICS_SCHEMA_VERSION


def test_v2_fails_closed_for_missing_raw_counters():
    payload = _payload(order_id="o", init_id="i")
    payload["compute"].pop("forward_tokens_train")
    with pytest.raises(MetricsSchemaError, match="forward_tokens_train"):
        validate_metrics_payload(payload)


def test_raw_compute_snapshot_rejects_selected_gt_train():
    with pytest.raises(MetricsSchemaError, match="selected_tokens"):
        raw_compute_snapshot(
            {
                "selected_tokens": 5,
                "forward_tokens_train": 4,
                "forward_tokens_history": 0,
                "forward_tokens_current": 0,
            }
        )


def test_migrate_v1_drops_validation_and_keeps_train_counters():
    v1 = {
        "run_id": "legacy",
        "method": "full",
        "seed": 1,
        "k": 0.6,
        "t0_tokens": 0,
        "alpha_start": 0.999,
        "alpha_end": 0.995,
        "compute": {
            "train_tokens": 100,
            "scoring_overhead_tokens": 20,
            "selected_tokens": 80,
        },
        "validation": {"contract": {}, "records": []},
        "val_loss_curve": [{"loss": 1.0}],
    }
    migrated = migrate_v1_payload(v1)
    assert migrated["schema_version"] == METRICS_SCHEMA_VERSION
    assert migrated["compute"]["forward_tokens_train"] == 100
    assert migrated["compute"]["forward_tokens_history"] == 20
    assert "validation" not in migrated
    assert "val_loss_curve" not in migrated
    assert "forward_tokens_validation" not in migrated["compute"]
