"""Unit tests for SmolLM2-parity W&B helpers (no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from token_selection.olmo_ext import wandb_logging as wl


def test_resolve_run_name_prefers_explicit():
    assert wl.resolve_run_name(explicit="control-x", arm="control", run_id="ignored") == "control-x"
    assert wl.resolve_run_name(run_id="rho-1-regmix10b-v1", arm="rho-1") == "rho-1-regmix10b-v1"
    assert wl.resolve_run_name(arm="blade", method="blade") == "blade-blade"


def test_task_loss_metrics_flatten():
    payload = {
        "macro_mean": 1.25,
        "macro_mean_accuracy": 0.4,
        "labels": {"arc_easy": 1.1, "hellaswag": 1.4},
        "accuracy_labels": {"arc_easy": 0.5},
        "task_families": {"qa": 1.2},
        "accuracy_families": {"qa": 0.45},
    }
    m = wl.task_loss_metrics(payload)
    assert m["eval/macro_bpb"] == 1.25
    assert m["eval/macro_acc"] == 0.4
    assert m["eval/bpb/arc_easy"] == 1.1
    assert m["eval/family_bpb/qa"] == 1.2


def test_wandb_enabled_requires_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    assert wl.wandb_enabled(mode="online", is_main=True) is False
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    # Still False if wandb package missing in CI; if present, True.
    got = wl.wandb_enabled(mode="online", is_main=True)
    if wl._wandb is None:
        assert got is False
    else:
        assert got is True
    assert wl.wandb_enabled(mode="disabled", is_main=True) is False
    assert wl.wandb_enabled(mode="online", is_main=False) is False


def test_eval_poller_logs_new_files(tmp_path, monkeypatch):
    class FakeRun:
        def __init__(self):
            self.logs = []
            self.arts = []

        def log(self, metrics, step=None):
            self.logs.append((step, dict(metrics)))

        def log_artifact(self, art):
            self.arts.append(art)

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    monkeypatch.setattr(wl, "_wandb", type("W", (), {"Artifact": FakeArtifact})())
    results = tmp_path / "tl"
    results.mkdir()
    run = FakeRun()
    poller = wl.WandbEvalPoller(results, run)
    assert poller.poll() == []
    (results / "step0000125_task_loss.json").write_text(
        json.dumps({"macro_mean": 2.0, "labels": {"a": 2.1}}),
        encoding="utf-8",
    )
    assert poller.poll() == [125]
    assert poller.poll() == []  # idempotent
    assert run.logs[0][0] == 125
    assert run.logs[0][1]["eval/macro_bpb"] == 2.0
    assert len(run.arts) == 1


def test_ensure_wandb_not_hard_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_DISABLED", "1")
    wl.ensure_wandb_not_hard_disabled()
    assert "WANDB_DISABLED" not in os.environ
    monkeypatch.setenv("WANDB_MODE", "disabled")
    wl.ensure_wandb_not_hard_disabled()
    assert os.environ.get("WANDB_DISABLED") == "1"


def test_add_wandb_argparse_options():
    import argparse

    p = argparse.ArgumentParser()
    wl.add_wandb_argparse_options(p, default_run_name="control-regmix10b-v1")
    ns = p.parse_args([])
    assert ns.wandb_project == wl.DEFAULT_WANDB_PROJECT
    assert ns.wandb_mode in {"online", "offline", "disabled"}
