"""Unit tests for SmolLM2-parity W&B helpers (no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from token_selection.olmo_ext import wandb_logging as wl


def _raw_labels(offset: float = 0.0) -> dict[str, float]:
    return {
        label: float(index + 1) + offset
        for index, label in enumerate(wl.TASK_LOSS_RAW_LABELS)
    }


def test_resolve_run_name_prefers_explicit():
    assert wl.resolve_run_name(explicit="control-x", arm="control", run_id="ignored") == "control-x"
    assert wl.resolve_run_name(run_id="rho-1-regmix10b-v1", arm="rho-1") == "rho-1-regmix10b-v1"
    assert wl.resolve_run_name(arm="blade", method="blade") == "blade-blade"


def test_task_loss_metrics_flatten():
    labels = _raw_labels()
    payload = {
        "macro_mean": 999.0,  # historical buggy macro must not be trusted
        "macro_mean_accuracy": 0.4,
        "labels": labels,
        "accuracy_labels": {"arc_easy": 0.5},
        "task_families": {"qa": 1.2},
        "accuracy_families": {"qa": 0.45},
    }
    m = wl.task_loss_metrics(payload)
    assert m["eval/macro_bpb"] == sum(labels.values()) / 20
    assert m["eval/macro_acc"] == 0.4
    assert m[f"eval/bpb/{wl.TASK_LOSS_RAW_LABELS[0]}"] == 1.0
    assert m["eval/family_bpb/qa"] == 1.2


def test_task_loss_metrics_accept_legacy_nested_shape():
    labels = _raw_labels(0.5)
    payload = {
        "task_loss_bpb": {
            **labels,
            "core_avg_rc_5shot_bpb": 4.2,
            "macro_mean_task_loss_bpb": -1.0,
        }
    }
    metrics = wl.task_loss_metrics(payload)
    assert wl.task_loss_payload_complete(payload) is True
    assert metrics["eval/macro_bpb"] == sum(labels.values()) / 20
    assert metrics["eval/bpb/core_avg_rc_5shot_bpb"] == 4.2
    assert "eval/bpb/macro_mean_task_loss_bpb" not in metrics


def test_partial_task_loss_does_not_emit_macro():
    payload = {
        "macro_mean": 1.25,
        "labels": {wl.TASK_LOSS_RAW_LABELS[0]: 1.25},
    }
    assert wl.task_loss_payload_complete(payload) is False
    metrics = wl.task_loss_metrics(payload)
    assert "eval/macro_bpb" not in metrics
    assert metrics[f"eval/bpb/{wl.TASK_LOSS_RAW_LABELS[0]}"] == 1.25


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

        def log_artifact(self, art, aliases=None):
            self.arts.append(art)
            return art

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
    path = results / "step0000125_task_loss.json"
    path.write_text(
        json.dumps({"macro_mean": 2.0, "labels": {wl.TASK_LOSS_RAW_LABELS[0]: 2.1}}),
        encoding="utf-8",
    )
    assert poller.poll() == []  # valid JSON, but not a contract-complete suite
    labels = _raw_labels()
    path.write_text(
        json.dumps({"macro_mean": 2.0, "labels": labels}),
        encoding="utf-8",
    )
    assert poller.poll() == [125]
    assert poller.poll() == []  # idempotent
    assert run.logs[0][0] == 125
    assert run.logs[0][1]["eval/macro_bpb"] == sum(labels.values()) / 20
    assert len(run.arts) == 1


def test_production_checkpoint_upload_waits_and_fails_closed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step125"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"state")

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name

        def add_dir(self, path):
            self.path = path

    class FailedUpload:
        def wait(self):
            raise RuntimeError("upload failed")

    class FakeRun:
        name = "unit-run"

        def log(self, *_args, **_kwargs):
            pass

        def log_artifact(self, _artifact, aliases=None):
            assert aliases == ["latest", "step-0000125"]
            return FailedUpload()

    monkeypatch.setattr(wl, "_wandb", type("W", (), {"Artifact": FakeArtifact})())
    with pytest.raises(wl.WandbArtifactError, match="required W&B checkpoint"):
        wl.wandb_log_checkpoint(FakeRun(), checkpoint, step=125, strict=True)


def test_production_artifact_callback_requires_active_run(tmp_path, monkeypatch):
    monkeypatch.setattr(wl, "_wandb", type("W", (), {"run": None})())
    callback = wl.WandbArtifactsCallback(
        results_dir=tmp_path / "eval",
        save_folder=tmp_path / "checkpoints",
        total_steps=2360,
        production=True,
        wandb_mode="online",
    )
    callback.trainer = type("Trainer", (), {"callbacks": {}})()
    with pytest.raises(wl.WandbArtifactError, match="no active run"):
        callback._maybe_log_checkpoint(125)


def test_wandb_run_from_trainer_falls_back_to_global_run(monkeypatch):
    active = object()
    monkeypatch.setattr(wl, "_wandb", type("W", (), {"run": active})())
    trainer = type("Trainer", (), {"callbacks": {}})()
    assert wl.wandb_run_from_trainer(trainer) is active


def test_restore_checkpoint_artifact_to_local_scratch(tmp_path, monkeypatch):
    source = tmp_path / "download"
    source.mkdir()
    (source / "state.pt").write_bytes(b"state")
    (source / "run_fingerprint.json").write_text("{}", encoding="utf-8")

    class Artifact:
        metadata = {"step": 125}

        def download(self, root):
            target = Path(root)
            for item in source.iterdir():
                (target / item.name).write_bytes(item.read_bytes())
            return str(target)

    class Api:
        def artifact(self, ref, type):
            assert ref.endswith(":latest")
            assert type == "model"
            return Artifact()

    save = tmp_path / "scratch" / "checkpoints"
    restored = wl.restore_checkpoint_artifact(
        "team/token-selection/unit-checkpoint:latest", save, api=Api()
    )
    assert restored == save / "step125"
    assert (restored / "state.pt").is_file()
    assert (save / "run_fingerprint.json").is_file()


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
