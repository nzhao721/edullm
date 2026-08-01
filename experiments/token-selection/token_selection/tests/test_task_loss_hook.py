"""Focused launch/failure semantics for the shared task-loss hook."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from token_selection.olmo_ext import task_loss_hook as hook


def _inputs(tmp_path):
    checkpoint = tmp_path / "step125"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"state")
    script = tmp_path / "eval.py"
    script.write_text("# unit-test evaluator\n", encoding="utf-8")
    return checkpoint, script, tmp_path / "step125_task_loss.json"


def test_strict_forces_sync_and_propagates_nonzero(monkeypatch, tmp_path):
    checkpoint, script, out = _inputs(tmp_path)
    seen = {}

    def fake_run(cmd, *, check, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return SimpleNamespace(returncode=7)

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(hook.subprocess, "run", fake_run)
    with pytest.raises(hook.TaskLossLaunchError, match="exited 7"):
        hook.trigger_task_loss_eval(
            checkpoint,
            run_name="unit",
            out_path=out,
            eval_script=script,
            async_=True,
            strict=True,
        )
    assert seen["cmd"][0]
    assert "RANK" not in seen["env"]


def test_strict_requires_output_after_success(monkeypatch, tmp_path):
    checkpoint, script, out = _inputs(tmp_path)
    monkeypatch.setattr(
        hook.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    with pytest.raises(hook.TaskLossLaunchError, match="did not write"):
        hook.trigger_task_loss_eval(
            checkpoint,
            run_name="unit",
            out_path=out,
            eval_script=script,
            strict=True,
        )


def test_strict_surfaces_missing_script(tmp_path):
    checkpoint, _, out = _inputs(tmp_path)
    with pytest.raises(hook.TaskLossLaunchError, match="script not found"):
        hook.trigger_task_loss_eval(
            checkpoint,
            run_name="unit",
            out_path=out,
            eval_script=tmp_path / "missing.py",
            strict=True,
        )


def test_non_strict_sync_failure_remains_warning_only(monkeypatch, tmp_path):
    checkpoint, script, out = _inputs(tmp_path)
    monkeypatch.setattr(
        hook.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=9),
    )
    assert (
        hook.trigger_task_loss_eval(
            checkpoint,
            run_name="unit",
            out_path=out,
            eval_script=script,
            async_=False,
            strict=False,
        )
        is None
    )
