"""Launch must pin idle physical GPU(s) and refuse anything else."""

from __future__ import annotations

import os

import pytest

from token_selection.scripts.train_olmo_template import pin_cuda_visible_devices


def test_pin_requires_config(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit, match="cuda_visible_devices is required"):
        pin_cuda_visible_devices({"train": {}})


def test_pin_accepts_multi_index(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    seen = []

    def fake_check_output(cmd, text=True, stderr=None):  # noqa: ARG001
        idx = cmd[2]
        seen.append(idx)
        return f"{idx}, GPU-{idx}, 0\n"

    monkeypatch.setattr(
        "token_selection.scripts.train_olmo_template.subprocess.check_output",
        fake_check_output,
    )
    pinned = pin_cuda_visible_devices({"train": {"cuda_visible_devices": "6,7"}})
    assert pinned == "6,7"
    assert seen == ["6", "7"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


def test_pin_rejects_conflicting_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit, match="conflicting pin"):
        pin_cuda_visible_devices({"train": {"cuda_visible_devices": "1"}})


def test_pin_sets_env_and_accepts_idle_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fake_check_output(cmd, text=True, stderr=None):  # noqa: ARG001
        assert cmd[:3] == ["nvidia-smi", "-i", "0"]
        return "0, GPU-deadbeef, 0\n"

    monkeypatch.setattr(
        "token_selection.scripts.train_olmo_template.subprocess.check_output",
        fake_check_output,
    )
    pinned = pin_cuda_visible_devices({"train": {"cuda_visible_devices": "0"}})
    assert pinned == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_pin_refuses_busy_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fake_check_output(cmd, text=True, stderr=None):  # noqa: ARG001
        return "0, GPU-deadbeef, 4096\n"

    monkeypatch.setattr(
        "token_selection.scripts.train_olmo_template.subprocess.check_output",
        fake_check_output,
    )
    with pytest.raises(SystemExit, match="not idle"):
        pin_cuda_visible_devices({"train": {"cuda_visible_devices": "0"}})


def test_pin_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    def fake_check_output(cmd, text=True, stderr=None):  # noqa: ARG001
        idx = cmd[2]
        return f"{idx}, GPU-{idx}, 0\n"

    monkeypatch.setattr(
        "token_selection.scripts.train_olmo_template.subprocess.check_output",
        fake_check_output,
    )
    pinned = pin_cuda_visible_devices({"train": {"cuda_visible_devices": ""}})
    assert pinned == "4,5"
