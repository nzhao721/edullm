"""Tests for post-hoc EMA checkpoint merge (α=0.8 convention)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ema_merge_checkpoints import (
    DEFAULT_ALPHA,
    DEFAULT_EMA_STEPS,
    ema_merge_state_dicts,
    ema_weights_closed_form,
    parse_args,
    write_ema_checkpoint,
)


def test_ema_steps_match_plan():
    assert DEFAULT_EMA_STEPS == (2000, 2125, 2250, 2384)
    assert DEFAULT_ALPHA == 0.8


def test_task_loss_default_on(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    args = parse_args(["--checkpoints-root", "/tmp/ckpts", "--arm-id", "control"])
    assert args.task_loss is True
    assert args.wandb_mode == "online"
    assert args.allow_local_only is False
    args_off = parse_args(
        ["--checkpoints-root", "/tmp/ckpts", "--arm-id", "control", "--no-task-loss"]
    )
    assert args_off.task_loss is False


def test_ema_checkpoint_publication_is_wandb_only_and_awaited():
    source = (_ROOT / "ema_merge_checkpoints.py").read_text(encoding="utf-8")
    assert "def upload_ema_artifacts" in source
    assert "checkpoint.add_dir" in source
    assert "wait()" in source
    assert "WANDB_API_KEY" in source
    assert "--allow-local-only" in source
    assert "sync_to_s3" not in source
    assert "edullm-checkpoints" not in source


def test_closed_form_weights_four_checkpoints():
    w = ema_weights_closed_form(4, alpha=0.8)
    # w0=0.512, w1=0.128, w2=0.16, w3=0.2  (before normalize — already sum to 1)
    assert abs(sum(w) - 1.0) < 1e-12
    assert abs(w[0] - 0.512) < 1e-12
    assert abs(w[1] - 0.128) < 1e-12
    assert abs(w[2] - 0.16) < 1e-12
    assert abs(w[3] - 0.2) < 1e-12


def test_recursive_merge_matches_closed_form():
    # Scalar tensors: c0=0, c1=1, c2=2, c3=3 → avg = sum w_i * i
    sds = [{"w": torch.tensor([float(i)])} for i in range(4)]
    merged = ema_merge_state_dicts(sds, alpha=0.8)
    weights = ema_weights_closed_form(4, alpha=0.8)
    expected = sum(weights[i] * float(i) for i in range(4))
    assert abs(float(merged["w"].item()) - expected) < 1e-6


def test_avg_update_convention():
    # Single-step: avg ← 0.8*avg + 0.2*newest
    a = {"w": torch.tensor([10.0])}
    b = {"w": torch.tensor([20.0])}
    out = ema_merge_state_dicts([a, b], alpha=0.8)
    assert abs(float(out["w"].item()) - (0.8 * 10 + 0.2 * 20)) < 1e-6


def test_write_ema_checkpoint_roundtrip(tmp_path: Path):
    model = {"layer.weight": torch.randn(4, 4)}
    template = {
        "train_module": {"model": model, "optim": {"state": {}}},
        "architecture": "olmo2_370M",
        "config_name": "OLMo-2-370M-scratch",
        "checkpoint_format": "full_state_dict_v1",
        "meta": {},
        "run_id": "control",
    }
    out = tmp_path / "step2384-ema"
    write_ema_checkpoint(
        out,
        model_sd=model,
        template_ckpt=template,
        steps=list(DEFAULT_EMA_STEPS),
        alpha=0.8,
        arm_id="control",
    )
    assert (out / "state.pt").is_file()
    ckpt = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
    assert ckpt["meta"]["ema_merge"]["alpha"] == 0.8
    assert ckpt["meta"]["ema_merge"]["steps"] == list(DEFAULT_EMA_STEPS)
    assert ckpt["train_module"]["optim"] is None
