"""Unit tests for evaluator contracts without importing the OLMo runtime."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _module(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture()
def evaluator_module(monkeypatch):
    class DummyEvaluatorType:
        downstream = "downstream"

    class DummyAttentionBackendName:
        torch = "torch"

    stubs = {
        "olmo": _module("olmo"),
        "olmo.config": _module(
            "olmo.config",
            EvaluatorConfig=object,
            EvaluatorType=DummyEvaluatorType,
            TrainConfig=object,
        ),
        "olmo.eval": _module("olmo.eval", build_evaluator=lambda *a, **k: None),
        "olmo.tokenizer": _module("olmo.tokenizer", Tokenizer=object),
        "olmo.torch_util": _module("olmo.torch_util", get_local_rank=lambda: 0),
        "olmo.util": _module(
            "olmo.util",
            prepare_cli_environment=lambda: None,
            add_cached_path_clients=lambda: None,
        ),
        "olmo_core": _module("olmo_core"),
        "olmo_core.nn": _module("olmo_core.nn"),
        "olmo_core.nn.attention": _module(
            "olmo_core.nn.attention",
            AttentionBackendName=DummyAttentionBackendName,
        ),
        "olmo_core.nn.transformer": _module(
            "olmo_core.nn.transformer",
            TransformerConfig=object,
        ),
        "olmo_core.distributed": _module("olmo_core.distributed"),
        "olmo_core.distributed.checkpoint": _module(
            "olmo_core.distributed.checkpoint",
            unshard_checkpoint=lambda **kwargs: None,
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)

    root = Path(__file__).resolve().parents[4]
    path = root / "scripts" / "farmshare" / "task_loss" / "eval_task_loss_olmo_core.py"
    spec = importlib.util.spec_from_file_location("_unit_eval_task_loss", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_disable = torch.compiler.disable
    try:
        spec.loader.exec_module(module)
    finally:
        torch.compiler.disable = original_disable
    return module


def test_macro_uses_exactly_twenty_raw_labels(evaluator_module):
    raw = {
        label: float(index + 1)
        for index, label in enumerate(evaluator_module.TASK_LABELS)
    }
    raw["unrelated_derived_metric"] = 10_000.0
    aggregated = evaluator_module._aggregate_task_loss_results(raw)
    assert aggregated["macro_mean_task_loss_bpb"] == sum(range(1, 21)) / 20

    partial = dict(list(raw.items())[:19])
    assert "macro_mean_task_loss_bpb" not in evaluator_module._aggregate_task_loss_results(
        partial
    )


def test_pause_eval_reload_restores_before_soft_failure(evaluator_module, monkeypatch):
    events: list[str] = []

    class FakeDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def barrier():
            events.append("barrier")

    monkeypatch.setattr(evaluator_module, "dist", FakeDist)

    def fail_eval(*args, **kwargs):
        events.append("eval")
        raise RuntimeError("eval failed")

    monkeypatch.setattr(evaluator_module, "run_task_loss_eval_distributed", fail_eval)
    restored, payload = evaluator_module.pause_eval_reload_distributed(
        Path("step125"),
        Path("result.json"),
        "unit",
        release_train_state=lambda: events.append("release"),
        reload_train_state=lambda: events.append("reload") or "train-module",
        strict=False,
    )
    assert restored == "train-module"
    assert payload is None
    assert events.index("release") < events.index("eval") < events.index("reload")


def test_pause_eval_reload_strict_raises_after_restore(evaluator_module, monkeypatch):
    events: list[str] = []

    class FakeDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def barrier():
            pass

    monkeypatch.setattr(evaluator_module, "dist", FakeDist)
    monkeypatch.setattr(
        evaluator_module,
        "run_task_loss_eval_distributed",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )
    with pytest.raises(RuntimeError, match="training state was restored"):
        evaluator_module.pause_eval_reload_distributed(
            Path("step125"),
            Path("result.json"),
            "unit",
            release_train_state=None,
            reload_train_state=lambda: events.append("reload") or object(),
            strict=True,
        )
    assert events == ["reload"]
