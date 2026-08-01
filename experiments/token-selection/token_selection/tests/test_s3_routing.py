"""S3 is restricted to pre-tokenized and bootstrap inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import token_selection.scripts.train_data_resolve as rtd_mod
from token_selection.scripts import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "run_rho_10b.yaml"


def _run_cfg() -> dict:
    return {
        "data": {"dataset_id": "pretrain/regmix-10b", "dataset_version": "v1"},
        "s3": {
            "dataset_bucket": "edullm-data",
        },
    }


def _fake_resolved(**overrides):
    base = {
        "dataset_id": "pretrain/regmix-10b",
        "version": "v1",
        "tokens_uri": "s3://edullm-data/pretrain/regmix-10b/v1/tokens",
        "paths": [
            "s3://edullm-data/pretrain/regmix-10b/v1/tokens/arxiv/train-00000.u32le.bin"
        ],
        "dtype": "uint32",
        "numpy_dtype": None,
        "rows": 100,
        "header_bytes": 0,
        "byte_order": "little",
        "resolved": SimpleNamespace(),
    }
    base.update(overrides)
    return base


def test_resolve_tokens_s3_rejects_legacy_and_missing_id():
    with pytest.raises(ValueError, match="edullm-datasets"):
        rtd_mod.resolve_tokens_s3(
            {"data": {"tokens_s3": "s3://edullm-datasets/regmix/regmix-10b/tokenized"}}
        )
    with pytest.raises(ValueError, match="dataset_id"):
        rtd_mod.resolve_tokens_s3({"data": {}})
    with pytest.raises(ValueError, match="REPLACE_ME|placeholder"):
        rtd_mod.resolve_tokens_s3({"data": {"tokens_s3": "s3://REPLACE_ME/tokens"}})


def test_resolve_tokens_s3_uses_edullm_data(monkeypatch):
    monkeypatch.setattr(
        rtd_mod,
        "resolve_train_dataset",
        lambda cfg, s3=None, split="train": _fake_resolved(),
    )
    assert rtd_mod.resolve_tokens_s3(_run_cfg()) == (
        "s3://edullm-data/pretrain/regmix-10b/v1/tokens"
    )


def test_run_config_points_at_edullm_data_corpus(monkeypatch):
    """The production config must name a published edullm-data dataset_id."""
    cfg = load_config(CONFIG_PATH)
    assert cfg["data"]["dataset_id"] == "pretrain/regmix-10b"
    assert "edullm-datasets" not in str(cfg.get("data") or {})
    assert cfg["data"]["tokenizer"] == "allenai/dolma2-tokenizer"
    assert "checkpoint_bucket" not in cfg["s3"]
    assert "prefix" not in cfg["s3"]

    fake_split = SimpleNamespace(
        paths=[
            "s3://edullm-data/pretrain/regmix-10b/v1/tokens/arxiv/train-00000.u32le.bin"
        ],
        dtype="uint32",
        rows=100,
        header_bytes=0,
        byte_order="little",
        numpy_dtype=None,
    )

    class _FakeS3:
        @staticmethod
        def default():
            return object()

    monkeypatch.setattr(
        rtd_mod,
        "_require_edullm_data",
        lambda: (
            lambda *a, **k: fake_split,
            lambda *a, **k: "v1",
            _FakeS3,
        ),
    )
    resolved = rtd_mod.resolve_train_dataset(cfg)
    assert resolved["tokens_uri"] == "s3://edullm-data/pretrain/regmix-10b/v1/tokens"
    assert rtd_mod.resolve_tokens_s3(cfg) == resolved["tokens_uri"]


def test_s3_artifact_export_helper_is_removed():
    helper = Path(__file__).resolve().parents[1] / "olmo_ext" / "s3_export.py"
    assert not helper.exists()
