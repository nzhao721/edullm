"""S3 routing for pre-tokenized inputs and per-run outputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import token_selection.scripts.train_data_resolve as rtd_mod
from token_selection.olmo_ext.s3_layout import (
    ARM_DIRS,
    CHECKPOINT_BUCKET,
    TOKEN_SEL_ROOT,
    arm_from_prefix,
    arm_prefix,
    arm_uri,
    default_s3_block,
)
from token_selection.scripts import load_config, s3_uri
from token_selection.scripts.sync_artifacts import sync_dir

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "run_rho_10b.yaml"


def _run_cfg() -> dict:
    return {
        "data": {"dataset_id": "pretrain/regmix-10b", "dataset_version": "v1"},
        "s3": {
            "dataset_bucket": "edullm-data",
            "checkpoint_bucket": "cb",
            "prefix": "proj/run",
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


def test_routing_per_run_outputs():
    cfg = _run_cfg()
    # Default bucket_key is checkpoint_bucket (never edullm-data for run outputs).
    assert s3_uri(cfg, "metrics") == "s3://cb/proj/run/metrics"
    assert s3_uri(cfg, bucket_key="checkpoint_bucket") == "s3://cb/proj/run"


def test_token_sel_layout_helpers():
    assert arm_prefix("blade") == "token-sel/blade"
    assert arm_uri("rho-1", "checkpoints") == (
        f"s3://{CHECKPOINT_BUCKET}/{TOKEN_SEL_ROOT}/rho-1/checkpoints"
    )
    assert arm_from_prefix("token-sel/rel-ema-exp") == "rel-ema-exp"
    assert arm_from_prefix("token-sel/rel-ema-refhq") == "rel-ema-refhq"
    block = default_s3_block("control")
    assert block["prefix"] == "token-sel/control"
    assert block["checkpoint_bucket"] == CHECKPOINT_BUCKET
    assert "dataset_bucket" not in block
    assert "blade" in ARM_DIRS and "rho-1" in ARM_DIRS
    assert "middle-ppl-token" in ARM_DIRS
    assert "reference" in ARM_DIRS
    with pytest.raises(ValueError, match="token-sel/"):
        arm_from_prefix("token-selection/rho-1")


def test_sync_dir_refuses_empty_mirrored_upload(tmp_path):
    empty = tmp_path / "tokens"
    with pytest.raises(SystemExit, match="Refusing mirrored"):
        sync_dir(
            empty,
            "s3://edullm-data/pretrain/regmix-10b/v1/tokens",
            profile="x",
            upload=True,
            mirror=True,
        )


def test_mirrored_token_download_keeps_the_derived_manifest(monkeypatch, tmp_path):
    """manifest.json is derived locally; a mirroring re-sync must not delete it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "token_selection.scripts.sync_artifacts._aws",
        lambda profile, args: calls.append(args),
    )
    sync_dir(
        tmp_path / "tokens",
        "s3://edullm-data/pretrain/regmix-10b/v1/tokens/",
        profile="x",
        upload=False,
        mirror=True,
        keep=("manifest.json",),
    )
    assert calls[0][:2] == ["s3", "sync"]
    assert "--delete" in calls[0]
    assert calls[0][calls[0].index("--exclude") + 1] == "manifest.json"


def test_run_config_points_at_edullm_data_corpus(monkeypatch):
    """The production config must name a published edullm-data dataset_id."""
    cfg = load_config(CONFIG_PATH)
    assert cfg["data"]["dataset_id"] == "pretrain/regmix-10b"
    assert "edullm-datasets" not in str(cfg.get("data") or {})
    assert cfg["data"]["tokenizer"] == "allenai/dolma2-tokenizer"
    assert cfg["s3"]["checkpoint_bucket"] == "edullm-checkpoints"
    assert cfg["s3"]["prefix"] == "token-sel/rho-1"

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


def test_s3_export_enabled_respects_skip_and_s3_export(monkeypatch):
    from token_selection.olmo_ext.s3_export import s3_export_enabled

    monkeypatch.delenv("S3_EXPORT", raising=False)
    monkeypatch.delenv("SKIP_S3_UPLOAD", raising=False)
    assert s3_export_enabled() is True
    assert s3_export_enabled(True) is True
    assert s3_export_enabled(False) is False

    monkeypatch.setenv("S3_EXPORT", "0")
    assert s3_export_enabled() is False
    monkeypatch.setenv("S3_EXPORT", "1")
    monkeypatch.setenv("SKIP_S3_UPLOAD", "1")
    assert s3_export_enabled() is False
    monkeypatch.setenv("SKIP_S3_UPLOAD", "true")
    assert s3_export_enabled() is False
    monkeypatch.setenv("SKIP_S3_UPLOAD", "0")
    assert s3_export_enabled() is True


def test_arm_uri_helpers_cover_fingerprint_and_metrics():
    from token_selection.olmo_ext.s3_layout import arm_uri

    assert arm_uri(
        "attention", "checkpoints", "attention_topk", "run_fingerprint.json"
    ) == (
        "s3://edullm-checkpoints/token-sel/attention/checkpoints/"
        "attention_topk/run_fingerprint.json"
    )
    assert arm_uri("attention", "metrics", "attention_topk") == (
        "s3://edullm-checkpoints/token-sel/attention/metrics/attention_topk"
    )
