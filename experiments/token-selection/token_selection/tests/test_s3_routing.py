"""S3 routing for pre-tokenized inputs and per-run outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from token_selection.olmo_ext.s3_layout import (
    ARM_DIRS,
    CHECKPOINT_BUCKET,
    TOKEN_SEL_ROOT,
    arm_from_prefix,
    arm_prefix,
    arm_uri,
    default_s3_block,
)
from token_selection.scripts import load_config, resolve_tokens_s3, s3_uri
from token_selection.scripts.sync_artifacts import sync_dir

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "run_rho_10b.yaml"


def _run_cfg() -> dict:
    return {
        "data": {"tokens_s3": "s3://token-bucket/corpus/tokens"},
        "s3": {
            "dataset_bucket": "db",
            "checkpoint_bucket": "cb",
            "prefix": "proj/run",
        },
    }


def test_resolve_tokens_s3():
    assert resolve_tokens_s3(_run_cfg()) == "s3://token-bucket/corpus/tokens"
    with pytest.raises(ValueError, match="tokens_s3"):
        resolve_tokens_s3({"data": {}})


def test_routing_per_run_outputs():
    cfg = _run_cfg()
    assert s3_uri(cfg, "metrics") == "s3://db/proj/run/metrics"
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
    assert block["dataset_bucket"] == "edullm-datasets"
    assert "blade" in ARM_DIRS and "rho-1" in ARM_DIRS
    assert "middle-ppl-token" in ARM_DIRS
    with pytest.raises(ValueError, match="token-sel/"):
        arm_from_prefix("token-selection/rho-1")


def test_sync_dir_refuses_empty_mirrored_upload(tmp_path):
    empty = tmp_path / "tokens"
    with pytest.raises(SystemExit, match="Refusing mirrored"):
        sync_dir(empty, "s3://token-bucket/corpus/tokens", profile="x", upload=True, mirror=True)


def test_mirrored_token_download_keeps_the_derived_manifest(monkeypatch, tmp_path):
    """manifest.json is derived locally; a mirroring re-sync must not delete it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "token_selection.scripts.sync_artifacts._aws",
        lambda profile, args: calls.append(args),
    )
    sync_dir(
        tmp_path / "tokens",
        "s3://token-bucket/corpus/tokens/",
        profile="x",
        upload=False,
        mirror=True,
        keep=("manifest.json",),
    )
    assert calls[0][:2] == ["s3", "sync"]
    assert "--delete" in calls[0]
    assert calls[0][calls[0].index("--exclude") + 1] == "manifest.json"


def test_run_config_points_at_the_real_corpus():
    """The production config has to match what is actually in the bucket."""
    cfg = load_config(CONFIG_PATH)
    assert resolve_tokens_s3(cfg) == "s3://edullm-datasets/regmix/regmix-10b/tokenized"
    # The corpus sidecars all record this tokenizer, and it sets the vocabulary size.
    assert cfg["data"]["tokenizer"] == "allenai/dolma2-tokenizer"
    # Output buckets must exist in the account, so they are pinned here too.
    assert cfg["s3"]["dataset_bucket"] == "edullm-datasets"
    assert cfg["s3"]["checkpoint_bucket"] == "edullm-checkpoints"
    assert cfg["s3"]["prefix"] == "token-sel/rho-1"


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

