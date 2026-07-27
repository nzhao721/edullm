"""S3 routing for pre-tokenized inputs and per-run outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    assert resolve_tokens_s3(cfg) == "s3://edullm-dataset-regmix/regmix-10b/tokenized"
    # The corpus sidecars all record this tokenizer, and it sets the vocabulary size.
    assert cfg["data"]["tokenizer"] == "allenai/dolma2-tokenizer"
    # Output buckets must exist in the account, so they are pinned here too.
    assert cfg["s3"]["dataset_bucket"] == "edullm-dataset-olmo"
    assert cfg["s3"]["checkpoint_bucket"] == "edullm-checkpoints"
