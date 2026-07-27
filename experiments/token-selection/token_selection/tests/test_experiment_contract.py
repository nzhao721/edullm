"""Tests for fail-closed scratch and deterministic-order contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from token_selection.olmo_ext.token_io import write_token_array
from token_selection.scripts import load_config, resolve_tokens_s3
from token_selection.scripts.experiment_contract import (
    build_order_contract,
    manifest_train_paths,
    validate_order_contract,
    validate_scratch_config,
    validate_token_manifest,
)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _config() -> dict:
    return {
        "seed": 42,
        "model": {"init_mode": "scratch", "init_seed": 42, "load_path": None},
        "train": {"data_loader_seed": 42, "global_batch_size": 128},
        "data": {"sequence_length": 16},
    }


def test_scratch_contract_rejects_checkpoint_or_seed_drift():
    cfg = _config()
    validate_scratch_config(cfg)

    cfg["model"]["load_path"] = "s3://checkpoint"
    with pytest.raises(ValueError, match="checkpoint"):
        validate_scratch_config(cfg)

    cfg = _config()
    cfg["train"]["data_loader_seed"] = 7
    with pytest.raises(ValueError, match="data_loader_seed"):
        validate_scratch_config(cfg)


def test_rho_requires_reference_load_path():
    cfg = _config()
    cfg["methods"] = ["rho_excess"]
    with pytest.raises(ValueError, match="reference.load_path"):
        validate_scratch_config(cfg, method="rho_excess")

    cfg["reference"] = {"load_path": "/tmp/ref.pt"}
    validate_scratch_config(cfg, method="rho_excess")


def test_validate_experiment_refuses_missing_rho_reference(tmp_path, monkeypatch):
    """Preflight must fail closed on a typo'd reference path, not only on null."""
    import json
    import sys

    from token_selection.scripts import validate_experiment as ve

    cfg_path = tmp_path / "run_rho.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "run_id": "rho-test",
                "seed": 42,
                "output_dir": str(tmp_path / "out"),
                "methods": ["rho_excess"],
                "k": 0.6,
                "t0_frac": 0.02,
                "alpha_start": 0.99,
                "alpha_end": 0.98,
                "reference": {"load_path": str(tmp_path / "missing_ref.pt")},
                "data": {
                    "tokens_s3": "s3://bucket/tokens",
                    "tokenizer": "allenai/dolma2-tokenizer",
                    "sequence_length": 8,
                },
                "model": {
                    "init_mode": "scratch",
                    "init_seed": 42,
                    "load_path": None,
                    "name": "x",
                    "arch": "olmo2_370M",
                },
                "train": {
                    "max_tokens": 64,
                    "global_batch_size": 16,
                    "data_loader_seed": 42,
                    "lr": 1e-4,
                },
                "s3": {
                    "dataset_bucket": "b",
                    "checkpoint_bucket": "c",
                    "prefix": "p",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_experiment", "--config", str(cfg_path)],
    )
    with pytest.raises(SystemExit, match="does not exist"):
        ve.main()


def test_order_contract_binds_token_manifest_and_loader_settings(tmp_path):
    out = tmp_path / "out"
    tokens = out / "tokens"
    tokens.mkdir(parents=True)
    manifest = {"n_tokens": 128, "sequence_length": 16}
    (tokens / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _config()

    contract = build_order_contract(cfg, output_dir=out, token_manifest=manifest)
    validate_order_contract(cfg, output_dir=out, contract=contract)

    changed = _config()
    changed["train"]["global_batch_size"] = 256
    with pytest.raises(ValueError, match="Order contract mismatch"):
        validate_order_contract(changed, output_dir=out, contract=contract)


def test_manifest_train_paths_refuses_stray_and_missing(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    write_token_array(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))
    (tokens / "manifest.json").write_text(
        json.dumps(
            {
                "n_tokens": 16,
                "shards": [{"source": "a", "path": "tokens_0000.npy", "n_tokens": 16}],
            }
        ),
        encoding="utf-8",
    )
    assert manifest_train_paths(tokens) == [str(tokens / "tokens_0000.npy")]

    # A stray shard not in the manifest must be refused (it would silently train).
    write_token_array(tokens / "tokens_0001.npy", np.arange(16, dtype=np.uint32))
    with pytest.raises(ValueError, match="Unlisted token shard"):
        manifest_train_paths(tokens)

    # A manifest that lists a missing file must also be refused.
    (tokens / "tokens_0001.npy").unlink()
    (tokens / "manifest.json").write_text(
        json.dumps(
            {
                "n_tokens": 32,
                "shards": [
                    {"source": "a", "path": "tokens_0000.npy", "n_tokens": 16},
                    {"source": "b", "path": "tokens_0002.npy", "n_tokens": 16},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from"):
        manifest_train_paths(tokens)


def _write_manifest(tokens, manifest) -> None:
    (tokens / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_rejects_np_save_shards(tmp_path):
    """A .npy header would be read as tokens and shift every sequence boundary."""
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    np.save(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))
    _write_manifest(
        tokens,
        {"n_tokens": 16, "shards": [{"path": "tokens_0000.npy", "n_tokens": 16}]},
    )
    with pytest.raises(ValueError, match="np.save"):
        validate_token_manifest(tokens)


def test_manifest_rejects_token_count_mismatch(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    write_token_array(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))
    _write_manifest(
        tokens,
        {"n_tokens": 99, "shards": [{"path": "tokens_0000.npy"}]},
    )
    with pytest.raises(ValueError, match="n_tokens=99"):
        validate_token_manifest(tokens)


def test_manifest_requires_n_tokens_and_shards(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    write_token_array(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))

    _write_manifest(tokens, {"n_tokens": 16, "paths": ["tokens_0000.npy"]})
    with pytest.raises(ValueError, match="lists no shards"):
        validate_token_manifest(tokens)

    _write_manifest(tokens, {"shards": [{"path": "tokens_0000.npy"}]})
    with pytest.raises(ValueError, match="positive integer 'n_tokens'"):
        validate_token_manifest(tokens)


def test_tokens_uri_placeholder_is_refused():
    with pytest.raises(ValueError, match="placeholder"):
        resolve_tokens_s3({"data": {"tokens_s3": "s3://REPLACE_ME/tokens"}})
    assert resolve_tokens_s3({"data": {"tokens_s3": "s3://real-bucket/tokens/"}}) == (
        "s3://real-bucket/tokens"
    )


def test_rho_and_middle_ppl_10b_run_identities_are_disjoint():
    """Arm isolation is config identity, not a git-branch fork.

    RHO and middle_ppl must not share run_id, output_dir, S3 prefix, or method.
    k=0.6 is intentionally shared; that is the keep rate, not arm identity.
    """
    rho = load_config(_CONFIGS / "run_rho_10b.yaml")
    mid = load_config(_CONFIGS / "run_middle_ppl_10b.yaml")

    assert rho["methods"] == ["rho_excess"]
    assert mid["methods"] == ["middle_ppl"]
    run_ids = {rho["run_id"], mid["run_id"]}
    outs = {rho["output_dir"], mid["output_dir"]}
    prefixes = {rho["s3"]["prefix"], mid["s3"]["prefix"]}
    assert len(run_ids) == 2
    assert len(outs) == 2
    assert len(prefixes) == 2
    assert float(rho["k"]) == float(mid["k"]) == 0.6
    assert int(mid["train"]["checkpoint_every_steps"]) == 250

    for cfg in (rho, mid):
        gpu = str((cfg.get("train") or {}).get("cuda_visible_devices") or "")
        assert gpu == ""
