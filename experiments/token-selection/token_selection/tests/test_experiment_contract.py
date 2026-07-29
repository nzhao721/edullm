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


def test_rho_requires_reference_load_path_or_s3():
    cfg = _config()
    cfg["methods"] = ["rho_excess"]
    with pytest.raises(ValueError, match="reference.load_path|s3_uri"):
        validate_scratch_config(cfg, method="rho_excess")

    cfg["reference"] = {"load_path": "/tmp/ref.pt"}
    validate_scratch_config(cfg, method="rho_excess")

    cfg["reference"] = {
        "load_path": None,
        "s3_uri": "s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/",
    }
    validate_scratch_config(cfg, method="rho_excess")


def test_learnability_requires_dual_reference_paths_or_s3():
    cfg = _config()
    cfg["methods"] = ["learnability"]
    with pytest.raises(ValueError, match="early|late|S3|s3"):
        validate_scratch_config(cfg, method="learnability")

    cfg["reference"] = {"early": {"load_path": "/tmp/early.pt"}}
    with pytest.raises(ValueError, match="late|S3|s3"):
        validate_scratch_config(cfg, method="learnability")

    cfg["reference"]["late"] = {"load_path": "/tmp/late.pt"}
    validate_scratch_config(cfg, method="learnability")

    cfg["reference"] = {
        "early": {
            "load_path": None,
            "s3_uri": "s3://edullm-checkpoints/x/step250/",
        },
        "late": {"load_path": None, "steps": [1000, 1125, 1315]},
    }
    validate_scratch_config(cfg, method="learnability")


def test_rel_ema_refhq_seed_requires_reference_load_path():
    cfg = _config()
    cfg["methods"] = ["rel_ema"]
    cfg["ema"] = {"seed_mode": "refhq", "schedule": "linear"}
    with pytest.raises(ValueError, match="seed_mode='refhq'"):
        validate_scratch_config(cfg, method="rel_ema")

    cfg["reference"] = {"load_path": "/tmp/refhq.pt"}
    validate_scratch_config(cfg, method="rel_ema")

    cfg["reference"] = {
        "load_path": None,
        "s3_uri": "s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/",
    }
    validate_scratch_config(cfg, method="rel_ema")

    # Zero-seed REL does not need a reference path.
    cfg2 = _config()
    cfg2["methods"] = ["rel_ema"]
    cfg2["ema"] = {"seed_mode": "zero", "schedule": "exp", "tau": 300}
    validate_scratch_config(cfg2, method="rel_ema")


def test_rel_ema_refhq_arm_contract():
    """RefHQ-seeded REL arm: constant α=0.9985, seed_mode=refhq, t0=0, S3 prefix."""
    arm_cfg = (
        Path(__file__).resolve().parents[2]
        / "rel-ema-refhq"
        / "configs"
        / "run_rel_ema_refhq_10b.yaml"
    )
    cfg = load_config(arm_cfg)
    assert cfg["methods"] == ["rel_ema"]
    assert cfg["run_id"] == "rel-ema-refhq-10b-scratch-v1"
    assert cfg["run_id"] != "rel-ema-10b-scratch-v1"
    assert int(cfg["t0_steps"]) == 0
    assert float(cfg.get("t0_frac", 0.0)) == 0.0
    assert float(cfg["k"]) == 0.6
    assert float(cfg["alpha_start"]) == 0.9985
    assert float(cfg["alpha_end"]) == 0.9985
    assert str(cfg.get("alpha_schedule") or "linear") == "linear"
    ema = cfg.get("ema") or {}
    assert str(ema.get("seed_mode")) == "refhq"
    assert str(ema.get("schedule") or "linear") == "linear"
    assert cfg["model"]["arch"] == "olmo2_370M"
    assert cfg["model"]["init_mode"] == "scratch"
    assert cfg["model"].get("load_path") is None
    assert int(cfg["train"]["checkpoint_every_steps"]) == 125
    assert cfg["train"].get("checkpoint_keep_last") is None
    assert cfg["train"].get("ephemeral_checkpoint_every_steps") is None
    assert cfg["train"]["dp_type"] == "hsdp"
    assert cfg["train"]["optim_type"] == "skip_step_adamw"
    assert bool(cfg["train"]["compile_model"]) is True
    assert float(cfg["train"]["lr"]) == 4.0e-4
    assert int(cfg["train"]["warmup_steps"]) == 24
    assert float(cfg["train"]["lr_alpha_f"]) == 0.1
    assert int(cfg["train"]["global_batch_size"]) == 4_194_304
    assert str((cfg.get("train") or {}).get("cuda_visible_devices") or "") == ""
    assert cfg["s3"]["prefix"] == "token-sel/rel-ema-refhq"
    assert (cfg.get("reference") or {}).get("step") == 1315
    # load_path may be null; s3_uri is enough — --launch auto-materializes DistCP→.pt
    assert (cfg.get("reference") or {}).get("load_path") is None
    assert str((cfg.get("reference") or {}).get("s3_uri") or "").startswith("s3://")
    validate_scratch_config(cfg, method="rel_ema")
    assert (cfg.get("eval") or {}).get("task_loss", {}).get("enabled") is True
    assert (cfg.get("eval") or {}).get("task_loss", {}).get("results_dir") == (
        "task_loss_results/rel-ema-refhq"
    )

    # Independent-var contrast vs rel-ema-exp (near-clone pair).
    exp_cfg = load_config(
        Path(__file__).resolve().parents[2]
        / "rel-ema-exp"
        / "configs"
        / "run_rel_ema_exp_10b.yaml"
    )
    assert (exp_cfg.get("ema") or {}).get("seed_mode") == "zero"
    assert str(exp_cfg.get("alpha_schedule") or "") == "exp"
    assert cfg["run_id"] != exp_cfg["run_id"]
    assert cfg["s3"]["prefix"] != exp_cfg["s3"]["prefix"]


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


def test_middle_ppl_token_arm_contract():
    """Middle-PPL token arm owns its YAML; shared scorer stays in the package."""
    arm_cfg = (
        Path(__file__).resolve().parents[2]
        / "middle-ppl-token"
        / "configs"
        / "run_middle_ppl_token_10b.yaml"
    )
    mid = load_config(arm_cfg)

    assert mid["methods"] == ["middle_ppl"]
    assert mid["run_id"] == "middle-ppl-token-10b-v1"
    assert int(mid["t0_steps"]) == 0
    assert float(mid.get("t0_frac", 0.0)) == 0.0
    assert float(mid["k"]) == 0.6
    assert int(mid["train"]["checkpoint_every_steps"]) == 125
    assert mid["train"].get("checkpoint_keep_last") is None
    assert mid["train"].get("ephemeral_checkpoint_every_steps") is None
    assert mid["train"].get("save_async") is False
    assert mid["train"].get("pre_train_checkpoint") is True
    assert mid["model"]["arch"] == "olmo2_370M"
    assert mid["model"]["init_mode"] == "scratch"
    assert str((mid.get("train") or {}).get("cuda_visible_devices") or "") == ""
    assert (mid.get("eval") or {}).get("task_loss", {}).get("enabled") is True
    assert (mid.get("eval") or {}).get("task_loss", {}).get("results_dir") == (
        "task_loss_results/middle-ppl-token"
    )
    assert mid["s3"]["prefix"] == "token-sel/middle-ppl-token"
    assert mid["s3"]["checkpoint_bucket"] == "edullm-checkpoints"
    assert mid["data"]["tokens_s3"].endswith("regmix/regmix-10b/tokenized") or (
        mid["data"]["tokens_s3"].rstrip("/").endswith("regmix/regmix-10b/tokenized")
    )

    # Package configs must not still own a middle_ppl arm YAML.
    assert not (_CONFIGS / "run_middle_ppl_10b.yaml").exists()


def test_rho_package_config_still_isolated_from_middle_ppl_token():
    """If the package still ships a RHO YAML, it must not collide with middle-ppl-token."""
    rho_path = _CONFIGS / "run_rho_10b.yaml"
    if not rho_path.exists():
        return
    rho = load_config(rho_path)
    mid = load_config(
        Path(__file__).resolve().parents[2]
        / "middle-ppl-token"
        / "configs"
        / "run_middle_ppl_token_10b.yaml"
    )
    assert rho["run_id"] != mid["run_id"]
    assert rho["output_dir"] != mid["output_dir"]
    assert rho["s3"]["prefix"] != mid["s3"]["prefix"]


def test_rho_1_arm_contract():
    """RHO-1 arm owns its YAML under the shared checkpoint + t0=0 contract."""
    arm_cfg = (
        Path(__file__).resolve().parents[2] / "rho-1" / "configs" / "run_rho_10b.yaml"
    )
    rho = load_config(arm_cfg)
    assert rho["methods"] == ["rho_excess"]
    assert rho["run_id"] == "rho-1-regmix10b-v1"
    assert rho["run_id"] != "rho-excess-10b-scratch-v1"
    assert rho.get("arm") == "rho-1"
    assert int(rho["t0_steps"]) == 0
    assert float(rho["t0_frac"]) == 0.0
    assert float(rho["k"]) == 0.6
    assert int(rho["train"]["checkpoint_every_steps"]) == 125
    assert rho["train"].get("checkpoint_keep_last") is None
    assert rho["train"].get("ephemeral_checkpoint_every_steps") is None
    assert rho["train"].get("compile_model") is True
    assert float(rho["train"].get("lr_alpha_f", 0)) == 0.1
    assert int(rho["train"].get("warmup_steps", 0)) == 24
    assert int(rho["train"].get("global_batch_size", 0)) == 4_194_304
    assert float(rho["train"].get("max_grad_norm", 0)) == 1.0
    assert float(rho["train"].get("z_loss_multiplier", 0)) == 1e-5
    assert rho["model"]["arch"] == "olmo2_370M"
    assert rho["reference"]["step"] == 1315
    assert str((rho.get("train") or {}).get("cuda_visible_devices") or "") == ""
    assert (rho.get("eval") or {}).get("task_loss", {}).get("enabled") is True
    assert (rho.get("eval") or {}).get("task_loss", {}).get("results_dir") == (
        "task_loss_results/rho-1"
    )
    assert rho["s3"]["prefix"] == "token-sel/rho-1"
    assert rho["s3"]["checkpoint_bucket"] == "edullm-checkpoints"


def test_learnability_token_arm_contract():
    """Learnability-token: dual RefHQ early−late, t0=0, S3 token-sel/learnability-token."""
    from token_selection.scripts import derive_steps

    arm_cfg = (
        Path(__file__).resolve().parents[2]
        / "learnability-token"
        / "configs"
        / "run_learnability_10b.yaml"
    )
    cfg = load_config(arm_cfg)
    assert cfg["methods"] == ["learnability"]
    assert cfg["run_id"] == "learnability-token-10b-scratch-v1"
    assert int(cfg["t0_steps"]) == 0
    assert float(cfg.get("t0_frac", 1.0)) == 0.0
    assert float(cfg["k"]) == 0.6
    total, t0 = derive_steps(cfg)
    assert total == 2384
    assert t0 == 0
    assert int(cfg["train"]["checkpoint_every_steps"]) == 125
    assert cfg["train"].get("checkpoint_keep_last") is None
    assert cfg["train"].get("ephemeral_checkpoint_every_steps") is None
    assert bool(cfg["train"].get("pre_train_checkpoint")) is True
    assert cfg["model"]["arch"] == "olmo2_370M"
    assert cfg["model"]["init_mode"] == "scratch"
    assert cfg["reference"]["early"]["step"] == 250
    assert cfg["reference"]["late"]["steps"] == [1000, 1125, 1315]
    # load_path may be null; S3 provenance is enough for --launch auto-materialize.
    assert cfg["reference"]["early"]["load_path"] is None
    assert cfg["reference"]["late"]["load_path"] is None
    assert str(cfg["reference"]["early"].get("s3_uri") or "").startswith("s3://")
    validate_scratch_config(cfg, method="learnability")
    assert str((cfg.get("train") or {}).get("cuda_visible_devices") or "") == ""
    assert (cfg.get("eval") or {}).get("task_loss", {}).get("enabled") is True
    assert cfg["eval"]["task_loss"]["results_dir"] == "task_loss_results/learnability-token"
    assert cfg["s3"]["prefix"] == "token-sel/learnability-token"
    assert cfg["s3"]["checkpoint_bucket"] == "edullm-checkpoints"
    assert cfg["data"]["tokens_s3"] == "s3://edullm-datasets/regmix/regmix-10b/tokenized"

