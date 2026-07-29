"""The guarded-resume path must never resume into a mismatched run identity."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from token_selection.olmo_ext.token_io import TOKEN_DTYPE, dtype_name, write_token_array
from token_selection.scripts.experiment_contract import build_order_contract
from token_selection.scripts.train_olmo_template import (
    _commit_run_fingerprint,
    _prepare_run_dir,
    _run_fingerprint,
    build_plan,
)


def _plan(save_folder) -> dict:
    return {
        "save_folder": str(save_folder),
        "run_id": "rho-excess-10b-scratch-v1",
        "method": "rho_excess",
        "seed": 42,
        "init_seed": 42,
        "model_name": "OLMo-2-370M-scratch",
        "model_arch": "olmo2_370M",
        "olmo_revision": "99e0009ed67679c90da970ec5ba439c9459e3757",
        "tokenizer": "allenai/OLMo-2-0425-1B",
        "sequence_length": 2048,
        "max_tokens": 10_000_000_000,
        "global_batch_size": 4_194_304,
        "lr": 4.0e-4,
        "warmup_steps": 24,
        "t0_steps": 48,
        "ts_cfg": {"k": 0.6, "alpha_start": 0.999, "alpha_end": 0.995},
        "reference_load_path": "",
        "data_order": {"contract": {"contract_sha256": "order-sha"}},
    }


def _launch_fresh(plan: dict) -> None:
    """Simulate a successful fresh launch: prepare the dir, then commit the fingerprint."""
    _prepare_run_dir(plan, resume=False)
    _commit_run_fingerprint(plan, resume=False)


def test_fingerprint_is_committed_only_after_build(tmp_path):
    save = tmp_path / "rho_excess"
    plan = _plan(save)
    # Preparing the dir alone must NOT write the fingerprint (a failed build must not
    # strand one), but it does create the empty save folder.
    _prepare_run_dir(plan, resume=False)
    assert not (save / "run_fingerprint.json").exists()
    _commit_run_fingerprint(plan, resume=False)
    assert (save / "run_fingerprint.json").exists()


def test_fresh_launch_then_refuses_reuse(tmp_path):
    save = tmp_path / "rho_excess"
    plan = _plan(save)
    _launch_fresh(plan)
    with pytest.raises(SystemExit, match="non-empty"):
        _prepare_run_dir(plan, resume=False)


def test_matching_resume_is_allowed(tmp_path):
    save = tmp_path / "rho_excess"
    plan = _plan(save)
    _launch_fresh(plan)
    _prepare_run_dir(plan, resume=True)  # must not raise


def test_resume_allows_max_tokens_extension(tmp_path):
    """A finished 5B segment can continue by raising max_tokens and --resume."""
    save = tmp_path / "rho_excess"
    plan = _plan(save)
    plan["max_tokens"] = 5_000_000_000
    _launch_fresh(plan)

    extended = copy.deepcopy(plan)
    extended["max_tokens"] = 10_000_000_000
    _prepare_run_dir(extended, resume=True)
    _commit_run_fingerprint(extended, resume=True)
    prior = json.loads((save / "run_fingerprint.json").read_text(encoding="utf-8"))
    assert prior["max_tokens"] == 10_000_000_000


def test_resume_refuses_max_tokens_decrease(tmp_path):
    save = tmp_path / "rho_excess"
    plan = _plan(save)
    _launch_fresh(plan)
    shrunk = copy.deepcopy(plan)
    shrunk["max_tokens"] = plan["max_tokens"] // 2
    with pytest.raises(SystemExit, match="Refusing to resume"):
        _prepare_run_dir(shrunk, resume=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("init_seed", 7),
        ("model_arch", "olmo2_1B"),
        ("order_contract_sha256_via", "order"),  # mutate nested order contract
        ("global_batch_size", 2_097_152),
        ("lr", 1.0e-3),
        ("warmup_steps", 10),
        ("t0_steps", 999),
        ("tokenizer", "some/other-tokenizer"),
        ("olmo_revision", "deadbeef"),
        ("rel_k_via", 0.4),  # mutate nested ts_cfg["k"]
        ("reference_content_sha256", "different-ref-bytes"),
    ],
)
def test_resume_refused_on_identity_change(tmp_path, field, value):
    save = tmp_path / "rho_excess"
    base = _plan(save)
    if field == "reference_content_sha256":
        # RHO fingerprints pin content hash; seed a matching RHO identity.
        ref = tmp_path / "ref.pt"
        import torch

        torch.save({"w": torch.tensor([1.0])}, ref)
        base["method"] = "rho_excess"
        base["reference_load_path"] = str(ref)
        base["reference_content_sha256"] = "abc123"
        save = tmp_path / "rho_excess"
        base["save_folder"] = str(save)
    _launch_fresh(base)

    changed = copy.deepcopy(base)
    if field == "order_contract_sha256_via":
        changed["data_order"]["contract"]["contract_sha256"] = "different-order"
    elif field == "rel_k_via":
        changed["ts_cfg"]["k"] = value
    else:
        changed[field] = value
    with pytest.raises(SystemExit, match="Refusing to resume"):
        _prepare_run_dir(changed, resume=True)


def test_resume_allows_reference_path_relocation(tmp_path):
    """Same reference bytes at a new host path must still --resume (Farmshare/AWS)."""
    import shutil

    import torch

    ref_a = tmp_path / "host_a" / "ref.pt"
    ref_b = tmp_path / "host_b" / "ref.pt"
    ref_a.parent.mkdir(parents=True)
    ref_b.parent.mkdir(parents=True)
    torch.save({"w": torch.tensor([1.0, 2.0, 3.0])}, ref_a)
    shutil.copy2(ref_a, ref_b)

    save = tmp_path / "rho_excess"
    plan = _plan(save)
    plan["method"] = "rho_excess"
    plan["reference_load_path"] = str(ref_a)
    _launch_fresh(plan)

    relocated = copy.deepcopy(plan)
    relocated["reference_load_path"] = str(ref_b)
    _prepare_run_dir(relocated, resume=True)
    _commit_run_fingerprint(relocated, resume=True)
    fp = json.loads((save / "run_fingerprint.json").read_text(encoding="utf-8"))
    assert fp["reference_load_path"] == str(ref_b)


def test_rho_fingerprint_pins_reference_bytes(tmp_path):
    """Replacing the file at the same path must refuse --resume."""
    import torch

    ref = tmp_path / "ref.pt"
    torch.save({"w": torch.tensor([1.0, 2.0])}, ref)
    save = tmp_path / "rho_excess"
    plan = _plan(save)
    plan["method"] = "rho_excess"
    plan["reference_load_path"] = str(ref)
    _launch_fresh(plan)

    fp = json.loads((save / "run_fingerprint.json").read_text(encoding="utf-8"))
    assert "reference_content_sha256" in fp
    assert fp["reference_load_path"] == str(ref)

    torch.save({"w": torch.tensor([9.0, 8.0])}, ref)  # same path, different bytes
    with pytest.raises(SystemExit, match="Refusing to resume"):
        _prepare_run_dir(plan, resume=True)


def test_fingerprint_omits_reference_content_hash_when_no_ref(tmp_path):
    """No reference path: fingerprint must not include RHO-only hash key."""
    fp = _run_fingerprint(_plan(tmp_path / "rho_excess"))
    assert fp["reference_load_path"] == ""
    assert "reference_content_sha256" not in fp



def test_resume_refused_without_prior_fingerprint(tmp_path):
    with pytest.raises(SystemExit, match="nothing to resume"):
        _prepare_run_dir(_plan(tmp_path / "full"), resume=True)


def _experiment_cfg() -> dict:
    return {
        "run_id": "rho-excess-10b-scratch-v1",
        "seed": 42,
        "k": 0.6,
        "t0_frac": 0.02,
        "alpha_start": 0.99,
        "alpha_end": 0.98,
        "data": {
            "tokens_s3": "s3://bucket/tokens",
            "tokenizer": "allenai/OLMo-2-0425-1B",
            "sequence_length": 8,
        },
        "s3": {
            "dataset_bucket": "edullm-datasets",
            "checkpoint_bucket": "edullm-checkpoints",
            "prefix": "token-sel/rho-1",
        },
        "model": {
            "name": "OLMo-2-370M-scratch",
            "arch": "olmo2_370M",
            "init_mode": "scratch",
            "init_seed": 42,
            "load_path": None,
        },
        "olmo_core": {"revision": "99e0009ed67679c90da970ec5ba439c9459e3757"},
        "train": {
            "max_tokens": 64,
            "global_batch_size": 16,
            "lr": 4.0e-4,
            "data_loader_seed": 42,
        },
    }


def _materialize_run_inputs(out: Path, cfg: dict) -> None:
    tokens_dir = out / "tokens"
    order_dir = out / "order"
    tokens_dir.mkdir(parents=True)
    order_dir.mkdir(parents=True)
    n_tokens = write_token_array(
        tokens_dir / "tokens_0000.npy", np.arange(64, dtype=np.uint32)
    )
    manifest = {
        "n_tokens": n_tokens,
        "dtype": dtype_name(TOKEN_DTYPE),
        "shards": [{"path": "tokens_0000.npy", "n_tokens": n_tokens}],
    }
    (tokens_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (order_dir / "manifest.json").write_text(
        json.dumps(
            {"order_contract": build_order_contract(cfg, output_dir=out, token_manifest=manifest)}
        ),
        encoding="utf-8",
    )


def test_dataset_cache_lives_outside_the_save_folder(tmp_path):
    """The build writes a dataset cache; inside save_folder it would block relaunch.

    The fresh-scratch guard rejects a non-empty save folder and --resume rejects a
    missing fingerprint, so a build that dies after caching the dataset but before the
    fingerprint is committed would leave no way forward.
    """
    cfg = _experiment_cfg()
    out = tmp_path / "run"
    out.mkdir()
    _materialize_run_inputs(out, cfg)

    plan = build_plan(cfg, method="rho_excess", out=out)
    save_folder = Path(plan["save_folder"])
    cache = Path(plan["dataset_cache"])
    assert save_folder not in cache.parents

    _prepare_run_dir(plan, resume=False)
    cache.mkdir(parents=True)
    (cache / "shard-index.bin").write_bytes(b"cached")
    # A failed build leaves the cache behind but the save folder still relaunchable.
    _prepare_run_dir(plan, resume=False)


def test_fingerprint_covers_the_fairness_critical_fields(tmp_path):
    fp = _run_fingerprint(_plan(tmp_path / "rho_excess"))
    for key in (
        "init_seed",
        "model_arch",
        "olmo_revision",
        "tokenizer",
        "order_contract_sha256",
        "global_batch_size",
        "max_tokens",
        "lr",
        "warmup_steps",
        "t0_steps",
        "rel_k",
        "rel_alpha_start",
        "rel_alpha_end",
        "alpha_schedule",
        "alpha_tau",
        "ema_seed_mode",
        "reference_load_path",
    ):
        assert key in fp
