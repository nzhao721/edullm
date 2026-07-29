"""Unit tests for RefHQ DistCP→.pt materialize helpers (no live S3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from token_selection.olmo_ext.refhq_materialize import (
    ensure_reference_paths,
    reference_source_ok,
    _step_slug,
)


def test_step_slug_from_uri():
    assert _step_slug("s3://bucket/checkpoints/step1315/") == "step1315"
    assert _step_slug("s3://bucket/checkpoints/step250") == "step250"


def test_reference_source_ok_rho_and_learnability():
    assert not reference_source_ok({"methods": ["rho_excess"]}, method="rho_excess")
    assert reference_source_ok(
        {
            "methods": ["rho_excess"],
            "reference": {"s3_uri": "s3://edullm-checkpoints/x/step1315/"},
        },
        method="rho_excess",
    )
    assert reference_source_ok(
        {
            "methods": ["learnability"],
            "reference": {
                "early": {"s3_uri": "s3://b/step250/"},
                "late": {"steps": [1000, 1125, 1315]},
            },
        },
        method="learnability",
    )


def test_ensure_reference_paths_reuses_local(tmp_path, monkeypatch):
    ref_pt = tmp_path / "ref.pt"
    ref_pt.write_bytes(b"fake")
    cfg = {
        "methods": ["rho_excess"],
        "reference": {
            "load_path": str(ref_pt),
            "s3_uri": "s3://should-not-touch/step1315/",
        },
    }

    def boom(*_a, **_k):
        raise AssertionError("should not download when local path exists")

    monkeypatch.setattr(
        "token_selection.olmo_ext.refhq_materialize.ensure_distcp_pt", boom
    )
    out = ensure_reference_paths(cfg, method="rho_excess", cache_dir=tmp_path / "cache")
    assert Path(out["reference.load_path"]) == ref_pt.resolve()


def test_ensure_reference_paths_fills_from_s3_uri(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    fake = cache / "refhq_step1315_model.pt"

    def fake_ensure(s3_uri, *, cache_dir=None, output_name=None, **_k):
        assert "step1315" in s3_uri
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_bytes(b"ok")
        return fake.resolve()

    monkeypatch.setattr(
        "token_selection.olmo_ext.refhq_materialize.ensure_distcp_pt", fake_ensure
    )
    cfg = {
        "methods": ["rho_excess"],
        "reference": {
            "load_path": None,
            "s3_uri": "s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/",
        },
    }
    out = ensure_reference_paths(cfg, method="rho_excess", cache_dir=cache)
    assert cfg["reference"]["load_path"] == str(fake.resolve())
    assert out["reference.load_path"] == str(fake.resolve())
