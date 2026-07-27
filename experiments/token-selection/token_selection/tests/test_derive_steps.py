"""Tests for step derivation from max_tokens / GBS."""

from __future__ import annotations

from token_selection.scripts import derive_steps


def test_derive_steps_10b_defaults():
    cfg = {
        "t0_frac": 0.02,
        "train": {"max_tokens": 10_000_000_000, "global_batch_size": 4_194_304},
    }
    total, t0 = derive_steps(cfg)
    assert total == 10_000_000_000 // 4_194_304
    assert t0 == max(1, int(round(total * 0.02)))
    assert t0 < total


def test_derive_steps_explicit_t0_wins():
    cfg = {
        "t0_frac": 0.02,
        "t0_steps": 24,
        "train": {"max_tokens": 10_000_000_000, "global_batch_size": 4_194_304},
    }
    total, t0 = derive_steps(cfg)
    assert total == 10_000_000_000 // 4_194_304
    assert t0 == 24


def test_derive_steps_5b_segment():
    cfg = {
        "t0_frac": 0.02,
        "t0_steps": 24,
        "train": {"max_tokens": 5_000_000_000, "global_batch_size": 4_194_304},
    }
    total, t0 = derive_steps(cfg)
    assert total == 5_000_000_000 // 4_194_304
    assert total == 1192
    assert t0 == 24
