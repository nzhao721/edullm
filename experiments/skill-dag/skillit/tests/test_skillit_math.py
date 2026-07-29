#!/usr/bin/env python3
"""Unit tests for Skill-It math (offline update + online derivative A)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SKILLIT = Path(__file__).resolve().parents[1]
if str(_SKILLIT) not in sys.path:
    sys.path.insert(0, str(_SKILLIT))

from skillit_math import (  # noqa: E402
    ETA_DEFAULT,
    load_offline_A,
    online_A_from_fit,
    predict_family_loss,
    skillit_update,
    softmax_weights,
)


def test_softmax_sums_to_one():
    p = softmax_weights(np.array([0.0, 0.0, 0.0]))
    assert p.shape == (3,)
    assert pytest.approx(float(p.sum()), rel=0, abs=1e-12) == 1.0
    assert np.allclose(p, 1.0 / 3.0)


def test_skillit_update_toy():
    # Domain 0 helps family 0 only; high L_0 → mass on domain 0.
    A = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    losses = np.array([2.0, 0.5], dtype=np.float64)
    p = skillit_update(A, losses, eta=ETA_DEFAULT, w=1.0)
    assert p.shape == (3,)
    assert pytest.approx(float(p.sum()), abs=1e-12) == 1.0
    # score_0 = 0.2 * 2.0 = 0.4; score_1 = 0.2 * 0.5 = 0.1; score_2 = 0
    expected = softmax_weights(np.array([0.4, 0.1, 0.0]))
    assert np.allclose(p, expected)
    assert p[0] > p[1] > p[2]


def test_skillit_update_eta_zero_is_uniform_when_scores_zero():
    A = np.zeros((3, 2))
    p = skillit_update(A, [1.0, 1.0], eta=0.0)
    assert np.allclose(p, 1.0 / 3.0)


def test_load_offline_A_requires_7_by_6_shape(tmp_path: Path):
    path = tmp_path / "A_offline.npy"
    expected = np.arange(42, dtype=np.float64).reshape(7, 6)
    np.save(path, expected)
    assert np.array_equal(load_offline_A(path), expected)

    wrong_shape = tmp_path / "wrong_shape.npy"
    np.save(wrong_shape, np.zeros((6, 7)))
    with pytest.raises(ValueError, match=r"expected A shape \(7, 6\)"):
        load_offline_A(wrong_shape)


def test_online_A_sign_and_shape():
    # L = c + k*exp(t·r). With k>0, L-c = k*exp(...) > 0.
    # Helpful domain has negative t → A = max(0, -t*(L-c)) > 0.
    fit = {
        "targets": {
            "arc_challenge": {
                "c": 1.0,
                "k": 0.5,
                "t": {
                    "dclm": -2.0,
                    "arxiv": 1.0,
                    "starcoder": 0.0,
                    "pes2o": 0.0,
                    "open-web-math": 0.0,
                    "algebraic-stack": 0.0,
                    "wiki": 0.0,
                },
            },
            "arc_easy": {
                "c": 1.5,
                "k": 0.25,
                "t": {
                    "dclm": 0.5,
                    "arxiv": -1.0,
                    "starcoder": 0.0,
                    "pes2o": 0.0,
                    "open-web-math": 0.0,
                    "algebraic-stack": 0.0,
                    "wiki": 0.0,
                },
            },
        }
    }
    domains = (
        "dclm",
        "arxiv",
        "starcoder",
        "pes2o",
        "open-web-math",
        "algebraic-stack",
        "wiki",
    )
    families = ("arc_challenge", "arc_easy")
    r = np.array([1.0 / 7.0] * 7)
    A = online_A_from_fit(fit, r, domains=domains, families=families)
    assert A.shape == (7, 2)
    assert np.all(A >= 0.0)

    L0 = predict_family_loss(fit["targets"]["arc_challenge"], r, domains=domains)
    delta0 = L0 - 1.0
    assert delta0 > 0
    # dclm helps arc_challenge: -(-2)*delta = 2*delta
    assert pytest.approx(A[0, 0], rel=1e-9) == 2.0 * delta0
    # arxiv hurts: -(+1)*delta < 0 → clipped to 0
    assert A[1, 0] == 0.0

    L1 = predict_family_loss(fit["targets"]["arc_easy"], r, domains=domains)
    delta1 = L1 - 1.5
    assert pytest.approx(A[1, 1], rel=1e-9) == 1.0 * delta1
    assert A[0, 1] == 0.0


def test_online_A_matches_hand_derivative():
    fit = {
        "targets": {
            "mmlu_stem": {
                "c": 2.0,
                "k": 1.0,
                "t": {d: 0.0 for d in (
                    "dclm", "arxiv", "starcoder", "pes2o",
                    "open-web-math", "algebraic-stack", "wiki",
                )},
            }
        }
    }
    fit["targets"]["mmlu_stem"]["t"]["dclm"] = -0.5
    domains = list(fit["targets"]["mmlu_stem"]["t"].keys())
    r = {d: 0.0 for d in domains}
    r["dclm"] = 1.0
    A = online_A_from_fit(fit, r, domains=domains, families=("mmlu_stem",))
    # L-c = k*exp(-0.5) ; A_dclm = max(0, -(-0.5)*(L-c)) = 0.5 * k * exp(-0.5)
    expected = 0.5 * 1.0 * np.exp(-0.5)
    assert pytest.approx(A[0, 0], rel=1e-9) == expected


if __name__ == "__main__":
    # Allow ``python tests/test_skillit_math.py`` without pytest installed.
    test_softmax_sums_to_one()
    test_skillit_update_toy()
    test_skillit_update_eta_zero_is_uniform_when_scores_zero()
    test_online_A_sign_and_shape()
    test_online_A_matches_hand_derivative()
    print("ok")
