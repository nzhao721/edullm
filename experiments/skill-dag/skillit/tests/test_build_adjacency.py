#!/usr/bin/env python3
"""Unit tests for offline Skill-It A construction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SKILLIT = Path(__file__).resolve().parents[1]
if str(_SKILLIT) not in sys.path:
    sys.path.insert(0, str(_SKILLIT))

from build_adjacency import build_A_from_extrapolated  # noqa: E402
from mixlaw_common import CURVE_FAMILIES, DOMAINS  # noqa: E402


def _run(run_name: str, losses: dict[str, float]) -> dict:
    return {
        "run_name": run_name,
        "families": {
            family: {"chinchilla": loss, "note": None}
            for family, loss in losses.items()
        },
    }


def test_build_A_uses_regmix_minus_onehot_and_clips_negative_values():
    L_reg = {family: 1.0 + j for j, family in enumerate(CURVE_FAMILIES)}
    runs = []
    for i, domain in enumerate(DOMAINS):
        losses = dict(L_reg)
        losses[CURVE_FAMILIES[0]] = L_reg[CURVE_FAMILIES[0]] - (0.1 * (i + 1))
        losses[CURVE_FAMILIES[1]] = L_reg[CURVE_FAMILIES[1]] + 0.5
        runs.append(_run(f"probe_{domain}", losses))

    A, detail = build_A_from_extrapolated({"chinchilla_steps": 5806, "runs": runs}, L_reg)

    assert A.shape == (7, 6)
    assert np.allclose(A[:, 0], np.arange(1, 8) * 0.1)
    assert np.all(A[:, 1] == 0.0)
    assert detail["chinchilla_step"] == 5806
    assert detail["reference"] == "regmix"
    assert detail["domain_order"] == list(DOMAINS)
    assert detail["family_order"] == list(CURVE_FAMILIES)
