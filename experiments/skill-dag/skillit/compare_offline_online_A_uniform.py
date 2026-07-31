#!/usr/bin/env python3
"""Compare offline vs online A at uniform (1/7) reference weights."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mixlaw"), str(ROOT / "skillit")]

from mixlaw_common import CURVE_FAMILIES, DOMAINS  # noqa: E402
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    online_A_from_fit,
    predict_family_loss,
)

SKILLIT = Path(__file__).resolve().parent
FAMS = list(CURVE_FAMILIES)


def uniform_weight_vector() -> np.ndarray:
    return np.ones(len(DOMAINS), dtype=np.float64) / len(DOMAINS)


def uniform_losses_from_fit(fit: dict) -> dict[str, float]:
    r = {d: 1.0 / len(DOMAINS) for d in DOMAINS}
    targets = fit["targets"]
    return {fam: predict_family_loss(targets[fam], r, domains=DOMAINS) for fam in FAMS}


def uni_losses_from_probe_extrapolation() -> dict[str, float]:
    partial = json.loads(
        (SKILLIT / "artifacts/partial_wave1/probe_chinchilla_partial.json").read_text(encoding="utf-8")
    )
    for run in partial["runs"]:
        if run["run_name"] == "probe_uni":
            return {
                fam: float(run["families"][fam]["chinchilla"])
                for fam in FAMS
                if run["families"][fam].get("chinchilla") is not None
            }
    raise KeyError("probe_uni missing from partial chinchilla report")


def onehot_chinchilla_losses() -> dict[str, dict[str, float]]:
    off = json.loads((SKILLIT / "artifacts/probes_full/A_offline.json").read_text(encoding="utf-8"))
    chin = off["chinchilla_losses"]
    return {
        run_name: {fam: float(chin[run_name][fam]) for fam in FAMS}
        for run_name in chin
        if run_name.startswith("probe_") and run_name != "probe_uni"
    }


def offline_A(L_ref: dict[str, float], onehots: dict[str, dict[str, float]]) -> np.ndarray:
    A = np.zeros((len(DOMAINS), len(FAMS)), dtype=np.float64)
    for i, dom in enumerate(DOMAINS):
        rn = f"probe_{dom}"
        for j, fam in enumerate(FAMS):
            A[i, j] = max(0.0, float(L_ref[fam]) - float(onehots[rn][fam]))
    return A


def print_tables(L_ref: dict[str, float], A_off: np.ndarray, A_on: np.ndarray, ref_label: str) -> None:
    hdr = " ".join(f"{f[:10]:>10}" for f in FAMS)
    print(f"\nReference L_j({ref_label}) @ Chinchilla (probe_uni extrapolation):")
    for fam in FAMS:
        print(f"  {fam:22s} {L_ref[fam]:.4f}")
    print(f"\nOFFLINE: A_ij = max(0, L_j({ref_label}) - L_j(one-hot)) @ Chinchilla")
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        print(f"{d:<18} " + " ".join(f"{A_off[i, j]:10.4f}" for j in range(len(FAMS))))
    print(f"\nONLINE: A_ij = max(0, -t_ij * (L_j(r) - c_j)) @ r=uniform (1/7)")
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        print(f"{d:<18} " + " ".join(f"{A_on[i, j]:10.4f}" for j in range(len(FAMS))))


def main() -> None:
    fit = load_fit_json(default_mixlaw_fit_path())
    L_uni_probe = uni_losses_from_probe_extrapolation()
    L_uni_fit = uniform_losses_from_fit(fit)
    onehots = onehot_chinchilla_losses()
    A_off = offline_A(L_uni_probe, onehots)
    A_on = online_A_from_fit(fit, uniform_weight_vector(), domains=DOMAINS, families=FAMS)

    print("=== Uniform reference (1/7 per domain) ===")
    print("\nL_j(uniform) mixing-law @ uniform weights (for comparison):")
    for fam in FAMS:
        print(f"  {fam:22s} {L_uni_fit[fam]:.4f}")

    print_tables(L_uni_probe, A_off, A_on, "uniform")

    print("\nNote: offline L_j(uniform) from probe_uni Chinchilla extrapolation;")
    print("      online A from mixlaw_fit_chinchilla.json at r_i=1/7.")


if __name__ == "__main__":
    main()
