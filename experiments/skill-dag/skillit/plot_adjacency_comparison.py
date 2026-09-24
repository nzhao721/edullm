#!/usr/bin/env python3
"""Figure IV: Probe vs. derivative adjacency matrices (A_ij), side by side.

Probe A is unchanged (offline, from one-hot probe extrapolation vs. the
Data Mixing Laws paper mixture reference -- artifacts/probes_full/A_offline.npy).

Derivative A is now evaluated at the LightGBM-optimized ("min1pct") mixture's
domain weights (mixlaw_fit_lightgbm_chinchilla.json: optimization.min1pct.weights)
instead of the previous reference, the Data Mixing Laws paper dataset's domain
weights (skillit_math.regmix_weight_vector()) -- this matches the mixture that
Skill-It training actually starts from, rather than an arbitrary comparison point.

Style matches the original Figure IV exactly: two heatmap panels sharing one
colorbar, blue sequential colormap, "-" for exact-zero cells, bold black title
above both panels, italic A_ij panel subtitles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mixlaw"), str(ROOT / "skillit")]

from skillit_math import (  # noqa: E402
    DEFAULT_DOMAINS,
    DEFAULT_FAMILIES,
    default_mixlaw_fit_path,
    load_fit_json,
    load_offline_A,
    online_A_from_fit,
)

SKILLIT = Path(__file__).resolve().parent
MIXLAW = SKILLIT.parent / "mixlaw"
# Committed alongside the experiment, as token-selection and curriculum do.
# Pass --out-dir to also drop a copy into an Overleaf working tree.
OUT_DIR = SKILLIT / "figures"

DOMAINS = list(DEFAULT_DOMAINS)
FAMILIES = list(DEFAULT_FAMILIES)
FAMILY_LABELS = {
    "arc_challenge": "arc\nchallenge",
    "arc_easy": "arc\neasy",
    "mmlu_humanities": "mmlu\nhumanities",
    "mmlu_other": "mmlu\nother",
    "mmlu_social_sciences": "mmlu\nsocial\nsciences",
    "mmlu_stem": "mmlu\nstem",
}

TITLE = r"Domain-to-skill adjacency matrices ($A_{ij}$)"
VMAX = 0.5


def lightgbm_min1pct_weights() -> list[float]:
    """LightGBM-optimized ("min1pct") mixture weights -- the actual Skill-It
    training starting point -- in DOMAINS order."""
    fit = json.loads((MIXLAW / "mixlaw_fit_lightgbm_chinchilla.json").read_text(encoding="utf-8"))
    w = fit["optimization"]["min1pct"]["weights"]
    return [float(w[d]) for d in DOMAINS]


def draw_panel(ax, A: np.ndarray, *, title: str) -> None:
    im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=VMAX, aspect="auto")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES], fontsize=8)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            val = A[i, j]
            text = "-" if val == 0 else f"{val:.2f}"
            color = "white" if val > VMAX * 0.6 else "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        action="append",
        default=None,
        metavar="DIR",
        help=f"Write the figure here (repeatable). Default: {OUT_DIR}",
    )
    args = ap.parse_args()
    out_dirs = [Path(d) for d in (args.out_dir or [])] or [OUT_DIR]

    fit = load_fit_json(default_mixlaw_fit_path())
    A_probe = load_offline_A(SKILLIT / "artifacts/probes_full/A_offline.npy")
    A_deriv = online_A_from_fit(fit, lightgbm_min1pct_weights(), domains=DOMAINS, families=FAMILIES)

    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=200)
    fig.suptitle(TITLE, fontsize=15, fontweight="bold", y=0.98)

    draw_panel(axes[0], A_probe, title=r"Probe adjacency $A_{ij}$")
    im = draw_panel(axes[1], A_deriv, title=r"Derivative adjacency $A_{ij}$")

    fig.subplots_adjust(left=0.12, right=0.87, top=0.82, bottom=0.15, wspace=0.45)
    cax = fig.add_axes((0.89, 0.15, 0.02, 0.67))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        "larger = domain helps that skill more\n(0 = no benefit under that construction)",
        fontsize=7,
        rotation=90,
        labelpad=6,
    )
    cbar.ax.tick_params(labelsize=8)

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "adjacency_comparison.png"
        fig.savefig(png_path, facecolor="white")
        print(f"Wrote {png_path}")
    plt.close(fig)
    print("\nDerivative A (LightGBM min1pct reference):")
    hdr = " ".join(f"{f[:10]:>10}" for f in FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        print(f"{d:<18} " + " ".join(f"{A_deriv[i, j]:10.4f}" for j in range(len(FAMILIES))))


if __name__ == "__main__":
    main()
