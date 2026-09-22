#!/usr/bin/env python3
"""Slideshow bar chart: Skill-It final macro task-loss with bootstrap CIs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "skillit_370m_bootstrap_results.json"
OUT_DIR = ROOT.parents[2] / "figures"

ARM_COLORS = {
    "probe": "#60A5FA",
    "derivative": "#FB923C",
    "offline_ml_start": "#A78BFA",
}
CONTROL_COLOR = "#94A3B8"
ERROR_COLOR = "#334155"
BAR_WIDTH = 0.58


def load_data() -> list[dict]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return payload["arms"]


def main() -> None:
    arms = load_data()
    labels = [
        arm["label"].replace(" w/ ", " w/\n") if arm["key"] == "offline_ml_start" else arm["label"]
        for arm in arms
    ]
    values = [arm["fitted_final"] for arm in arms]
    yerr_lo = [arm["fitted_final"] - arm["ci_lo"] for arm in arms]
    yerr_hi = [arm["ci_hi"] - arm["fitted_final"] for arm in arms]

    colors = [
        CONTROL_COLOR if arm["key"] == "control" else ARM_COLORS[arm["key"]]
        for arm in arms
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.72, bottom=0.24)

    ax.set_facecolor("white")

    edgecolors = [
        "none" if arm["key"] == "control" else "white"
        for arm in arms
    ]

    x = np.arange(len(arms))
    bars = ax.bar(
        x,
        values,
        width=BAR_WIDTH,
        color=colors,
        edgecolor=edgecolors,
        linewidth=1.8,
        zorder=3,
    )

    ax.errorbar(
        x,
        values,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor=ERROR_COLOR,
        elinewidth=2.6,
        capsize=12,
        capthick=2.6,
        zorder=5,
    )

    for bar, arm in zip(bars, arms, strict=True):
        half_lo = arm["fitted_final"] - arm["ci_lo"]
        half_hi = arm["ci_hi"] - arm["fitted_final"]
        half = (half_lo + half_hi) / 2
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            arm["ci_hi"] + 0.002,
            f"{arm['fitted_final']:.4f}\n± {half:.4f}",
            ha="center",
            va="bottom",
            fontsize=30,
            fontweight="semibold",
            color="#111827",
            linespacing=1.05,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=30, rotation=0, ha="center")
    ax.set_ylabel(
        "Fitted final macro bits per byte\n(↓ lower is better)",
        fontsize=28,
        fontweight="bold",
        labelpad=20,
    )
    ax.set_ylim(1.595, 1.625)
    ax.tick_params(axis="y", labelsize=22, length=6, width=1.2)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)

    plot_pos = ax.get_position()
    title_x = plot_pos.x0 + plot_pos.width / 2
    fig.text(
        title_x,
        plot_pos.y1 + 0.10,
        "Skill-It: schedule-based mixtures\ndo not improve on the control",
        fontsize=34,
        fontweight="bold",
        color="#111827",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "skillit_slideshow.png"
    pdf_path = OUT_DIR / "skillit_slideshow.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.35)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
