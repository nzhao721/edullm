#!/usr/bin/env python3
"""Slideshow chart (16:9, large type): MixLaw optimized mix vs control.

NOT the paper's Figure III -- that is plot_figure_iii_compute_savings.py,
which reproduces the published panel. This is the talk slide, kept on the
same data and the same estimator.

Two revisions to the originally published panel:

* The control is now the average of its two seeds (12536 and 12345) instead
  of seed 12536 alone, and its seed-to-seed envelope is shaded. The control
  is the only arm that was run twice, so that band is the only uncertainty
  this figure can honestly draw; no band is drawn for the single-seed
  optimized arm because none was measured.
* Both arrows are now measured between the two arms' *fitted* power laws.
  The published panel mixed estimators -- it located the "faster" arrow by
  interpolating MixLaw's observed checkpoint ladder down to the control's
  observed final value, while the "longer" arrow came from the fits -- so
  the two arrows implied slightly different speedups. Measuring both off
  the fits makes the figure internally consistent and makes it agree with
  the numbers quoted in the text.

With those two changes the arrows read 21.7% fewer steps and 1.53x longer.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from mixlaw_power_law import PowerLawFit, first_step_at_or_below, fit_power_law

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "mixlaw_370m_wandb_curves.json"
# The two-seed control average and the individual seeds live with Figure I's
# data, written by figure_i_control_seed_average.py -- one source of truth.
CONTROL_PATH = ROOT / "figure_i_wandb_curves.json"
OUT_DIRS = [ROOT / "figures"]

COLORS = {
    "optimized": "#2563EB",
    "control": "#4B5563",
    "control_band": "#9CA3AF",
}
MIN_PLOT_STEP = 875
FINAL_STEP = 2384


def add_arrow_label(ax, x: float, y: float, text: str, *, va: str = "bottom") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va=va,
        fontsize=28,
        fontweight="bold",
        color="#111827",
        zorder=5,
        clip_on=False,
    )


def load_curves() -> tuple[list[int], list[float], list[float]]:
    """MixLaw curve, plus the two-seed *average* control on the same steps."""
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    steps = payload["steps"]
    optimized = payload["runs"]["ML-pilot_caps"]["curve"]

    ctrl = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    avg = {int(pt[0]): float(pt[1]) for pt in ctrl["olmo_mix_1124"]}
    missing = [s for s in steps if s not in avg]
    if missing:
        raise SystemExit(f"two-seed control average is missing steps {missing}")
    return steps, optimized, [avg[s] for s in steps]


def load_control_band(min_step: int) -> tuple[list[int], list[float], list[float]]:
    """Per-step min/max envelope of the two control seeds."""
    ctrl = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    a = {int(p[0]): float(p[1]) for p in ctrl["olmo_mix_1124_seed6198"]}
    b = {int(p[0]): float(p[1]) for p in ctrl["olmo_mix_1124_seed12345"]}
    st = [s for s in sorted(set(a) & set(b)) if s >= min_step]
    return st, [min(a[s], b[s]) for s in st], [max(a[s], b[s]) for s in st]


def plot_series(ax, steps: list[int], losses: list[float], *, color: str, label: str) -> None:
    mask = [s >= MIN_PLOT_STEP for s in steps]
    xs = [s for s, keep in zip(steps, mask, strict=True) if keep]
    ys = [y for y, keep in zip(losses, mask, strict=True) if keep]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=3.5,
        marker="s",
        markersize=7,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
        label=label,
    )


def plot_extrapolation(ax, fit: PowerLawFit, *, start_step: int, end_step: float, color: str) -> None:
    xs = np.linspace(start_step, end_step, 200)
    ax.plot(xs, fit.predict(xs), color=color, linewidth=3.0, linestyle=(0, (6, 4)), zorder=2)


def add_horizontal_arrow(ax, y: float, x_start: float, x_end: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            arrowstyle="<->",
            mutation_scale=18,
            linewidth=2.4,
            color="#111827",
            zorder=4,
        )
    )


def main() -> None:
    steps, optimized, control = load_curves()
    optimized_fit = fit_power_law(steps, optimized, final_step=FINAL_STEP)
    control_fit = fit_power_law(steps, control, final_step=FINAL_STEP)

    # Both arrows are measured between the two fitted power laws so the
    # "fewer steps" and "longer" claims come from the same estimator.
    control_final = control_fit.fitted_final
    optimized_final = optimized_fit.fitted_final
    reach_control_final_step = optimized_fit.step_for_loss(control_final)
    control_match_step = control_fit.step_for_loss(optimized_final)

    faster_pct = (FINAL_STEP - reach_control_final_step) / FINAL_STEP * 100
    longer_ratio = control_match_step / FINAL_STEP

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.10, 0.90],
        hspace=0.06,
        left=0.10,
        right=0.97,
        top=0.96,
        bottom=0.18,
    )

    title_ax = fig.add_subplot(gs[0])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.5,
        "Optimized domain mixture reaches control quality with less training",
        fontsize=30,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
        transform=title_ax.transAxes,
    )

    ax = fig.add_subplot(gs[1])
    ax.set_facecolor("white")

    plot_series(
        ax,
        steps,
        optimized,
        color=COLORS["optimized"],
        label="Optimized mix",
    )
    band_steps, band_lo, band_hi = load_control_band(MIN_PLOT_STEP)
    ax.fill_between(
        band_steps,
        band_lo,
        band_hi,
        color=COLORS["control_band"],
        alpha=0.40,
        linewidth=0,
        zorder=1,
        label="Control seed range (n=2)",
    )
    plot_series(
        ax,
        steps,
        control,
        color=COLORS["control"],
        label="Control (2-seed average)",
    )
    plot_extrapolation(
        ax,
        control_fit,
        start_step=FINAL_STEP,
        end_step=control_match_step,
        color=COLORS["control"],
    )

    faster_arrow_y = control_final
    add_horizontal_arrow(
        ax,
        faster_arrow_y,
        reach_control_final_step,
        FINAL_STEP,
    )
    add_arrow_label(
        ax,
        (reach_control_final_step + FINAL_STEP) / 2,
        faster_arrow_y + 0.030,
        rf"$\mathbf{{{faster_pct:.1f}\%\ fewer\ steps}}$",
    )

    longer_arrow_y = optimized_final + 0.001
    add_horizontal_arrow(ax, longer_arrow_y, FINAL_STEP, control_match_step)
    add_arrow_label(
        ax,
        (FINAL_STEP + control_match_step) / 2,
        longer_arrow_y - 0.004,
        rf"$\mathbf{{{longer_ratio:.2f}\times\ longer}}$",
        va="top",
    )

    ax.set_xlabel("Training step", fontsize=24, fontweight="bold", labelpad=14)
    ax.set_ylabel(
        "Validation macro bits per byte (↓ lower is better)",
        fontsize=24,
        fontweight="bold",
        labelpad=14,
    )
    ax.set_xlim(MIN_PLOT_STEP - 20, control_match_step + 120)
    ax.set_ylim(1.595, 1.850)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
    ax.tick_params(axis="both", labelsize=18, length=6, width=1.2)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=36,
        handlelength=2.0,
    )

    for out_dir in OUT_DIRS:
        if not out_dir.parent.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            path = out_dir / f"mixlaw_slideshow.{ext}"
            fig.savefig(path, dpi=300, facecolor="white")
            print(f"Wrote {path}")
    plt.close(fig)
    print(f"faster_pct={faster_pct:.1f}")
    print(f"longer_ratio={longer_ratio:.2f}")


if __name__ == "__main__":
    main()
