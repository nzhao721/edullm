#!/usr/bin/env python3
"""Slideshow chart: Linear MTLD curriculum vs random-shuffle control with iso-loss arrows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

_ROOT = Path(__file__).resolve().parent
_MIXLAW_ROOT = _ROOT.parent / "skill-dag" / "mixlaw"
if str(_MIXLAW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIXLAW_ROOT))

from mixlaw_power_law import PowerLawFit, first_step_at_or_below, fit_power_law  # noqa: E402

DATA_PATH = _ROOT / "curriculum_linear_mtld_370m_wandb_curves.json"
OUT_DIR = _ROOT.parents[1] / "figures"
OUT_STEM = "curriculum_linear_mtld_slideshow"
TITLE = "Linear MTLD curriculum reaches\ncontrol quality with less training"
TITLE_FONTSIZE = 46

COLORS = {
    "linear_mtld": "#2563EB",
    "control": "#6B7280",
}
MIN_PLOT_STEP = 875
FINAL_STEP = 2384
XLABEL = "Training step"
YLABEL = "Validation macro bits per byte\n(↓ lower is better)"
AXIS_LABEL_FONTSIZE = 38
TICK_FONTSIZE = 30
ANNOTATION_FONTSIZE = 28


def add_arrow_label(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        fontweight="bold",
        color="#111827",
        zorder=5,
        clip_on=False,
    )


def load_curves() -> tuple[list[int], list[float], list[float]]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    runs = payload["runs"]
    mtld_by_step = dict(
        zip(runs["linear_mtld"]["steps"], runs["linear_mtld"]["curve"], strict=True)
    )
    ctrl_by_step = dict(zip(runs["control"]["steps"], runs["control"]["curve"], strict=True))
    shared_steps = sorted(s for s in mtld_by_step if s in ctrl_by_step and s <= FINAL_STEP)
    return (
        shared_steps,
        [mtld_by_step[s] for s in shared_steps],
        [ctrl_by_step[s] for s in shared_steps],
    )


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


def control_step_for_loss(
    fit: PowerLawFit,
    steps: list[int],
    losses: list[float],
    target: float,
) -> float:
    try:
        return fit.step_for_loss(target)
    except ValueError:
        s0, s1 = steps[-2], steps[-1]
        y0, y1 = losses[-2], losses[-1]
        if y1 <= target:
            return float(s1)
        slope = (y1 - y0) / (s1 - s0)
        if slope >= 0:
            raise ValueError("Control curve is not improving at the final checkpoint.")
        return float(s1 + (target - y1) / slope)


def main() -> None:
    steps, linear_mtld, control = load_curves()
    linear_fit = fit_power_law(steps, linear_mtld, final_step=FINAL_STEP)
    control_fit = fit_power_law(steps, control, final_step=FINAL_STEP)

    control_final = control[-1]
    linear_final = linear_fit.fitted_final
    reach_control_final_step = first_step_at_or_below(steps, linear_mtld, control_final)
    control_match_step = control_step_for_loss(control_fit, steps, control, linear_final)

    faster_pct = (FINAL_STEP - reach_control_final_step) / FINAL_STEP * 100

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
        height_ratios=[0.14, 0.86],
        hspace=0.08,
        left=0.12,
        right=0.97,
        top=0.96,
        bottom=0.20,
    )

    title_ax = fig.add_subplot(gs[0])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.55,
        TITLE,
        fontsize=TITLE_FONTSIZE,
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
        linear_mtld,
        color=COLORS["linear_mtld"],
        label="Linear + MTLD",
    )
    plot_series(
        ax,
        steps,
        control,
        color=COLORS["control"],
        label="Control",
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
        faster_arrow_y + 0.020,
        rf"$\mathbf{{{faster_pct:.0f}\%\ faster}}$",
    )

    longer_arrow_y = linear_final + 0.030
    add_horizontal_arrow(ax, longer_arrow_y, FINAL_STEP, control_match_step)
    add_arrow_label(
        ax,
        (FINAL_STEP + control_match_step) / 2,
        longer_arrow_y + 0.048,
        r"$\mathbf{Plateaus\ better}$",
    )

    ax.set_xlabel(XLABEL, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=18)
    ax.set_ylabel(YLABEL, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=28)
    ax.set_xlim(MIN_PLOT_STEP - 20, control_match_step + 120)
    late_linear = [y for s, y in zip(steps, linear_mtld, strict=True) if s >= MIN_PLOT_STEP]
    late_control = [y for s, y in zip(steps, control, strict=True) if s >= MIN_PLOT_STEP]
    y_min = min(min(late_linear), min(late_control)) - 0.02
    y_max = max(max(late_linear), max(late_control)) + 0.06
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=6, width=1.2)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=36,
        handlelength=2.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"faster_pct={faster_pct:.1f}")
    print(f"control_asymptote_a={control_fit.a:.4f}, linear_fitted_final={linear_final:.4f}")


if __name__ == "__main__":
    main()
