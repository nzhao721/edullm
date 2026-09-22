#!/usr/bin/env python3
"""Slideshow charts: HPO comparisons as two separate two-line figures with iso-loss arrows."""

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

DATA_PATH = _ROOT / "hpo_mtld_370m_wandb_curves.json"
OUT_DIR = _ROOT.parents[1] / "figures"

COLORS = {
    "quadratic_mtld": "#5EEAD4",
    "no_proxy_winner": "#A78BFA",
    "control": "#0F766E",
}
LABELS = {
    "quadratic_mtld": "CL + HPO",
    "no_proxy_winner": "HPO",
    "control": "Control",
}
MIN_PLOT_STEP = 0
YLIM_MIN_STEP = 1000

XLABEL = "Training step"
YLABEL = "Validation macro bits per byte\n(↓ lower is better)"
TITLE_FONTSIZE = 46
AXIS_LABEL_FONTSIZE = 38
TICK_FONTSIZE = 30
LEGEND_FONTSIZE = 30
ANNOTATION_FONTSIZE = 38

CHARTS = (
    {
        "stem": "hpo_mtld_cl_control_slideshow",
        "title": "CL + HPO compounds\nefficiency gains",
        "series": ("quadratic_mtld", "control"),
    },
    {
        "stem": "hpo_mtld_hpo_control_slideshow",
        "title": "HPO reaches\ncontrol quality faster",
        "series": ("no_proxy_winner", "control"),
    },
    {
        "stem": "hpo_mtld_cl_hpo_slideshow",
        "title": "CL + HPO improves on\nHPO alone",
        "series": ("quadratic_mtld", "no_proxy_winner"),
    },
)


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


def load_curves() -> tuple[dict[str, tuple[list[int], list[float]]], int]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    out: dict[str, tuple[list[int], list[float]]] = {}
    for key, run in payload["runs"].items():
        out[key] = (run["steps"], run["curve"])
    return out, int(payload["final_step"])


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


def crossing_step(
    fit: PowerLawFit,
    steps: list[int],
    losses: list[float],
    target: float,
) -> float:
    try:
        return first_step_at_or_below(steps, losses, target)
    except ValueError:
        return fit.step_for_loss(target)


def iso_loss_annotation(
    *,
    fast_key: str,
    slow_key: str,
    target_loss: float,
    curves: dict[str, tuple[list[int], list[float]]],
    fits: dict[str, PowerLawFit],
    final_step: int,
) -> tuple[float, float, float]:
    fast_steps, fast_losses = curves[fast_key]
    reach_step = crossing_step(fits[fast_key], fast_steps, fast_losses, target_loss)
    faster_ratio = final_step / reach_step
    return reach_step, target_loss, faster_ratio


def write_chart(
    *,
    stem: str,
    title: str,
    series: tuple[str, str],
    curves: dict[str, tuple[list[int], list[float]]],
    fits: dict[str, PowerLawFit],
    control_observed_final: float,
    winner_fitted_final: float,
    final_step: int,
) -> None:
    fast_key, slow_key = series
    if slow_key == "control":
        target_loss = control_observed_final
        arrow_y_offset = 0.095
    else:
        target_loss = winner_fitted_final
        arrow_y_offset = 0.06

    reach_step, arrow_y, faster_ratio = iso_loss_annotation(
        fast_key=fast_key,
        slow_key=slow_key,
        target_loss=target_loss,
        curves=curves,
        fits=fits,
        final_step=final_step,
    )
    reach_step = min(reach_step, final_step)
    faster_ratio = final_step / reach_step

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
        title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
        transform=title_ax.transAxes,
    )

    ax = fig.add_subplot(gs[1])
    ax.set_facecolor("white")

    for key in series:
        steps, losses = curves[key]
        plot_series(ax, steps, losses, color=COLORS[key], label=LABELS[key])

    add_horizontal_arrow(ax, arrow_y, reach_step, final_step)
    label_x = reach_step + 0.78 * (final_step - reach_step)
    add_arrow_label(
        ax,
        label_x,
        arrow_y + arrow_y_offset,
        f"{faster_ratio:.1f}x faster",
    )

    ax.set_xlabel(XLABEL, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=18)
    ax.set_ylabel(YLABEL, fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", labelpad=28)
    x_pad = max(120, int(final_step * 0.03))
    ax.set_xlim(MIN_PLOT_STEP - 20, final_step + x_pad)

    visible_losses: list[float] = []
    for key in series:
        steps, losses = curves[key]
        visible_losses.extend(
            y for s, y in zip(steps, losses, strict=True) if YLIM_MIN_STEP <= s <= final_step
        )
    y_min = min(visible_losses) - 0.02
    y_max = max(visible_losses) + 0.12
    ax.set_ylim(y_min, y_max)
    x_tick = 500 if final_step <= 3000 else 20000
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=6, width=1.2)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=2.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"{stem}: reach_step={reach_step:.1f}, faster_ratio={faster_ratio:.2f}")


def main() -> None:
    curves, final_step = load_curves()
    fits = {
        key: fit_power_law(steps, losses, final_step=final_step)
        for key, (steps, losses) in curves.items()
    }
    control_observed_final = curves["control"][1][-1]
    winner_fitted_final = fits["no_proxy_winner"].fitted_final

    for chart in CHARTS:
        write_chart(
            stem=chart["stem"],
            title=chart["title"],
            series=chart["series"],
            curves=curves,
            fits=fits,
            control_observed_final=control_observed_final,
            winner_fitted_final=winner_fitted_final,
            final_step=final_step,
        )


if __name__ == "__main__":
    main()
