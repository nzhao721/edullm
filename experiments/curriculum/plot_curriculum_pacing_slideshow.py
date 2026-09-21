#!/usr/bin/env python3
"""Slideshow chart: curriculum difficulty exposure over time (three pacings)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from curriculum_pacing import (  # noqa: E402
    TOTAL_STEPS,
    WARMUP_SWITCH,
    PoolSpec,
    interleave_subbucket_durations,
    segment_index,
    split_equal_mass,
)

OUT_DIR = _ROOT.parents[1] / "figures"
OUT_STEM = "curriculum_pacing_slideshow"
N_CHUNKS = 10_000
N_DIFFICULTY_BINS = 120
STEP_STRIDE = 4
WARMUP_N_TAKE = max(1, N_CHUNKS // WARMUP_SWITCH)
# Slideshow-visible easy→hard diagonal (points, not heatmap bins).
WARMUP_DIAGONAL_LINEWIDTH = 22.0
ACTIVE_COLOR = "#1E3A8A"
INACTIVE_COLOR = "#FFFFFF"


@dataclass(frozen=True)
class PanelSpec:
    pacing: str
    title: str
    n_buckets: int = 10
    n_cycles: int = 10


PANELS = (
    PanelSpec("linear", "Linear pacing", n_buckets=5),
    PanelSpec("warmup", "Warmup pacing"),
    PanelSpec("interleave", "Interleaved pacing", n_buckets=5, n_cycles=3),
)


def equal_segment_boundaries(total: int, n_segments: int) -> tuple[int, ...]:
    """Inclusive starts, exclusive ends for ``n_segments`` equal-length segments."""
    if n_segments <= 0:
        raise ValueError("n_segments must be > 0")
    boundaries = [0]
    for i in range(1, n_segments + 1):
        boundaries.append(int(round(i * total / n_segments)))
    boundaries[-1] = int(total)
    return tuple(boundaries)


def interleave_subbucket_index_local(
    local_step: int,
    segment_steps: int,
    n_buckets: int,
) -> int:
    durs = interleave_subbucket_durations(segment_steps, n_buckets)
    cum = 0
    for i, d in enumerate(durs):
        cum += d
        if local_step < cum:
            return i
    return n_buckets - 1


def pool_for_visual(step: int, n_chunks: int, spec: PanelSpec) -> PoolSpec:
    n = int(n_chunks)
    name = spec.pacing

    if name == "warmup":
        if int(step) < WARMUP_SWITCH:
            return PoolSpec(
                mode=name,
                start=0,
                end=n,
                ordered=True,
                ordered_offset=int(step),
            )
        return PoolSpec(mode=name, start=0, end=n, ordered=False)

    buckets = split_equal_mass(n, spec.n_buckets)

    if name == "linear":
        boundaries = equal_segment_boundaries(TOTAL_STEPS, spec.n_buckets)
        seg = segment_index(step, boundaries)
        lo, hi = buckets[seg]
        return PoolSpec(mode=name, start=lo, end=max(lo + 1, hi) if hi == lo else hi, ordered=False)

    if name == "interleave":
        boundaries = equal_segment_boundaries(TOTAL_STEPS, spec.n_cycles)
        seg = segment_index(step, boundaries)
        start, end = boundaries[seg], boundaries[seg + 1]
        sub = interleave_subbucket_index_local(int(step) - start, end - start, spec.n_buckets)
        lo, hi = buckets[sub]
        return PoolSpec(mode=name, start=lo, end=max(lo + 1, hi) if hi == lo else hi, ordered=False)

    raise ValueError(f"Unknown pacing {name!r}")


def difficulty_mass(step: int, spec: PanelSpec) -> np.ndarray:
    """Per-bin sampling mass over normalized difficulty in [0, 1]."""
    mass = np.zeros(N_DIFFICULTY_BINS, dtype=float)
    edges = np.linspace(0.0, 1.0, N_DIFFICULTY_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    pool = pool_for_visual(step, N_CHUNKS, spec)

    if pool.ordered:
        visual_take = WARMUP_N_TAKE
        start = (int(pool.ordered_offset or step) * WARMUP_N_TAKE) % N_CHUNKS
        end = start + visual_take
        lo = start / N_CHUNKS
        hi = min(1.0, end / N_CHUNKS)
        if hi <= lo:
            hi = min(1.0, lo + (visual_take / N_CHUNKS))
        mask = (centers >= lo) & (centers < hi)
        if not mask.any():
            idx = min(N_DIFFICULTY_BINS - 1, int(lo * N_DIFFICULTY_BINS))
            mass[idx] = 1.0
        else:
            mass[mask] = 1.0
        return mass / mass.sum()

    lo = pool.start / N_CHUNKS
    hi = pool.end / N_CHUNKS
    if hi <= lo:
        hi = min(1.0, lo + 1.0 / spec.n_buckets)
    mask = (centers >= lo) & (centers < hi)
    if not mask.any():
        idx = min(N_DIFFICULTY_BINS - 1, int(lo * N_DIFFICULTY_BINS))
        mass[idx] = 1.0
    else:
        mass[mask] = 1.0
    return mass / mass.sum()


def build_heatmap(spec: PanelSpec) -> np.ndarray:
    steps = np.arange(0, TOTAL_STEPS, STEP_STRIDE)
    heatmap = np.vstack([difficulty_mass(int(step), spec) for step in steps])
    return (heatmap > 0).astype(float)


def plot_panel(ax: plt.Axes, spec: PanelSpec) -> None:
    heatmap = build_heatmap(spec)
    cmap = ListedColormap([INACTIVE_COLOR, ACTIVE_COLOR])

    ax.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[0.0, 1.0, 0.0, 1.0],
        interpolation="nearest",
    )

    if spec.pacing == "warmup":
        switch_x = WARMUP_SWITCH / TOTAL_STEPS
        ax.plot(
            [0.0, switch_x],
            [0.0, 1.0],
            color=ACTIVE_COLOR,
            linewidth=WARMUP_DIAGONAL_LINEWIDTH,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.axvline(switch_x, color="#111827", linewidth=4.0, linestyle="-", alpha=0.9, zorder=4)
        ax.text(
            switch_x + 0.03,
            0.94,
            "Random shuffle",
            fontsize=18,
            fontweight="semibold",
            color="#FFFFFF",
            va="top",
        )
        ax.text(
            switch_x - 0.08,
            0.94,
            "Easy → hard",
            fontsize=18,
            fontweight="semibold",
            color="#111827",
            ha="right",
            va="top",
        )

    ax.set_title(spec.title, fontsize=28, fontweight="bold", color="#111827", pad=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Training Steps", fontsize=24, fontweight="bold", labelpad=10)
    ax.set_ylabel("Difficulty", fontsize=24, fontweight="bold", labelpad=10)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, spec in zip(axes, PANELS, strict=True):
        ax.set_facecolor("white")
        plot_panel(ax, spec)

    fig.subplots_adjust(wspace=0.28)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
