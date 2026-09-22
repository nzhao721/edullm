"""Figure II: Task loss curves for the dynamic-reweighting arms (Probe, Derivative)
compared to the static control (Data Mixing Laws paper mix / regmix-control).

Regenerated using the completed FarmShare probe rerun (job 1730368) and the
existing derivative/control runs. The stale third arm ("Probe with optimized
start", from the superseded RunPod-era run set) has been dropped.

The curve data lives alongside this script in `figure_ii_curves.py` so the
figure is reproducible from the repository rather than from a scratch
directory.

Two revisions to the originally published panel:

* The palette no longer pairs red against green. Red-green is the most
  common form of color vision deficiency, and the two arms that the figure
  most wants the reader to tell apart (the LightGBM static mixture and the
  derivative arm) were exactly that pair. Each series now also carries its
  own dash pattern and marker, so the panel survives grayscale printing.
* The control is drawn as the envelope of its two seeds rather than as a
  bare average line. That band is the only uncertainty in this figure that
  the data actually supports: the control was run at two seeds (12536 and
  12345), while every reweighting arm was run once. The band therefore sets
  the scale against which the single-seed arms should be read, and no band
  is drawn for those arms because none was measured.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_ii_curves import CURVES, OLMO_SEED12536, OLMO_SEED12345  # noqa: E402

SKILLIT = Path(__file__).resolve().parent
# Committed alongside the experiment, next to the data it reads.
OUT_DIRS = [SKILLIT / "figures"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})

# Colorblind-safe: grey control, then blue / orange / purple. No red-green pair.
# Distinct dash patterns and markers so the series also separate in grayscale.
SERIES = [
    ("olmo_average", "Olmo-mix-1124 average (control)", "#6B7280", (0, (5, 2)), "s"),
    ("lgbm_control", "LightGBM static", "#7C3AED", (0, (1, 1.6)), "D"),
    ("probe", "Probe", "#2563EB", "-", "o"),
    ("derivative", "Derivative", "#D97706", (0, (6, 1.6, 1.4, 1.6)), "^"),
]

CONTROL_BAND_COLOR = "#9CA3AF"
X_LEFT = 700

def lead_in(all_steps, all_y, window_min, x_left):
    """Segment running left from the first in-window point.

    Uses the slope to the last point *before* the window, i.e. the slope the
    curve would have had if that checkpoint were still plotted, and returns
    the two endpoints. The far endpoint is placed at x_left; when the implied
    value there is off the top of the panel, matplotlib's clipping is what
    makes the line leave through the top edge rather than the side.
    """
    before = all_steps < window_min
    after = all_steps >= window_min
    if not before.any() or not after.any():
        return None
    s0, y0 = all_steps[before][-1], all_y[before][-1]
    s1, y1 = all_steps[after][0], all_y[after][0]
    slope = (y1 - y0) / (s1 - s0)
    return (np.array([x_left, s1]), np.array([y1 + slope * (x_left - s1), y1]))

def seed_envelope(steps_a, y_a, steps_b, y_b):
    """Per-step control envelope, on the steps the two seeds share.

    Returned as series so the lower and upper boundary can each be extended
    at its own slope, the same way the plotted lines are.
    """
    a = dict(zip(steps_a, y_a))
    b = dict(zip(steps_b, y_b))
    st = np.array(sorted(set(a) & set(b)), dtype=float)
    lo = np.array([min(a[s], b[s]) for s in st], dtype=float)
    hi = np.array([max(a[s], b[s]) for s in st], dtype=float)
    return st, lo, hi



def control_band(min_step: int) -> tuple[list[int], list[float], list[float]]:
    """Per-step envelope of the two control seeds, on their shared steps."""
    a = dict(zip(OLMO_SEED12536["steps"], OLMO_SEED12536["curve"]))
    b = dict(zip(OLMO_SEED12345["steps"], OLMO_SEED12345["curve"]))
    steps = [s for s in sorted(set(a) & set(b)) if s >= min_step]
    lo = [min(a[s], b[s]) for s in steps]
    hi = [max(a[s], b[s]) for s in steps]
    return steps, lo, hi

fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=200)
ax.set_title("Task loss for dynamic reweighting and static mixtures",
             fontsize=15, fontweight="bold", pad=12)

MIN_STEP = 700

BAND_ST, BAND_LO, BAND_HI = seed_envelope(
    np.array(OLMO_SEED12536["steps"], dtype=float),
    np.array(OLMO_SEED12536["curve"], dtype=float),
    np.array(OLMO_SEED12345["steps"], dtype=float),
    np.array(OLMO_SEED12345["curve"], dtype=float),
)

band_steps, band_lo, band_hi = control_band(MIN_STEP)
ax.fill_between(band_steps, band_lo, band_hi, color=CONTROL_BAND_COLOR, alpha=0.38,
                linewidth=0, zorder=1,
                label="Control seed range (n=2)")

for key, label, color, ls, marker in SERIES:
    d = CURVES[key]
    steps = [s for s in d["steps"] if s >= MIN_STEP]
    vals = [v for s, v in zip(d["steps"], d["curve"]) if s >= MIN_STEP]
    ax.plot(steps, vals, color=color, linestyle=ls, marker=marker,
            markersize=3.5, linewidth=1.8, label=label, zorder=3)

# This panel leaves its y-limits to autoscale, so freeze them before adding the
# lead-in segments -- otherwise an off-panel endpoint would stretch the axis.
_ylim = ax.get_ylim()

# Lead-in: extend each curve off the left edge at the slope implied by the
# checkpoint before the window, so it reads as a run already in progress.
for key, label, color, ls, marker in SERIES:
    d = CURVES[key]
    _seg = lead_in(np.array(d["steps"], dtype=float), np.array(d["curve"], dtype=float),
                   MIN_STEP, X_LEFT)
    if _seg is not None:
        ax.plot(_seg[0], _seg[1], color=color, linestyle=ls, linewidth=1.8, zorder=3)

_lo_seg = lead_in(BAND_ST, BAND_LO, MIN_STEP, X_LEFT)
_hi_seg = lead_in(BAND_ST, BAND_HI, MIN_STEP, X_LEFT)
if _lo_seg is not None and _hi_seg is not None:
    ax.fill_between(_lo_seg[0], _lo_seg[1], _hi_seg[1], color=CONTROL_BAND_COLOR,
                    alpha=0.38, linewidth=0, zorder=1)

ax.set_ylim(_ylim)

ax.set_xlabel("Training step", labelpad=8)
ax.set_ylabel("Validation macro bits-per-byte\n(20-task OLMES avg, \u2193 lower is better)")
ax.grid(True, linestyle=":", linewidth=0.7, color="#c9c9c9", alpha=0.9)
ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.92,
          edgecolor="none", facecolor="white", borderpad=0.5, labelspacing=0.35)
ax.set_xlim(MIN_STEP, 2450)
# Same axes rectangle as Figures I and III, so the 12pt type occupies the
# same fraction of the canvas in all three.
fig.subplots_adjust(left=0.115, right=0.97, top=0.90, bottom=0.13)

# Inset: final steps, rescaled
# (sized/positioned to leave room for the larger tick labels and title
# below without colliding with the main plot's right/top edges)
# Top at 0.88 of the axes, not 0.98: restoring the full axes box left the
# inset title with no clearance and it collided with the top spine.
axins = ax.inset_axes([0.52, 0.46, 0.46, 0.42])
ins_steps, ins_lo, ins_hi = control_band(1900)
axins.fill_between(ins_steps, ins_lo, ins_hi, color=CONTROL_BAND_COLOR, alpha=0.38,
                   linewidth=0, zorder=1)
for key, label, color, ls, marker in SERIES:
    d = CURVES[key]
    steps = [s for s in d["steps"] if s >= 1900]
    vals = [v for s, v in zip(d["steps"], d["curve"]) if s >= 1900]
    axins.plot(steps, vals, color=color, linestyle=ls, marker=marker, markersize=3,
               linewidth=1.5, zorder=3)
# Same lead-in treatment in the inset. Both of its axes are autoscaled, so
# capture the limits first and restore them after, then extend each curve to
# the left spine at the slope implied by the checkpoint before the zoom window.
_ins_xlim, _ins_ylim = axins.get_xlim(), axins.get_ylim()
for key, label, color, ls, marker in SERIES:
    d = CURVES[key]
    _seg = lead_in(np.array(d["steps"], dtype=float), np.array(d["curve"], dtype=float),
                   1900, _ins_xlim[0])
    if _seg is not None:
        axins.plot(_seg[0], _seg[1], color=color, linestyle=ls, linewidth=1.5, zorder=3)

_ilo_seg = lead_in(BAND_ST, BAND_LO, 1900, _ins_xlim[0])
_ihi_seg = lead_in(BAND_ST, BAND_HI, 1900, _ins_xlim[0])
if _ilo_seg is not None and _ihi_seg is not None:
    axins.fill_between(_ilo_seg[0], _ilo_seg[1], _ihi_seg[1],
                       color=CONTROL_BAND_COLOR, alpha=0.38, linewidth=0, zorder=1)

axins.set_xlim(_ins_xlim)
axins.set_ylim(_ins_ylim)

axins.set_title("final steps, rescaled", fontsize=9.3, style="italic", pad=2)
axins.tick_params(labelsize=8, length=2)
axins.grid(True, linestyle=":", linewidth=0.6, color="#d5d5d5")
axins.yaxis.set_major_locator(mticker.MaxNLocator(5))

for out_dir in OUT_DIRS:
    if not out_dir.parent.exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_ii_dynamic_reweighting.png"
    pdf_path = out_dir / "figure_ii_dynamic_reweighting.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
plt.close(fig)
