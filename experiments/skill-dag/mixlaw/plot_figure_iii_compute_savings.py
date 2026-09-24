"""Figure III: compute savings, MixLaw optimized mix vs the Olmo-mix-1124 control.

This is the generator behind the paper's Figure III. Two changes from the
originally published panel, both to the numbers rather than the styling:

* The control is the average of its two seeds (12536 and 12345) instead of seed
  12536 alone, and its seed-to-seed envelope is shaded. The control is the only
  arm that was run twice, so that band is the only uncertainty this panel can
  honestly show; no band is drawn for the single-seed optimized arm.
* Both arrows are measured between the two arms' fitted power laws. The
  published panel mixed estimators -- it crossed MixLaw's fitted curve against
  the control's *observed* final value for the upper arrow while taking the
  lower arrow from the fits -- so the two arrows implied different speedups.

With those changes the arrows read 21.7% fewer steps and 1.53x longer, which
are the numbers quoted in section 4.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

MIXLAW = Path(__file__).resolve().parent
OUT_DIRS = [MIXLAW / "figures"]

mixlaw_curves = json.loads((MIXLAW / "mixlaw_370m_wandb_curves.json").read_text(encoding="utf-8"))
# Two-seed control average and the individual seeds, written by
# figure_i_control_seed_average.py and shared with Figure I.
_ctrl_data = json.loads((MIXLAW / "figure_i_wandb_curves.json").read_text(encoding="utf-8"))

COL_CONTROL = "#6b7280"      # gray
COL_MIXLAW = "#2563eb"       # blue
COL_CONTROL_BAND = "#9ca3af"  # lighter gray, control seed envelope
WINDOW_MIN = 700
X_LEFT = 650

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


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
})

steps = np.array(mixlaw_curves["steps"], dtype=float)
ml = np.array(mixlaw_curves["runs"]["ML-pilot_caps"]["curve"], dtype=float)
_avg = {int(p[0]): float(p[1]) for p in _ctrl_data["olmo_mix_1124"]}
_s12536 = {int(p[0]): float(p[1]) for p in _ctrl_data["olmo_mix_1124_seed6198"]}
_s12345 = {int(p[0]): float(p[1]) for p in _ctrl_data["olmo_mix_1124_seed12345"]}
_missing = [int(x) for x in steps if int(x) not in _avg]
if _missing:
    raise SystemExit(f"two-seed control average is missing steps {_missing}")
ctrl = np.array([_avg[int(x)] for x in steps], dtype=float)
band_lo = np.array([min(_s12536[int(x)], _s12345[int(x)]) for x in steps], dtype=float)
band_hi = np.array([max(_s12536[int(x)], _s12345[int(x)]) for x in steps], dtype=float)
mask = steps >= 1000

def powerlaw(x, a, b, alpha):
    return a + b / np.power(x, alpha)

def fit(x, y):
    best = None
    for alpha0 in (0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
        try:
            p0 = [y.min(), (y.max() - y.min()) * (x.min() ** alpha0), alpha0]
            popt, _ = curve_fit(powerlaw, x, y, p0=p0, maxfev=20000,
                                 bounds=([0, 0, 0.05], [10, 1e6, 3.0]))
            resid = np.sum((powerlaw(x, *popt) - y) ** 2)
            if best is None or resid < best[0]:
                best = (resid, popt)
        except Exception:
            pass
    return best[1]

popt_ml = fit(steps[mask], ml[mask])
popt_ctrl = fit(steps[mask], ctrl[mask])

FINAL_STEP = 2384
ml_final_fitted = powerlaw(FINAL_STEP, *popt_ml)
# Fitted, not observed: the lower arrow already comes from the fits, so using
# the observed final here made the two arrows disagree.
ctrl_final_fitted = powerlaw(FINAL_STEP, *popt_ctrl)

fine = np.linspace(1000, FINAL_STEP, 20000)
ml_fine = powerlaw(fine, *popt_ml)
idx = np.argmax(ml_fine <= ctrl_final_fitted)
cross_step = fine[idx]
pct_faster = (1 - cross_step / FINAL_STEP) * 100

big = np.linspace(1000, 45000, 4_000_000)
ctrl_big = powerlaw(big, *popt_ctrl)
idx2 = np.argmax(ctrl_big <= ml_final_fitted)
cross_step2 = big[idx2]
times_longer = cross_step2 / FINAL_STEP

fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=200)

keep = steps >= 700
ax.fill_between(steps[keep], band_lo[keep], band_hi[keep], color=COL_CONTROL_BAND,
                alpha=0.40, linewidth=0, zorder=1, label="Control seed range (n=2)")
ax.plot(steps[keep], ctrl[keep], color=COL_CONTROL, linestyle="-", marker="s", markersize=4,
        linewidth=2.0, zorder=3, label="Olmo-mix-1124 (control, 2-seed avg)")
ax.plot(steps[keep], ml[keep], color=COL_MIXLAW, linestyle="-", marker="o", markersize=4,
        linewidth=2.4, zorder=4, label="MixLaw fit (ours, observed)")

# Lead-in: extend both curves off the left edge at the slope implied by the
# checkpoint before the window, so the panel reads as a run already under way
# rather than one that begins at step 750.
for _y, _color, _lw, _z in ((ctrl, COL_CONTROL, 2.0, 3), (ml, COL_MIXLAW, 2.4, 4)):
    _seg = lead_in(steps, _y, WINDOW_MIN, X_LEFT)
    if _seg is not None:
        ax.plot(_seg[0], _seg[1], color=_color, linestyle="-", linewidth=_lw, zorder=_z)

_lo_seg = lead_in(steps, band_lo, WINDOW_MIN, X_LEFT)
_hi_seg = lead_in(steps, band_hi, WINDOW_MIN, X_LEFT)
if _lo_seg is not None and _hi_seg is not None:
    ax.fill_between(_lo_seg[0], _lo_seg[1], _hi_seg[1], color=COL_CONTROL_BAND,
                    alpha=0.40, linewidth=0, zorder=1)

ext = np.linspace(FINAL_STEP, cross_step2 * 1.03, 500)
ax.plot(ext, powerlaw(ext, *popt_ctrl), color=COL_CONTROL, linestyle=(0, (2, 2)), linewidth=1.8,
        zorder=2, label="Control, power-law extrapolation")

ax.axhline(ml_final_fitted, color="#999999", linestyle=":", linewidth=1.0, zorder=1)

# "27% faster" arrow sits ABOVE both curves for the whole [cross_step, FINAL_STEP] span,
# not just above their final values (the curves are still elevated above their final
# values in that window, which is what caused the earlier collision).
window = (steps >= cross_step) & (steps <= FINAL_STEP)
local_max = (max(band_hi[window].max(), ml[window].max()) if window.any()
             else ctrl_final_fitted)
faster_y = local_max + 0.018
ax.annotate("", xy=(cross_step, faster_y), xytext=(FINAL_STEP, faster_y),
            arrowprops=dict(arrowstyle="<->", color="#111", lw=1.6))
ax.plot([cross_step, cross_step], [faster_y - 0.004, local_max + 0.002], color="#111", lw=0.8, ls=":")
ax.plot([FINAL_STEP, FINAL_STEP], [faster_y - 0.004, local_max + 0.002], color="#111", lw=0.8, ls=":")
ax.text((cross_step + FINAL_STEP) / 2, faster_y + 0.010, f"{pct_faster:.1f}% fewer steps",
        ha="center", fontsize=11, fontweight="bold")

ax.annotate("", xy=(FINAL_STEP, ml_final_fitted), xytext=(cross_step2, ml_final_fitted),
            arrowprops=dict(arrowstyle="<->", color="#111", lw=1.6))
ax.text((FINAL_STEP + cross_step2) / 2, ml_final_fitted - 0.016, f"{times_longer:.2f}× longer",
        ha="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Training step", labelpad=8)
ax.set_ylabel("Validation macro bits-per-byte\n(20-task OLMES avg, $\\downarrow$ lower is better)")
ax.set_title("MixLaw fit and the control's power-law extrapolation", fontsize=15, fontweight="bold", pad=12)
ax.grid(True, linestyle=":", linewidth=0.7, color="#c9c9c9", alpha=0.9)
ax.set_axisbelow(True)
ax.set_xlim(650, cross_step2 * 1.05)
ax.set_ylim(1.585, 1.90)
ax.legend(loc="upper right", frameon=False, fontsize=10)

fig.subplots_adjust(left=0.115, right=0.97, top=0.90, bottom=0.13)
for out_dir in OUT_DIRS:
    if not out_dir.parent.exists():
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        path = out_dir / f"figure_iii_compute_savings.{ext}"
        fig.savefig(path)
        print(f"Wrote {path}")
print("pct_faster=", pct_faster, "times_longer=", times_longer)
