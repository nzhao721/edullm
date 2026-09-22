#!/usr/bin/env python3
"""Power-law fit + alpha-free residual bootstrap for the 370M validation arms.

Reproduces Tables II and III of the paper from the committed curve file
``skill_dag_370m_wandb_curves.json``. No network or W&B access required.

Method
------
For each arm we fit

    y = a + b / step**alpha

to the eval points with ``step >= 1000`` (earlier points are dominated by
initialization noise). Uncertainty is a **residual bootstrap** (Efron, 1979;
Freedman, 1981, for the regression form): residuals about the point fit are
resampled with replacement, added back to the fitted curve, and the model is
re-fit on each of ``--n-boot`` draws. The 95% CI is the 2.5th/97.5th percentile
of the resulting distribution of the fitted value at ``final_step``.

The bootstrap is **alpha-free**: ``alpha`` is re-selected on every draw rather
than frozen at the point estimate. Holding alpha fixed understates the interval
by roughly a third at this sample size, so every arm in the paper uses the
alpha-free form.

Each arm is resampled from its **own independent random stream**, spawned from
``--seed`` via ``SeedSequence.spawn``. This matters: the arms are separate
training runs with no shared randomness, so their bootstrap distributions must
be independent. Drawing every arm's resample indices from one seeded generator
(which happens by default when all arms have the same number of eval points)
silently couples them and distorts every between-arm interval -- here it made
the derivative arm's bootstrap draws correlate +0.91 with the control's and the
probe arm's -0.43, shrinking one difference interval and inflating the other.

The Olmo-mix-1124 control is the *average of the two dataloader seeds*: its
bootstrap distribution is the element-by-element mean of the two seeds' own
alpha-free bootstrap distributions, and its CI is the 2.5/97.5 percentiles of
that averaged distribution.

p-values are two-sided bootstrap tests on the difference of two independent
distributions, ``p = 2 * min(P(diff <= 0), P(diff >= 0))``, floored at the
bootstrap resolution ``1/n_boot``.

Usage
-----
    python fit_and_bootstrap_370m.py                       # writes results JSON
    python fit_and_bootstrap_370m.py --n-boot 10000 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CURVES_PATH = HERE / "skill_dag_370m_wandb_curves.json"
RESULTS_PATH = HERE / "skill_dag_370m_bootstrap_results.json"

ALPHA_GRID = np.linspace(0.05, 3.0, 400)
MIN_STEP = 1000

# Arms compared against the Olmo-mix-1124 seed average.
VS_CONTROL = [
    "data-mixing-laws-paper",
    "mixlaw-fit",
    "lightgbm",
    "skillit-probe",
    "skillit-derivative",
]


def _ols_over_alpha(X: np.ndarray, Y: np.ndarray):
    """Closed-form 2-parameter OLS of every row of Y on every row of X.

    X: (A, n) candidate regressors (step**-alpha). Y: (B, n) targets.
    Returns (sse, slope, intercept), each (A, B).
    """
    xm = X.mean(axis=1, keepdims=True)
    ym = Y.mean(axis=1, keepdims=True)
    Xc, Yc = X - xm, Y - ym
    Sxx = (Xc * Xc).sum(axis=1)
    Sxy = Xc @ Yc.T
    Syy = (Yc * Yc).sum(axis=1)
    slope = Sxy / Sxx[:, None]
    sse = Syy[None, :] - (Sxy**2) / Sxx[:, None]
    intercept = ym.T - slope * xm
    return sse, slope, intercept


def fit_and_bootstrap(steps, losses, *, final_step, n_boot, seed, chunk=20_000):
    """Point fit + alpha-free residual bootstrap. Returns (fitted, finals).

    ``seed`` should be a per-arm :class:`numpy.random.SeedSequence` so that arms
    are resampled independently; see the module docstring. Draws are generated
    in blocks of ``chunk`` to bound peak memory at large ``n_boot``.
    """
    s = np.asarray(steps, dtype=float)
    y = np.asarray(losses, dtype=float)
    mask = s >= MIN_STEP
    s, y = s[mask], y[mask]
    if s.size < 3:
        raise ValueError(f"need >=3 points with step >= {MIN_STEP}, got {s.size}")

    X = s[None, :] ** (-ALPHA_GRID[:, None])          # (A, n)
    sse0, slope0, icept0 = _ols_over_alpha(X, y[None, :])
    best = int(np.argmin(sse0[:, 0]))
    alpha0 = float(ALPHA_GRID[best])
    a0, b0 = float(icept0[best, 0]), float(slope0[best, 0])
    pred = a0 + b0 * s ** (-alpha0)
    resid = y - pred
    fitted = a0 + b0 * final_step ** (-alpha0)

    rng = np.random.default_rng(seed)
    finals = np.empty(n_boot, dtype=float)
    for lo in range(0, n_boot, chunk):
        hi = min(lo + chunk, n_boot)
        draws = rng.integers(0, resid.size, (hi - lo, resid.size))
        Y = pred[None, :] + resid[draws]               # (B, n)
        sse, slope, icept = _ols_over_alpha(X, Y)
        pick = np.argmin(sse, axis=0)                  # alpha-free: per-draw alpha
        cols = np.arange(hi - lo)
        finals[lo:hi] = icept[pick, cols] + slope[pick, cols] * final_step ** (-ALPHA_GRID[pick])
    return fitted, finals


def ci(finals):
    lo, hi = np.percentile(finals, [2.5, 97.5])
    return float(lo), float(hi)


def diff_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a - b, floored at 1/n.

    ``a`` and ``b`` are independent bootstrap distributions (one per arm), so
    the difference is taken draw-by-draw only to build its sampling
    distribution -- the draws are not paired observations.
    """
    d = a - b
    n = d.size
    p = 2.0 * min((d >= 0).mean(), (d <= 0).mean())
    return float(max(min(p, 1.0), 1.0 / n))


def fmt_p(p: float, n_boot: int) -> str:
    floor = 1.0 / n_boot
    return f"p < {floor:g}" if p <= floor else f"p = {p:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curves", type=Path, default=CURVES_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()

    data = json.loads(args.curves.read_text(encoding="utf-8"))
    final_step = int(data["final_step"])
    runs = data["runs"]

    # One independent stream per arm -- see the module docstring.
    streams = np.random.SeedSequence(args.seed).spawn(len(runs))

    fitted, finals, observed = {}, {}, {}
    for (key, run), stream in zip(runs.items(), streams):
        f, dist = fit_and_bootstrap(
            run["steps"], run["macro_bpb"],
            final_step=final_step, n_boot=args.n_boot, seed=stream,
        )
        fitted[key], finals[key] = f, dist
        observed[key] = float(run["macro_bpb"][-1])

    # Control = element-by-element average of the two seeds' bootstrap draws.
    s1, s2 = "olmo-mix-1124-seed6198", "olmo-mix-1124-seed12345"
    finals["olmo-mix-1124-average"] = 0.5 * (finals[s1] + finals[s2])
    fitted["olmo-mix-1124-average"] = 0.5 * (fitted[s1] + fitted[s2])
    observed["olmo-mix-1124-average"] = 0.5 * (observed[s1] + observed[s2])

    out = {
        "method": "power-law fit (y = a + b/step**alpha) on steps >= 1000; "
                  "alpha-free residual bootstrap; one independent resampling "
                  "stream per arm",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "final_step": final_step,
        "alpha_grid": {"lo": 0.05, "hi": 3.0, "n": int(ALPHA_GRID.size)},
        "arms": {},
        "comparisons": {},
    }

    order = list(runs) + ["olmo-mix-1124-average"]
    print(f"{'arm':28s} {'fitted':>8s} {'observed':>9s}   95% CI")
    for key in order:
        lo, hi = ci(finals[key])
        label = runs[key]["label"] if key in runs else "Olmo-mix-1124 average"
        out["arms"][key] = {
            "label": label,
            "fitted_final": round(fitted[key], 6),
            "observed_final": round(observed[key], 6),
            "ci95": [round(lo, 6), round(hi, 6)],
        }
        print(f"{key:28s} {fitted[key]:8.4f} {observed[key]:9.4f}   [{lo:.4f}, {hi:.4f}]")

    ctrl = finals["olmo-mix-1124-average"]
    print()
    for key in VS_CONTROL:
        d = finals[key] - ctrl
        p = diff_p(finals[key], ctrl)
        lo, hi = ci(d)
        out["comparisons"][f"{key}_vs_olmo_average"] = {
            "mean_diff_bpb": round(float(d.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "p_value": p,
            "p_display": fmt_p(p, args.n_boot),
        }
        print(f"{key:28s} vs olmo avg: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {fmt_p(p, args.n_boot)}")

    # Dynamic arms against the best static mixture (LightGBM), which is the
    # comparison the dynamic-reweighting conclusion actually rests on.
    print()
    for key in ("skillit-probe", "skillit-derivative"):
        d = finals[key] - finals["lightgbm"]
        p = diff_p(finals[key], finals["lightgbm"])
        lo, hi = ci(d)
        out["comparisons"][f"{key}_vs_lightgbm"] = {
            "mean_diff_bpb": round(float(d.mean()), 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "p_value": p,
            "p_display": fmt_p(p, args.n_boot),
        }
        print(f"{key:28s} vs LightGBM: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {fmt_p(p, args.n_boot)}")

    # Seed-variance estimate quoted in the paper.
    d = finals[s1] - finals[s2]
    p = diff_p(finals[s1], finals[s2])
    lo, hi = ci(d)
    out["seed_variance_estimate"] = {
        "description": "Olmo-mix-1124 seed 12536 minus seed 12345 (dataloader seed only)",
        "mean_diff_bpb": round(float(d.mean()), 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "p_value": p,
        "p_display": fmt_p(p, args.n_boot),
    }
    print(f"\nseed variance (12536 - 12345): {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {fmt_p(p, args.n_boot)}")

    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
