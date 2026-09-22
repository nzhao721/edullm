"""Power-law fit for 370M MixLaw validation curves.

Fits y = a + b / step^alpha on eval points with step >= min_step, following
the methodology documented in experiments/skill-dag/mixlaw/README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float
    a: float
    b: float
    sse: float
    min_step: int
    final_step: int

    def predict(self, step: float | np.ndarray) -> float | np.ndarray:
        return self.a + self.b * np.power(step, -self.alpha)

    @property
    def fitted_final(self) -> float:
        return float(self.predict(self.final_step))

    def step_for_loss(self, target: float) -> float:
        """Invert the fitted curve to find the step that reaches target loss."""
        if self.b <= 0:
            raise ValueError("Power-law b must be positive for inversion.")
        ratio = (target - self.a) / self.b
        if ratio <= 0:
            raise ValueError(
                f"Target {target:.6f} is not reachable above asymptote {self.a:.6f}."
            )
        return float(ratio ** (-1.0 / self.alpha))


def fit_power_law(
    steps: Iterable[int],
    losses: Iterable[float],
    *,
    min_step: int = 1000,
    final_step: int = 2384,
    alpha_grid: np.ndarray | None = None,
) -> PowerLawFit:
    step_arr = np.asarray(list(steps), dtype=float)
    loss_arr = np.asarray(list(losses), dtype=float)
    mask = step_arr >= min_step
    s = step_arr[mask]
    y = loss_arr[mask]
    if len(s) < 3:
        raise ValueError("Need at least three points with step >= min_step.")

    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 3.0, 400)

    best: PowerLawFit | None = None
    design = np.ones((len(s), 2))
    for alpha in alpha_grid:
        x = s ** (-alpha)
        design[:, 1] = x
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        a, b = coeffs
        pred = a + b * x
        sse = float(np.sum((y - pred) ** 2))
        candidate = PowerLawFit(
            alpha=float(alpha),
            a=float(a),
            b=float(b),
            sse=sse,
            min_step=min_step,
            final_step=final_step,
        )
        if best is None or candidate.sse < best.sse:
            best = candidate
    assert best is not None
    return best


def _fit_ab(steps: np.ndarray, losses: np.ndarray, alpha: float) -> tuple[float, float]:
    x = steps ** (-alpha)
    design = np.column_stack([np.ones(len(steps)), x])
    a, b = np.linalg.lstsq(design, losses, rcond=None)[0]
    return float(a), float(b)


def residual_bootstrap_ci(
    steps: Iterable[int],
    losses: Iterable[float],
    *,
    min_step: int = 1000,
    final_step: int = 2384,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Return (fitted_final, observed_final, ci_lo, ci_hi) via residual bootstrap.

    The power-law exponent alpha is fixed from the initial fit; each bootstrap
    draw resamples residuals and refits (a, b) only.
    """
    step_list = list(steps)
    loss_list = list(losses)
    fit = fit_power_law(step_list, loss_list, min_step=min_step, final_step=final_step)

    step_arr = np.asarray(step_list, dtype=float)
    loss_arr = np.asarray(loss_list, dtype=float)
    mask = step_arr >= min_step
    s = step_arr[mask]
    y = loss_arr[mask]
    if len(s) < 3:
        raise ValueError("Need at least three points with step >= min_step.")

    pred = fit.a + fit.b * s ** (-fit.alpha)
    resid = y - pred
    design = np.column_stack([np.ones(len(s)), s ** (-fit.alpha)])

    rng = np.random.default_rng(seed)
    finals = np.empty(n_boot)
    for i in range(n_boot):
        y_boot = pred + resid[rng.integers(0, len(resid), len(resid))]
        a, b = np.linalg.lstsq(design, y_boot, rcond=None)[0]
        finals[i] = a + b * final_step ** (-fit.alpha)

    ci_lo, ci_hi = np.percentile(finals, [2.5, 97.5])
    observed_pairs = sorted(zip(step_list, loss_list, strict=True))
    observed = float(observed_pairs[-1][1])
    return fit.fitted_final, observed, float(ci_lo), float(ci_hi)


def first_step_at_or_below(
    steps: Iterable[int],
    losses: Iterable[float],
    threshold: float,
) -> float:
    """Return the first training step where the measured curve reaches threshold."""
    pairs = sorted(zip(steps, losses, strict=True))
    for (s0, y0), (s1, y1) in zip(pairs, pairs[1:]):
        if y0 <= threshold:
            return float(s0)
        if y0 > threshold >= y1:
            if math.isclose(y0, y1):
                return float(s1)
            frac = (y0 - threshold) / (y0 - y1)
            return float(s0 + frac * (s1 - s0))
    raise ValueError(f"Curve never reaches threshold {threshold:.6f}.")
