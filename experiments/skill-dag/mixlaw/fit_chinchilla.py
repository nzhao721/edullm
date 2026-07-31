#!/usr/bin/env python3
"""Fit the Chinchilla-projected mixing law with regularized k and t.

Among multi-start solutions that still fit well, prefer parameterizations where
|t_ij| and k_i stay in a physically plausible range, then optimize / sample
near-optimal mixtures under those parameters.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

from fit_mixing_law import DOMAINS, _predict, optimize_simplex, sample_feasible_mixtures
from mixlaw_common import (
    CURVE_FAMILIES,
    MIXTURE_OPT_CONSTRAINTS,
    NEAR_OPT_DOMAIN_CAPS,
    NEAR_OPT_DOMAIN_FLOORS,
    macro_curve,
)

DATA = Path("mixlaw_data.json")
CHIN = Path("mixlaw_chinchilla_extrapolated.json")
OUT = Path("mixlaw_fit_chinchilla.json")

# Soft preferences for parsimonious solutions.
T_SOFT = 4.0          # prefer |t| below this; soft L2 beyond
T_HARD = 8.0          # hard bound on each t_j
K_RATIO_SOFT = 20.0   # prefer k / y_std below this
LAMBDA_T = 0.02       # weight on soft |t| penalty in residual stack
LAMBDA_K = 0.05       # weight on soft log(k) penalty
RMSE_SLACK = 1.35     # accept solutions up to this × best unconstrained RMSE
N_STARTS = 128
SEED = 0


def build_targets(data: dict, chin: dict) -> tuple[list[str], dict[str, np.ndarray]]:
    """Chinchilla-extrapolated losses for the six in-run curve families."""
    chin_by = {r["run_name"]: r for r in chin["runs"]}
    families = list(chin["curve_families"])
    y_by_fam: dict[str, np.ndarray] = {}
    for fam in families:
        vals = []
        for run in data["runs"]:
            entry = chin_by[run["run_name"]]["families"][fam]
            if entry.get("chinchilla") is None:
                raise SystemExit(f"{run['run_name']}::{fam} missing Chinchilla extrapolation")
            vals.append(float(entry["chinchilla"]))
        y_by_fam[fam] = np.array(vals, dtype=float)
    return families, y_by_fam


def moderation_score(theta: np.ndarray, y_std: float) -> float:
    """Lower is better: small |t| and k on the order of the observed spread."""
    k = float(np.exp(theta[1]))
    t = theta[2:]
    t_pen = float(np.sqrt(np.mean(t**2))) + 0.25 * float(np.max(np.abs(t)))
    k_ratio = k / max(y_std, 1e-6)
    k_pen = abs(math.log10(max(k_ratio, 1e-12)))  # prefer k ~ y_std
    # Extra hit if k is huge relative to spread.
    if k_ratio > K_RATIO_SOFT:
        k_pen += math.log10(k_ratio / K_RATIO_SOFT)
    return t_pen + 0.5 * k_pen


def regularized_residuals(
    theta: np.ndarray, R: np.ndarray, y: np.ndarray, delta: float, y_std: float
) -> np.ndarray:
    pred = _predict(theta, R)
    resid = pred - y
    absr = np.abs(resid)
    small = absr <= delta
    huber = np.empty_like(resid)
    huber[small] = resid[small]
    huber[~small] = np.sign(resid[~small]) * np.sqrt(
        2.0 * delta * (absr[~small] - delta / 2.0)
    )

    t = theta[2:]
    # Soft hinge beyond T_SOFT.
    t_excess = np.maximum(np.abs(t) - T_SOFT, 0.0)
    t_pen = np.sqrt(LAMBDA_T) * t_excess

    k = float(np.exp(theta[1]))
    k_ratio = k / max(y_std, 1e-6)
    log_excess = max(math.log(k_ratio / K_RATIO_SOFT), 0.0) if k_ratio > K_RATIO_SOFT else 0.0
    # Also gently pull k toward y_std when it is already near the target scale.
    log_pull = math.log(max(k_ratio, 1e-12))
    k_pen = np.array([math.sqrt(LAMBDA_K) * (log_excess + 0.15 * log_pull)])

    return np.concatenate([huber, t_pen, k_pen])


def fit_family_candidates(
    R: np.ndarray, y: np.ndarray, seed: int
) -> list[dict]:
    """Return multi-start solutions ranked by (rmse, moderation)."""
    rng = np.random.default_rng(seed)
    y_min, y_max = float(y.min()), float(y.max())
    spread = max(y_max - y_min, 1e-6)
    y_std = float(y.std())
    n_dom = R.shape[1]

    # Bounds: c > 0 via a=log(c); keep c below observed min; k positive; |t|<=T_HARD.
    # a: c in (0.05, y_min]
    # b: k in [1e-6, 50 * spread]
    lo = np.concatenate(
        [
            [math.log(0.05), math.log(1e-6)],
            np.full(n_dom, -T_HARD),
        ]
    )
    hi = np.concatenate(
        [
            [math.log(max(y_min, 0.1)), math.log(50.0 * spread)],
            np.full(n_dom, T_HARD),
        ]
    )

    sols: list[dict] = []
    for i in range(N_STARTS):
        a0 = math.log(max(y_min - spread * rng.uniform(0.05, 0.8), 0.05))
        # Prefer k starts near the observed spread.
        b0 = math.log(spread * rng.uniform(0.2, 5.0))
        t0 = rng.normal(0.0, 1.5, size=n_dom)
        t0 = np.clip(t0, -T_HARD, T_HARD)
        x0 = np.concatenate([[a0, b0], t0])
        try:
            res = least_squares(
                regularized_residuals,
                x0,
                args=(R, y, 1e-3, y_std),
                bounds=(lo, hi),
                method="trf",
                max_nfev=8_000,
            )
        except Exception:
            continue
        theta = res.x
        rmse = float(np.sqrt(np.mean((_predict(theta, R) - y) ** 2)))
        sols.append(
            {
                "theta": theta,
                "rmse": rmse,
                "moderation": moderation_score(theta, y_std),
                "c": float(np.exp(theta[0])),
                "k": float(np.exp(theta[1])),
                "t": {d: float(v) for d, v in zip(DOMAINS, theta[2:])},
                "max_abs_t": float(np.max(np.abs(theta[2:]))),
                "k_over_std": float(np.exp(theta[1]) / max(y_std, 1e-6)),
            }
        )

    if not sols:
        raise SystemExit("no successful starts")

    best_rmse = min(s["rmse"] for s in sols)
    # Keep well-fitting solutions, then pick the most parsimonious.
    eligible = [s for s in sols if s["rmse"] <= best_rmse * RMSE_SLACK]
    eligible.sort(key=lambda s: (s["moderation"], s["rmse"]))
    return eligible


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    chin = json.loads(CHIN.read_text(encoding="utf-8"))
    families, y_by_fam = build_targets(data, chin)
    runs = data["runs"]
    R = np.array([[run["weights"][d] for d in DOMAINS] for run in runs], dtype=float)

    report: dict = {
        "n_runs": len(runs),
        "domain_order": list(DOMAINS),
        "extrapolate_to_step": chin["chinchilla_steps"],
        "regularization": {
            "t_soft": T_SOFT,
            "t_hard": T_HARD,
            "k_ratio_soft": K_RATIO_SOFT,
            "lambda_t": LAMBDA_T,
            "lambda_k": LAMBDA_K,
            "rmse_slack": RMSE_SLACK,
            "n_starts": N_STARTS,
        },
        "targets": {},
        "candidates": {},
    }

    thetas: dict[str, np.ndarray] = {}
    print("=== Regularized Chinchilla mixing-law fit ===\n")
    for i, fam in enumerate(families):
        y = y_by_fam[fam]
        cands = fit_family_candidates(R, y, seed=SEED + 17 * i)
        pick = cands[0]
        thetas[fam] = pick["theta"]
        report["targets"][fam] = {
            "c": pick["c"],
            "k": pick["k"],
            "t": pick["t"],
            "fit_rmse": pick["rmse"],
            "moderation": pick["moderation"],
            "max_abs_t": pick["max_abs_t"],
            "k_over_std": pick["k_over_std"],
            "observed": {
                "min": float(y.min()),
                "max": float(y.max()),
                "std": float(y.std()),
            },
            "n_eligible": len(cands),
            "best_unconstrained_rmse_proxy": min(c["rmse"] for c in cands),
        }
        # Keep a few alternate near-optimal candidates for inspection.
        report["candidates"][fam] = [
            {
                "c": c["c"],
                "k": c["k"],
                "t": c["t"],
                "fit_rmse": c["rmse"],
                "moderation": c["moderation"],
                "max_abs_t": c["max_abs_t"],
                "k_over_std": c["k_over_std"],
            }
            for c in cands[:5]
        ]
        print(
            f"{fam:22s}  c={pick['c']:.4f}  k={pick['k']:.4g}  "
            f"max|t|={pick['max_abs_t']:.2f}  k/std={pick['k_over_std']:.2f}  "
            f"rmse={pick['rmse']:.4f}  (eligible {len(cands)})"
        )
        print("  t:", " ".join(f"{d[:4]}={pick['t'][d]:+.2f}" for d in DOMAINS))

    def objective(r: np.ndarray) -> float:
        return float(
            sum(_predict(thetas[f], r[None, :])[0] for f in families) / len(families)
        )

    # Uncapped optimum + capped / balanced optima.
    optima = {}
    for name, caps, floors in MIXTURE_OPT_CONSTRAINTS:
        r_star, val = optimize_simplex(
            objective, 7, caps, floors, n_starts=512, seed=SEED
        )
        optima[name] = {
            "weights": {d: float(v) for d, v in zip(DOMAINS, r_star)},
            "predicted_macro": float(val),
            "max_w": float(np.max(r_star)),
        }

    # Score pilot mixes.
    pilot_rows = []
    for run in runs:
        r = np.array([run["weights"][d] for d in DOMAINS], dtype=float)
        pred = objective(r)
        pilot_rows.append(
            {
                "run_name": run["run_name"],
                "tag": run["tag"],
                "weights": run["weights"],
                "predicted_macro": pred,
                "max_w": float(max(run["weights"].values())),
                "measured_curve_6": macro_curve(run["task_loss_families"]),
            }
        )
    pilot_rows.sort(key=lambda x: x["predicted_macro"])

    # Near-optimal samples: all domains >= 1%, within +0.04 bpb of uncapped optimum.
    rng = np.random.default_rng(SEED + 99)
    best_pred = optima["uncapped"]["predicted_macro"]
    floor_v = np.asarray(NEAR_OPT_DOMAIN_FLOORS, dtype=float)
    cap_v = np.asarray(NEAR_OPT_DOMAIN_CAPS, dtype=float)
    samples = []
    for _ in range(80_000):
        r = sample_feasible_mixtures(rng, floor_v, cap_v, 1)[0]
        val = objective(r)
        if val <= best_pred + 0.04:
            samples.append((val, r))
    samples.sort(key=lambda x: x[0])

    # Deduplicate samples.
    novel = []
    seen: set[tuple[float, ...]] = set()
    for val, r in samples:
        key = tuple(round(float(x), 2) for x in r)
        if key in seen:
            continue
        seen.add(key)
        novel.append(
            {
                "predicted_macro": float(val),
                "min_w": float(r.min()),
                "max_w": float(r.max()),
                "weights": {d: float(v) for d, v in zip(DOMAINS, r)},
            }
        )
        if len(novel) >= 8:
            break

    report["optimization"] = optima
    report["pilot_ranked"] = pilot_rows
    report["near_optimal_balanced_samples"] = novel

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

    print("\n=== Optima (regularized model) ===")
    for name, o in optima.items():
        w = o["weights"]
        print(
            f"{name:12s}  pred={o['predicted_macro']:.4f}  max_w={o['max_w']:.3f}  "
            + " ".join(f"{d[:4]}={w[d]:.3f}" for d in DOMAINS if w[d] >= 0.01)
        )

    print("\n=== Top pilot mixes under regularized model ===")
    for row in pilot_rows[:8]:
        print(
            f"  {row['run_name']} ({row['tag']:10s}) pred={row['predicted_macro']:.4f} "
            f"max_w={row['max_w']:.3f} measured6={row['measured_curve_6']:.4f}"
        )

    print("\n=== Near-optimal balanced samples (min_w>=0.01, within +0.04 of uncapped) ===")
    for i, s in enumerate(novel, 1):
        w = s["weights"]
        print(
            f"  #{i} pred={s['predicted_macro']:.4f} max_w={s['max_w']:.3f}  "
            + " ".join(f"{d[:4]}={w[d]:.3f}" for d in DOMAINS if w[d] >= 0.02)
        )


if __name__ == "__main__":
    main()
