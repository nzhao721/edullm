#!/usr/bin/env python3
"""Fit Ye et al.'s data mixing law to the 24 measured task losses and optimize the mixture.

Implements the three pieces of arXiv:2403.16952 that a single-scale pilot can
support, with task loss (bits-per-byte, the OLMo-ladder metric) as the target
instead of LM validation loss:

  1. **Mixing law** (their Eq. 2), per target task family i:

         L_i(r) = c_i + k_i * exp( sum_j t_ij * r_j )

     9 free parameters per target (c_i, k_i, and one t_ij per domain) fitted from
     24 mixture observations. Fitted on a Huber loss from many random starts,
     with c_i and k_i reparameterized as exponentials so positivity is structural
     rather than a constraint the optimizer can violate.

  2. **Step law** (their nested pipeline, stage 1), per mixture:

         L(s) = L_inf + A * s^(-alpha)

     fitted to the in-run task-loss curve. This converts a short probe into an
     estimate of the loss at a longer budget, so the mixing law can be fitted on
     extrapolated targets rather than only on what the pilot directly measured.

  3. **Simplex optimization**: minimize the fitted law over the mixture simplex,
     subject to per-domain availability caps (a mixture that needs more tokens of a
     domain than the corpus has is not a usable answer).

Because the answer is an extrapolation from 24 points to a 7-dimensional simplex,
``fit`` always reports leave-one-out cross-validation. LOO error is the honest
measure of whether the pilot bought enough signal, and it is the number to look at
before trusting the recommended mixture.

Subcommands:
    collect   gather per-run task_loss_final.json / task_loss.jsonl into one file
    fit       fit the mixing law, cross-validate, and optimize over the simplex
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, DOMAINS, load_mixtures, macro_curve, task_family

DATA_NAME = "mixlaw_data.json"


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def cmd_collect(args: argparse.Namespace) -> None:
    mixtures = {m.id: m for m in load_mixtures()}

    runs = []
    missing = []
    for mix_id, mix in sorted(mixtures.items()):
        progress = args.runs_dir / mix.run_name / "progress"
        final = progress / "task_loss_final.json"
        if not final.is_file():
            missing.append(mix.run_name)
            continue

        payload = json.loads(final.read_text(encoding="utf-8"))
        labels = {
            k: float(v)
            for k, v in payload["labels"].items()
            if k in CURVE_TASK_LOSS_LABELS
        }
        families = {
            k: float(v) for k, v in payload["task_families"].items() if k in CURVE_FAMILIES
        }
        if set(families) != set(CURVE_FAMILIES):
            raise SystemExit(
                f"{mix.run_name}: expected curve families {CURVE_FAMILIES}, got {sorted(families)}"
            )
        curve = []
        curve_path = progress / "task_loss.jsonl"
        if curve_path.is_file():
            for line in curve_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    curve.append(json.loads(line))

        meta_path = progress / "run_meta.json"
        runs.append(
            {
                "id": mix_id,
                "tag": mix.tag,
                "run_name": mix.run_name,
                "weights": mix.weights,
                "task_loss_labels": labels,
                "task_loss_families": families,
                "macro_mean": macro_curve(families),
                "curve": curve,
                "meta": json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.is_file()
                else None,
            }
        )

    if missing:
        print(f"warning: {len(missing)} runs have no final eval yet: {' '.join(missing)}")
    if not runs:
        raise SystemExit("no completed runs found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"domain_order": list(DOMAINS), "runs": runs}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"collected {len(runs)} runs -> {args.out}")


# --------------------------------------------------------------------------- #
# mixing law
# --------------------------------------------------------------------------- #
def _predict(theta: np.ndarray, R: np.ndarray) -> np.ndarray:
    """L(r) = exp(a) + exp(b) * exp(R @ t), with theta = [a, b, t...]."""
    a, b, t = theta[0], theta[1], theta[2:]
    # Clip the exponent so a wild trial point returns a large residual instead of inf.
    return np.exp(a) + np.exp(b) * np.exp(np.clip(R @ t, -60.0, 60.0))


def _huber_residuals(theta: np.ndarray, R: np.ndarray, y: np.ndarray, delta: float) -> np.ndarray:
    """Residuals scaled so that least_squares' sum of squares equals a Huber loss."""
    resid = _predict(theta, R) - y
    absr = np.abs(resid)
    small = absr <= delta
    out = np.empty_like(resid)
    out[small] = resid[small]
    # sqrt(2*delta*(|r| - delta/2)) squares back to the linear Huber arm.
    out[~small] = np.sign(resid[~small]) * np.sqrt(2.0 * delta * (absr[~small] - delta / 2.0))
    return out


def fit_mixing_law(
    R: np.ndarray,
    y: np.ndarray,
    n_starts: int = 256,
    delta: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Fit one target. Returns (theta, rmse). Requires scipy."""
    from scipy.optimize import least_squares

    n_domains = R.shape[1]
    rng = np.random.default_rng(seed)

    y_min, y_max = float(y.min()), float(y.max())
    spread = max(y_max - y_min, 1e-6)

    best_theta: Optional[np.ndarray] = None
    best_cost = math.inf
    for _ in range(n_starts):
        # c just below the observed minimum, k around the observed spread, t small.
        a0 = math.log(max(y_min - spread * rng.uniform(0.05, 1.5), 1e-4))
        b0 = math.log(spread * rng.uniform(0.1, 5.0))
        t0 = rng.normal(0.0, 1.0, size=n_domains)
        try:
            res = least_squares(
                _huber_residuals,
                np.concatenate([[a0, b0], t0]),
                args=(R, y, delta),
                method="trf",
                max_nfev=20_000,
            )
        except Exception:
            continue
        if res.cost < best_cost:
            best_cost = float(res.cost)
            best_theta = res.x

    if best_theta is None:
        raise SystemExit("mixing-law fit failed from every start")

    rmse = float(np.sqrt(np.mean((_predict(best_theta, R) - y) ** 2)))
    return best_theta, rmse


def loo_cv(R: np.ndarray, y: np.ndarray, n_starts: int, delta: float, seed: int) -> dict:
    """Leave-one-out error: refit on 23 points, predict the 24th."""
    preds = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        theta, _ = fit_mixing_law(R[keep], y[keep], n_starts=n_starts, delta=delta, seed=seed + i)
        preds[i] = _predict(theta, R[i : i + 1])[0]
    err = preds - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        # Spread of the targets, for scale: LOO RMSE only means something relative
        # to how much the mixtures actually move the loss.
        "target_std": float(np.std(y)),
        "predictions": preds.tolist(),
    }


# --------------------------------------------------------------------------- #
# step law
# --------------------------------------------------------------------------- #
def fit_step_law(
    steps: np.ndarray,
    losses: np.ndarray,
    seed: int = 0,
    n_starts: int = 8,
    max_nfev: int = 2_000,
) -> Optional[dict]:
    """L(s) = L_inf + A * s^(-alpha). Returns None if there are too few points."""
    if len(steps) < 5:
        return None

    ymin = float(losses.min())
    span = max(float(losses.max() - ymin), 1e-3)

    def model(s: np.ndarray, l_inf: float, a: float, alpha: float) -> np.ndarray:
        return l_inf + a * np.power(s, -alpha)

    from scipy.optimize import curve_fit, least_squares

    try:
        p0 = (ymin * 0.98, span, 0.25)
        bounds = ([ymin * 0.5, 0.0, 0.01], [ymin * 1.001, span * 50.0, 2.0])
        p, _ = curve_fit(model, steps, losses, p0=p0, bounds=bounds, maxfev=5_000)
        resid = model(steps, *p) - losses
        return {
            "L_inf": float(p[0]),
            "A": float(p[1]),
            "alpha": float(p[2]),
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "n_points": int(len(steps)),
        }
    except Exception:
        pass

    rng = np.random.default_rng(seed)

    def resid(p: np.ndarray) -> np.ndarray:
        l_inf, log_a, log_alpha = p
        return l_inf + np.exp(log_a) * steps ** (-np.exp(log_alpha)) - losses

    best, best_cost = None, math.inf
    for _ in range(n_starts):
        p0 = np.array(
            [
                ymin * rng.uniform(0.5, 0.999),
                math.log(max(span * rng.uniform(0.5, 20.0), 1e-3)),
                math.log(rng.uniform(0.05, 1.0)),
            ]
        )
        try:
            res = least_squares(resid, p0, method="trf", max_nfev=max_nfev)
        except Exception:
            continue
        if res.cost < best_cost:
            best_cost, best = float(res.cost), res.x

    if best is None:
        return None
    l_inf, log_a, log_alpha = best
    return {
        "L_inf": float(l_inf),
        "A": float(np.exp(log_a)),
        "alpha": float(np.exp(log_alpha)),
        "rmse": float(np.sqrt(np.mean(resid(best) ** 2))),
        "n_points": int(len(steps)),
    }


def extrapolate(step_law: dict, step: int) -> float:
    return step_law["L_inf"] + step_law["A"] * step ** (-step_law["alpha"])


# --------------------------------------------------------------------------- #
# simplex optimization
# --------------------------------------------------------------------------- #
def project_to_box_simplex(
    r: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    max_iter: int = 40,
) -> np.ndarray | None:
    """Clip-renormalize iterate until weights lie in the box and sum to 1."""
    r = np.clip(np.asarray(r, dtype=float), floors, caps)
    for _ in range(max_iter):
        total = float(r.sum())
        if total <= 0:
            return None
        r = r / total
        clipped = np.clip(r, floors, caps)
        if np.allclose(clipped, r, rtol=0, atol=1e-10):
            return clipped
        r = clipped
    return None


def sample_feasible_mixtures(
    rng: np.random.Generator,
    floors: Sequence[float],
    caps: Sequence[float],
    n: int,
) -> np.ndarray:
    """Sample mixture weights inside per-domain box constraints (sum to 1)."""
    n_domains = len(floors)
    floor_v = np.asarray(floors, dtype=float)
    cap_v = np.asarray(caps, dtype=float)
    if np.all(floor_v <= 1e-12) and np.all(cap_v >= 1.0 - 1e-12):
        return rng.dirichlet(np.ones(n_domains), size=n)

    if float(floor_v.sum()) > 1.0 + 1e-9:
        raise RuntimeError(f"infeasible mixture box: floor sum={floor_v.sum()}")

    samples = np.zeros((n, n_domains), dtype=float)
    for i in range(n):
        for _ in range(10_000):
            r0 = rng.dirichlet(np.ones(n_domains))
            r = project_to_box_simplex(r0, floor_v, cap_v)
            if r is not None:
                samples[i] = r
                break
        else:
            raise RuntimeError(
                "failed to sample feasible mixture after 10000 tries "
                f"(floors={list(floors)}, caps={list(caps)})"
            )
    return samples


def optimize_simplex(
    objective: Callable[[np.ndarray], float],
    n_domains: int,
    caps: Sequence[float],
    floors: Sequence[float],
    n_starts: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    bounds = [(float(lo), float(hi)) for lo, hi in zip(floors, caps)]
    constraint = {"type": "eq", "fun": lambda r: float(np.sum(r) - 1.0)}

    best_r, best_val = None, math.inf
    for _ in range(n_starts):
        # Dirichlet starts spread over the simplex, then clipped into the box.
        r0 = rng.dirichlet(np.ones(n_domains) * rng.uniform(0.2, 3.0))
        r0 = np.clip(r0, [b[0] for b in bounds], [b[1] for b in bounds])
        total = r0.sum()
        if total <= 0:
            continue
        r0 = r0 / total
        try:
            res = minimize(
                objective,
                r0,
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint],
                options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception:
            continue
        if not res.success:
            continue
        r = np.clip(res.x, 0.0, 1.0)
        r = r / r.sum()
        val = objective(r)
        if val < best_val:
            best_val, best_r = float(val), r

    if best_r is None:
        raise SystemExit("simplex optimization failed from every start")
    return best_r, best_val


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def cmd_fit(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_text(encoding="utf-8"))
    runs = data["runs"]
    if len(runs) < 12:
        print(f"warning: fitting 9 parameters per target from only {len(runs)} points")

    R = np.array([[run["weights"][d] for d in DOMAINS] for run in runs], dtype=float)

    # Fit only the six in-run curve families (Chinchilla targets when extrapolating).
    families = list(CURVE_FAMILIES)
    if args.targets:
        unknown = [t for t in args.targets if t not in families]
        if unknown:
            raise SystemExit(f"unknown target families {unknown}; available: {families}")
        families = list(args.targets)

    # Optional step-law extrapolation of each run's target to a longer budget.
    step_laws: dict[str, dict] = {}
    if args.extrapolate_to_step:
        for run in runs:
            per_family: dict[str, list[tuple[int, float]]] = {}
            for point in run["curve"]:
                for label, value in point["task_loss_bpb"].items():
                    per_family.setdefault(task_family(label), []).append((point["step"], value))
            for fam, points in per_family.items():
                points.sort()
                steps = np.array([p[0] for p in points], dtype=float)
                losses = np.array([p[1] for p in points], dtype=float)
                law = fit_step_law(steps, losses, seed=args.seed + run["id"])
                if law is not None:
                    step_laws[f"{run['run_name']}::{fam}"] = law

    report: dict = {
        "n_runs": len(runs),
        "domain_order": list(DOMAINS),
        "targets": {},
        "step_laws": step_laws or None,
        "extrapolate_to_step": args.extrapolate_to_step,
    }

    per_family_theta: dict[str, np.ndarray] = {}
    for fam in families:
        y = np.array([run["task_loss_families"][fam] for run in runs], dtype=float)

        if args.extrapolate_to_step:
            extrapolated = []
            for run in runs:
                law = step_laws.get(f"{run['run_name']}::{fam}")
                measured = run["task_loss_families"][fam]
                extrapolated.append(
                    extrapolate(law, args.extrapolate_to_step) if law else measured
                )
            y = np.array(extrapolated, dtype=float)

        theta, rmse = fit_mixing_law(R, y, n_starts=args.n_starts, delta=args.huber_delta, seed=args.seed)
        per_family_theta[fam] = theta

        entry = {
            "c": float(np.exp(theta[0])),
            "k": float(np.exp(theta[1])),
            "t": {d: float(v) for d, v in zip(DOMAINS, theta[2:])},
            "fit_rmse": rmse,
            "observed": {"min": float(y.min()), "max": float(y.max()), "std": float(y.std())},
        }
        if not args.skip_cv:
            entry["loo_cv"] = loo_cv(R, y, args.n_starts, args.huber_delta, args.seed)
        report["targets"][fam] = entry

        print(f"\n{fam}")
        print(f"  observed task loss (bpb): {y.min():.4f} .. {y.max():.4f} (std {y.std():.4f})")
        print(f"  fit RMSE: {rmse:.5f}")
        if "loo_cv" in entry:
            cv = entry["loo_cv"]
            ratio = cv["rmse"] / max(cv["target_std"], 1e-12)
            print(f"  LOO RMSE: {cv['rmse']:.5f} ({ratio:.2%} of target std), max |err| {cv['max_abs']:.5f}")
        print("  t (more negative = this domain lowers the task loss more):")
        for d, v in sorted(entry["t"].items(), key=lambda kv: kv[1]):
            print(f"    {d:<18} {v:+.4f}")

    # --- optimize the macro-average over task families ---------------------- #
    weights = {fam: 1.0 / len(families) for fam in families}
    if args.target_weights:
        for spec in args.target_weights:
            fam, _, raw = spec.partition("=")
            if fam not in families:
                raise SystemExit(f"--target-weights names unknown family {fam}")
            weights[fam] = float(raw)
        total = sum(weights.values())
        weights = {fam: w / total for fam, w in weights.items()}

    def objective(r: np.ndarray) -> float:
        return float(
            sum(weights[fam] * _predict(per_family_theta[fam], r[None, :])[0] for fam in families)
        )

    caps = [1.0] * len(DOMAINS)
    floors = [0.0] * len(DOMAINS)
    for spec in args.caps or []:
        d, _, raw = spec.partition("=")
        if d not in DOMAINS:
            raise SystemExit(f"--caps names unknown domain {d}")
        caps[DOMAINS.index(d)] = float(raw)
    for spec in args.floors or []:
        d, _, raw = spec.partition("=")
        if d not in DOMAINS:
            raise SystemExit(f"--floors names unknown domain {d}")
        floors[DOMAINS.index(d)] = float(raw)

    r_star, val = optimize_simplex(
        objective, len(DOMAINS), caps, floors, n_starts=args.opt_starts, seed=args.seed
    )

    observed_best = min(runs, key=lambda run: sum(weights[f] * run["task_loss_families"][f] for f in families))
    observed_best_val = sum(weights[f] * observed_best["task_loss_families"][f] for f in families)

    report["optimization"] = {
        "target_weights": weights,
        "caps": {d: c for d, c in zip(DOMAINS, caps)},
        "floors": {d: f for d, f in zip(DOMAINS, floors)},
        "optimal_weights": {d: float(v) for d, v in zip(DOMAINS, r_star)},
        "predicted_task_loss_bpb": val,
        "best_observed_run": observed_best["run_name"],
        "best_observed_task_loss_bpb": observed_best_val,
        "predicted_improvement": observed_best_val - val,
    }

    print("\noptimal mixture (predicted):")
    for d, v in sorted(zip(DOMAINS, r_star), key=lambda kv: -kv[1]):
        print(f"  {d:<18} {v:.4f}")
    print(f"\npredicted task loss (bpb): {val:.5f}")
    print(f"best measured mixture:     {observed_best['run_name']} at {observed_best_val:.5f}")
    print(f"predicted improvement:     {observed_best_val - val:+.5f}")
    if not args.skip_cv:
        worst_cv = max(report["targets"][f]["loo_cv"]["rmse"] for f in families)
        if worst_cv > abs(observed_best_val - val):
            print(
                "\nCAUTION: the predicted gain is smaller than the worst LOO error, so this "
                "mixture is not distinguishable from the best measured one at the current "
                "number of pilot runs."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)

    c = sub.add_parser("collect", help="gather per-run results into one file")
    c.set_defaults(func=cmd_collect)
    c.add_argument("--runs-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, default=Path(DATA_NAME))

    f = sub.add_parser("fit", help="fit the mixing law, cross-validate, optimize")
    f.set_defaults(func=cmd_fit)
    f.add_argument("--data", type=Path, default=Path(DATA_NAME))
    f.add_argument("--out", type=Path, default=Path("mixlaw_fit.json"))
    f.add_argument("--targets", nargs="*", default=None, help="Task families to fit (default: all)")
    f.add_argument(
        "--target-weights",
        nargs="*",
        default=None,
        metavar="FAMILY=W",
        help="Weights for the optimized objective (default: uniform over families)",
    )
    f.add_argument(
        "--caps",
        nargs="*",
        default=None,
        metavar="DOMAIN=MAX",
        help="Per-domain upper bounds, e.g. availability limits: wiki=0.122",
    )
    f.add_argument("--floors", nargs="*", default=None, metavar="DOMAIN=MIN")
    f.add_argument(
        "--extrapolate-to-step",
        type=int,
        default=None,
        help="Fit the step law per run and fit the mixing law on loss extrapolated to this step",
    )
    f.add_argument("--n-starts", type=int, default=256, help="Random starts per mixing-law fit")
    f.add_argument("--opt-starts", type=int, default=512)
    f.add_argument("--huber-delta", type=float, default=1e-3)
    f.add_argument("--skip-cv", action="store_true", help="Skip leave-one-out CV (much faster)")
    f.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
