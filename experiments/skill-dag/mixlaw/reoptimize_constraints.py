#!/usr/bin/env python3
"""Re-run mixture optimization and near-optimal sampling on existing fits.

Does not refit family models or rerun hyperparameter / LOO search. Mixing-law
parameters are read from JSON; LightGBM boosters are reloaded with the saved
hyperparameters only so new mixture weights can be scored.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from fit_chinchilla import SEED, build_targets
from fit_lightgbm_chinchilla import (
    DOMAINS,
    OPT_CONSTRAINTS,
    OPT_SEED_OFFSET,
    fit_family_model,
    macro_predict_batch,
    optimize_simplex_global,
)
from fit_mixing_law import optimize_simplex, sample_feasible_mixtures
from mixlaw_common import macro_curve

ROOT = Path(__file__).parent
DATA = ROOT / "mixlaw_data.json"
CHIN = ROOT / "mixlaw_chinchilla_extrapolated.json"
ML_FIT = ROOT / "mixlaw_fit_chinchilla.json"
LGB_FIT = ROOT / "mixlaw_fit_lightgbm_chinchilla.json"

MIN_DOMAIN_WEIGHT = 0.01
NEAR_OPT_BAND = 0.04
NEAR_OPT_DRAWS = 200_000
NEAR_OPT_KEEP = 8
MIN_LINF_FROM_OPTIMA = 0.08
MIN_L1_FROM_OPTIMA = 0.20
MIN_LINF_AMONG_KEPT = 0.06
MIN_L1_AMONG_KEPT = 0.15


def theta_from_target(target: dict) -> np.ndarray:
    a = math.log(target["c"])
    b = math.log(target["k"])
    t = [target["t"][d] for d in DOMAINS]
    return np.array([a, b, *t], dtype=float)


def optima_weight_vectors(optima: dict) -> list[np.ndarray]:
    return [
        np.array([entry["weights"][d] for d in DOMAINS], dtype=float)
        for entry in optima.values()
    ]


def far_from_reference(
    r: np.ndarray,
    references: list[np.ndarray],
    min_linf: float,
    min_l1: float,
) -> bool:
    for ref in references:
        if float(np.max(np.abs(r - ref))) < min_linf:
            return False
        if float(np.sum(np.abs(r - ref))) < min_l1:
            return False
    return True


def sample_near_optimal(
    objective,
    uncapped_pred: float,
    rng: np.random.Generator,
    floors: list[float],
    caps: list[float],
    *,
    opt_vectors: list[np.ndarray],
) -> list[dict]:
    floor_v = np.asarray(floors, dtype=float)
    cap_v = np.asarray(caps, dtype=float)
    samples: list[tuple[float, np.ndarray]] = []
    for _ in range(NEAR_OPT_DRAWS):
        r = sample_feasible_mixtures(rng, floor_v, cap_v, 1)[0]
        if not far_from_reference(
            r, opt_vectors, MIN_LINF_FROM_OPTIMA, MIN_L1_FROM_OPTIMA
        ):
            continue
        val = float(objective(r))
        if val <= uncapped_pred + NEAR_OPT_BAND:
            samples.append((val, r))
    samples.sort(key=lambda x: x[0])

    out: list[dict] = []
    kept_vectors: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for val, r in samples:
        key = tuple(round(float(x), 2) for x in r)
        if key in seen:
            continue
        if not far_from_reference(
            r, kept_vectors, MIN_LINF_AMONG_KEPT, MIN_L1_AMONG_KEPT
        ):
            continue
        seen.add(key)
        kept_vectors.append(r)
        out.append(
            {
                "predicted_macro": float(val),
                "min_w": float(r.min()),
                "max_w": float(r.max()),
                "weights": {d: float(v) for d, v in zip(DOMAINS, r)},
            }
        )
        if len(out) >= NEAR_OPT_KEEP:
            break
    return out


def mixing_law_objective(fit: dict):
    from fit_mixing_law import _predict

    families = list(fit["families"]) if "families" in fit else sorted(fit["targets"].keys())
    thetas = {fam: theta_from_target(fit["targets"][fam]) for fam in families}

    def objective(r: np.ndarray) -> float:
        return float(
            sum(_predict(thetas[f], r[None, :])[0] for f in families) / len(families)
        )

    return objective


def resample_near_optimal_mixing_law(fit: dict) -> list[dict]:
    return sample_near_optimal(
        mixing_law_objective(fit),
        fit["optimization"]["uncapped"]["predicted_macro"],
        np.random.default_rng(SEED + 99),
        [MIN_DOMAIN_WEIGHT] * 7,
        [1.0] * 7,
        opt_vectors=optima_weight_vectors(fit["optimization"]),
    )


def resample_near_optimal_lightgbm(
    fit: dict, runs: list[dict], families: list[str], y_by_fam: dict
) -> list[dict]:
    X = np.array([[run["weights"][d] for d in DOMAINS] for run in runs], dtype=float)
    lgb_params = {k: v for k, v in fit["lgb_params"].items() if k != "num_boost_round"}
    num_boost_round = int(fit["lgb_params"]["num_boost_round"])
    models = {
        fam: fit_family_model(X, y_by_fam[fam], lgb_params, num_boost_round)
        for fam in families
    }

    def objective(r: np.ndarray) -> float:
        return float(macro_predict_batch(models, families, r[None, :])[0])

    return sample_near_optimal(
        objective,
        fit["optimization"]["uncapped"]["predicted_macro"],
        np.random.default_rng(SEED + 199),
        [MIN_DOMAIN_WEIGHT] * 7,
        [1.0] * 7,
        opt_vectors=optima_weight_vectors(fit["optimization"]),
    )


def optimize_mixing_law(fit: dict, runs: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    objective = mixing_law_objective(fit)

    optima: dict = {}
    for name, caps, floors in OPT_CONSTRAINTS:
        r_star, val = optimize_simplex(objective, 7, caps, floors, n_starts=512, seed=SEED)
        optima[name] = {
            "weights": {d: float(v) for d, v in zip(DOMAINS, r_star)},
            "predicted_macro": float(val),
            "max_w": float(np.max(r_star)),
        }

    pilot_rows = []
    for run in runs:
        r = np.array([run["weights"][d] for d in DOMAINS], dtype=float)
        pilot_rows.append(
            {
                "run_name": run["run_name"],
                "tag": run["tag"],
                "weights": run["weights"],
                "predicted_macro": objective(r),
                "max_w": float(max(run["weights"].values())),
                "measured_curve_6": macro_curve(run["task_loss_families"]),
            }
        )
    pilot_rows.sort(key=lambda x: x["predicted_macro"])

    near = sample_near_optimal(
        objective,
        optima["uncapped"]["predicted_macro"],
        np.random.default_rng(SEED + 99),
        [MIN_DOMAIN_WEIGHT] * 7,
        [1.0] * 7,
        opt_vectors=optima_weight_vectors(optima),
    )
    return optima, pilot_rows, near


def optimize_lightgbm(
    fit: dict, runs: list[dict], families: list[str], y_by_fam: dict
) -> tuple[dict, list[dict], list[dict], dict]:
    X = np.array([[run["weights"][d] for d in DOMAINS] for run in runs], dtype=float)
    lgb_params = {k: v for k, v in fit["lgb_params"].items() if k != "num_boost_round"}
    num_boost_round = int(fit["lgb_params"]["num_boost_round"])
    models = {
        fam: fit_family_model(X, y_by_fam[fam], lgb_params, num_boost_round)
        for fam in families
    }

    def objective(r: np.ndarray) -> float:
        return float(macro_predict_batch(models, families, r[None, :])[0])

    def objective_batch(R: np.ndarray) -> np.ndarray:
        return macro_predict_batch(models, families, R)

    optima: dict = {}
    optimization_meta: dict = {}
    constraint_order = [c[0] for c in OPT_CONSTRAINTS if c[0] != "uncapped"] + ["uncapped"]
    constraint_by_name = {name: (caps, floors) for name, caps, floors in OPT_CONSTRAINTS}

    for name in constraint_order:
        caps, floors = constraint_by_name[name]
        extra_starts: list[np.ndarray] = []
        if name == "uncapped":
            for other in optima.values():
                extra_starts.append(
                    np.array([other["weights"][d] for d in DOMAINS], dtype=float)
                )
        r_star, val, meta = optimize_simplex_global(
            objective,
            objective_batch,
            7,
            caps,
            floors,
            seed=SEED + OPT_SEED_OFFSET[name],
            extra_starts=extra_starts,
        )
        optima[name] = {
            "weights": {d: float(v) for d, v in zip(DOMAINS, r_star)},
            "predicted_macro": float(val),
            "max_w": float(np.max(r_star)),
        }
        optimization_meta[name] = meta

    pilot_rows = []
    for run in runs:
        r = np.array([run["weights"][d] for d in DOMAINS], dtype=float)
        pilot_rows.append(
            {
                "run_name": run["run_name"],
                "tag": run["tag"],
                "weights": run["weights"],
                "predicted_macro": objective(r),
                "max_w": float(max(run["weights"].values())),
                "measured_curve_6": macro_curve(run["task_loss_families"]),
            }
        )
    pilot_rows.sort(key=lambda x: x["predicted_macro"])

    near = sample_near_optimal(
        objective,
        optima["uncapped"]["predicted_macro"],
        np.random.default_rng(SEED + 199),
        [MIN_DOMAIN_WEIGHT] * 7,
        [1.0] * 7,
        opt_vectors=optima_weight_vectors(optima),
    )
    return optima, pilot_rows, near, optimization_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constraints-only",
        action="store_true",
        help="Re-run constrained optima only; leave family fits and near-optimal samples unchanged.",
    )
    parser.add_argument(
        "--near-optimal-only",
        action="store_true",
        help="Only resample near-optimal mixtures; leave optimization and model fits unchanged.",
    )
    parser.add_argument(
        "--lightgbm",
        action="store_true",
        help="Include LightGBM (reloads saved hyperparameters to score mixtures).",
    )
    args = parser.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    chin = json.loads(CHIN.read_text(encoding="utf-8"))
    runs = data["runs"]
    families, y_by_fam = build_targets(data, chin)

    ml = json.loads(ML_FIT.read_text(encoding="utf-8"))
    lgb = json.loads(LGB_FIT.read_text(encoding="utf-8"))

    if args.near_optimal_only:
        ml["near_optimal_balanced_samples"] = resample_near_optimal_mixing_law(ml)
        ML_FIT.write_text(json.dumps(ml, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ML_FIT} (near-optimal only)")

        if args.lightgbm:
            lgb["near_optimal_balanced_samples"] = resample_near_optimal_lightgbm(
                lgb, runs, families, y_by_fam
            )
            LGB_FIT.write_text(json.dumps(lgb, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {LGB_FIT} (near-optimal only)")
        return

    if args.constraints_only:
        ml_opt, _, _ = optimize_mixing_law(ml, runs)
        ml["optimization"] = ml_opt
        ML_FIT.write_text(json.dumps(ml, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ML_FIT} (constraints only)")

        if args.lightgbm:
            for key in ("max40", "max45"):
                lgb["optimization"].pop(key, None)
                if "optimization_method" in lgb:
                    lgb["optimization_method"].pop(key, None)
            lgb_opt, _, _, lgb_opt_meta = optimize_lightgbm(lgb, runs, families, y_by_fam)
            lgb["optimization"] = lgb_opt
            lgb["optimization_method"] = lgb_opt_meta
            LGB_FIT.write_text(json.dumps(lgb, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {LGB_FIT} (constraints only)")
        else:
            print("pass --lightgbm to refresh LightGBM constraint optima")

        print("\n=== Mixing law optima ===")
        for name, o in ml_opt.items():
            print(f"  {name:10s} pred={o['predicted_macro']:.4f} max_w={o['max_w']:.3f}")
        if args.lightgbm:
            print("\n=== LightGBM optima ===")
            for name, o in lgb["optimization"].items():
                print(f"  {name:10s} pred={o['predicted_macro']:.4f} max_w={o['max_w']:.3f}")
        return

    ml_opt, ml_pilot, ml_near = optimize_mixing_law(ml, runs)
    ml["optimization"] = ml_opt
    ml["pilot_ranked"] = ml_pilot
    ml["near_optimal_balanced_samples"] = ml_near
    ML_FIT.write_text(json.dumps(ml, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ML_FIT}")

    for key in ("max40", "max45"):
        lgb["optimization"].pop(key, None)
        if "optimization_method" in lgb:
            lgb["optimization_method"].pop(key, None)

    if args.lightgbm:
        lgb_opt, lgb_pilot, lgb_near, lgb_opt_meta = optimize_lightgbm(
            lgb, runs, families, y_by_fam
        )
        lgb["optimization"] = lgb_opt
        lgb["pilot_ranked"] = lgb_pilot
        lgb["near_optimal_balanced_samples"] = lgb_near
        lgb["optimization_method"] = lgb_opt_meta
        print(f"wrote {LGB_FIT} (LightGBM rescored)")
    else:
        LGB_FIT.write_text(json.dumps(lgb, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {LGB_FIT} (removed max40/max45 only; pass --lightgbm to add min1pct)")

    print("\n=== Mixing law optima ===")
    for name, o in ml_opt.items():
        print(f"  {name:10s} pred={o['predicted_macro']:.4f} max_w={o['max_w']:.3f}")

    if args.lightgbm:
        print("\n=== LightGBM optima ===")
        for name, o in lgb["optimization"].items():
            print(f"  {name:10s} pred={o['predicted_macro']:.4f} max_w={o['max_w']:.3f}")


if __name__ == "__main__":
    main()
