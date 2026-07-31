#!/usr/bin/env python3
"""LightGBM surrogate for Chinchilla-extrapolated task losses vs mixture weights."""
from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fit_chinchilla import SEED, build_targets
from fit_mixing_law import sample_feasible_mixtures
from mixlaw_common import MIXTURE_OPT_CONSTRAINTS, NEAR_OPT_DOMAIN_CAPS, NEAR_OPT_DOMAIN_FLOORS, macro_curve

DATA = Path("mixlaw_data.json")
CHIN = Path("mixlaw_chinchilla_extrapolated.json")
OUT = Path("mixlaw_fit_lightgbm_chinchilla.json")
DOMAINS = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)

OPT_CONSTRAINTS = MIXTURE_OPT_CONSTRAINTS

BASE_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "seed": SEED,
}

# Hand-picked conservative defaults (pre-search baseline).
DEFAULT_LGB_PARAMS = {
    **BASE_LGB_PARAMS,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 3,
    "learning_rate": 0.05,
    "lambda_l2": 1.0,
}

# Small LOO-driven grid for n=24, p=7.
HYPERPARAM_GRID = {
    "num_leaves": [3, 5, 7],
    "max_depth": [2, 3],
    "min_data_in_leaf": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "lambda_l2": [0.1, 1.0],
    "num_boost_round": [100, 200],
}

OPT_SEED_OFFSET = {"uncapped": 0, "pilot_caps": 1, "min1pct": 2}

GLOBAL_OPT = {
    "n_random": 50_000,
    "n_polish": 32,
    "n_slsqp_starts": 64,
    "n_refine_perturbations": 64,
}


def is_full_simplex(floors: np.ndarray, caps: np.ndarray) -> bool:
    return bool(np.all(floors <= 1e-12) and np.all(caps >= 1.0 - 1e-12))


def project_to_box_simplex(
    r: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
    max_iter: int = 40,
) -> np.ndarray | None:
    """Clip-renormalize iterate until inside the box (or give up)."""
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


def macro_predict_batch(
    models: dict[str, lgb.Booster],
    families: list[str],
    R: np.ndarray,
) -> np.ndarray:
    preds = np.stack([predict_family(models[f], R) for f in families])
    return preds.mean(0)


def polish_slsqp(
    objective: Callable[[np.ndarray], float],
    r0: np.ndarray,
    floors: np.ndarray,
    caps: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    from scipy.optimize import minimize

    bounds = [(float(lo), float(hi)) for lo, hi in zip(floors, caps)]
    constraint = {"type": "eq", "fun": lambda r: float(np.sum(r) - 1.0)}
    try:
        res = minimize(
            objective,
            r0,
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint],
            options={"maxiter": 300, "ftol": 1e-10},
        )
    except Exception:
        return None
    if not res.success:
        return None
    r = project_to_box_simplex(res.x, floors, caps)
    if r is None:
        return None
    return r, float(objective(r))


def optimize_simplex_global(
    objective: Callable[[np.ndarray], float],
    objective_batch: Callable[[np.ndarray], np.ndarray],
    n_domains: int,
    caps: Sequence[float],
    floors: Sequence[float],
    *,
    n_random: int = GLOBAL_OPT["n_random"],
    n_polish: int = GLOBAL_OPT["n_polish"],
    n_slsqp_starts: int = GLOBAL_OPT["n_slsqp_starts"],
    n_refine_perturbations: int = GLOBAL_OPT["n_refine_perturbations"],
    extra_starts: Sequence[np.ndarray] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float, dict]:
    """Dense random search + local polish for non-convex LightGBM objectives."""
    rng = np.random.default_rng(seed)
    floor_v = np.asarray(floors, dtype=float)
    cap_v = np.asarray(caps, dtype=float)

    R_rand = sample_feasible_mixtures(rng, floor_v, cap_v, n_random)
    vals_rand = objective_batch(R_rand)
    best_idx = int(np.argmin(vals_rand))
    best_r, best_val = R_rand[best_idx].copy(), float(vals_rand[best_idx])
    random_best = best_val
    print(
        f"  random search: best={random_best:.4f} from {n_random} samples",
        flush=True,
    )

    polish_idx = np.argpartition(vals_rand, min(n_polish, len(vals_rand) - 1))[:n_polish]
    polished = 0
    for idx in polish_idx:
        result = polish_slsqp(objective, R_rand[idx], floor_v, cap_v)
        if result is None:
            continue
        polished += 1
        r, val = result
        if val < best_val:
            best_r, best_val = r, val

    slsqp_improved = 0
    start_points: list[np.ndarray] = []
    if extra_starts:
        start_points.extend(extra_starts)
    for _ in range(n_slsqp_starts):
        alpha = rng.uniform(0.15, 4.0, size=n_domains)
        r0 = project_to_box_simplex(rng.dirichlet(alpha), floor_v, cap_v)
        if r0 is not None:
            start_points.append(r0)
    for r0 in start_points:
        result = polish_slsqp(objective, r0, floor_v, cap_v)
        if result is None:
            continue
        slsqp_improved += 1
        r, val = result
        if val < best_val:
            best_r, best_val = r, val
    print(
        f"  after polish/slsqp: best={best_val:.4f} ({polished} polished, "
        f"{slsqp_improved} slsqp)",
        flush=True,
    )

    refine_improved = 0
    for _ in range(n_refine_perturbations):
        noise = rng.normal(0.0, 0.03, size=n_domains)
        r0 = project_to_box_simplex(best_r * np.exp(noise), floor_v, cap_v)
        if r0 is None:
            continue
        result = polish_slsqp(objective, r0, floor_v, cap_v)
        if result is None:
            continue
        refine_improved += 1
        r, val = result
        if val < best_val:
            best_r, best_val = r, val

    meta = {
        "method": "random_search + slsqp_polish",
        "n_random": n_random,
        "n_polish": n_polish,
        "n_slsqp_starts": n_slsqp_starts,
        "n_refine_perturbations": n_refine_perturbations,
        "random_best_predicted_macro": random_best,
        "final_predicted_macro": best_val,
        "improvement_over_random_best": float(random_best - best_val),
        "n_polish_converged": polished,
        "n_slsqp_converged": slsqp_improved,
        "n_refine_converged": refine_improved,
    }
    return best_r, best_val, meta


def fit_family_model(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    num_boost_round: int,
) -> lgb.Booster:
    train = lgb.Dataset(X, label=y, feature_name=list(DOMAINS))
    train_params = {k: v for k, v in params.items() if k != "num_boost_round"}
    return lgb.train(
        {**train_params, "num_boost_round": num_boost_round},
        train,
        valid_sets=[train],
        callbacks=[lgb.log_evaluation(period=0)],
    )


def predict_family(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    num_boost_round: int,
) -> dict:
    n = len(y)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        keep = np.arange(n) != i
        model = fit_family_model(X[keep], y[keep], params, num_boost_round)
        preds[i] = float(predict_family(model, X[i : i + 1])[0])
    err = preds - y
    return {
        "loo_rmse": float(np.sqrt(np.mean(err**2))),
        "loo_mae": float(np.mean(np.abs(err))),
        "loo_max_abs": float(np.max(np.abs(err))),
        "target_std": float(y.std()),
        "loo_rmse_over_std": float(np.sqrt(np.mean(err**2)) / max(float(y.std()), 1e-12)),
        "predictions": preds.tolist(),
        "errors": err.tolist(),
    }


def macro_loo_score(
    X: np.ndarray,
    y_by_fam: dict[str, np.ndarray],
    families: list[str],
    params: dict,
    num_boost_round: int,
) -> tuple[float, float]:
    """Return (mean per-family LOO RMSE, macro LOO RMSE)."""
    fam_preds: list[np.ndarray] = []
    fam_loo: list[float] = []
    for fam in families:
        loo = loo_cv(X, y_by_fam[fam], params, num_boost_round)
        fam_preds.append(np.asarray(loo["predictions"], dtype=float))
        fam_loo.append(loo["loo_rmse"])
    pred_macro = np.mean(np.stack(fam_preds), axis=0)
    true_macro = np.mean(np.stack([y_by_fam[f] for f in families]), axis=0)
    macro_err = pred_macro - true_macro
    macro_loo = float(np.sqrt(np.mean(macro_err**2)))
    return float(np.mean(fam_loo)), macro_loo


def search_hyperparams(
    X: np.ndarray,
    y_by_fam: dict[str, np.ndarray],
    families: list[str],
) -> dict:
    keys = list(HYPERPARAM_GRID.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(HYPERPARAM_GRID[k] for k in keys))]
    print(f"=== LOO hyperparameter search ({len(combos)} configs) ===\n")

    baseline = {**DEFAULT_LGB_PARAMS}
    baseline_nbr = 200
    baseline_mean, baseline_macro = macro_loo_score(
        X, y_by_fam, families, baseline, baseline_nbr
    )
    print(
        f"baseline defaults: mean LOO={baseline_mean:.4f}, macro LOO={baseline_macro:.4f}"
    )

    rows: list[dict] = []
    for idx, combo in enumerate(combos, 1):
        params = {**BASE_LGB_PARAMS, **{k: v for k, v in combo.items() if k != "num_boost_round"}}
        nbr = int(combo["num_boost_round"])
        mean_loo, macro_loo = macro_loo_score(X, y_by_fam, families, params, nbr)
        rows.append(
            {
                "rank_metric": macro_loo,
                "mean_loo_rmse": mean_loo,
                "macro_loo_rmse": macro_loo,
                **combo,
            }
        )
        if idx % 24 == 0 or idx == len(combos):
            print(f"  evaluated {idx}/{len(combos)} configs", flush=True)

    rows.sort(key=lambda r: (r["macro_loo_rmse"], r["mean_loo_rmse"]))
    best = rows[0]
    selected_params = {
        **BASE_LGB_PARAMS,
        "num_leaves": best["num_leaves"],
        "max_depth": best["max_depth"],
        "min_data_in_leaf": best["min_data_in_leaf"],
        "learning_rate": best["learning_rate"],
        "lambda_l2": best["lambda_l2"],
    }
    selected_nbr = int(best["num_boost_round"])

    print(
        f"\nselected: macro LOO={best['macro_loo_rmse']:.4f} "
        f"(baseline {baseline_macro:.4f}), params={selected_params}, "
        f"num_boost_round={selected_nbr}"
    )

    return {
        "objective": "minimize macro LOO RMSE across 6 families",
        "grid": HYPERPARAM_GRID,
        "n_configs": len(combos),
        "baseline_defaults": {
            "params": DEFAULT_LGB_PARAMS,
            "num_boost_round": baseline_nbr,
            "mean_loo_rmse": baseline_mean,
            "macro_loo_rmse": baseline_macro,
        },
        "selected": {
            "params": selected_params,
            "num_boost_round": selected_nbr,
            "mean_loo_rmse": best["mean_loo_rmse"],
            "macro_loo_rmse": best["macro_loo_rmse"],
        },
        "top_configs": rows[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hyperparam-search",
        action="store_true",
        help="Reuse hyperparameters from an existing output JSON.",
    )
    args = parser.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    chin = json.loads(CHIN.read_text(encoding="utf-8"))
    families, y_by_fam = build_targets(data, chin)
    runs = data["runs"]
    X = np.array([[run["weights"][d] for d in DOMAINS] for run in runs], dtype=float)

    if args.skip_hyperparam_search and OUT.exists():
        cached = json.loads(OUT.read_text(encoding="utf-8"))
        hyperparam_search = cached["hyperparam_search"]
        lgb_params = {
            k: v for k, v in cached["lgb_params"].items() if k != "num_boost_round"
        }
        num_boost_round = int(cached["lgb_params"]["num_boost_round"])
        print("=== Reusing cached LOO-selected hyperparameters ===\n")
    else:
        hyperparam_search = search_hyperparams(X, y_by_fam, families)
        lgb_params = hyperparam_search["selected"]["params"]
        num_boost_round = hyperparam_search["selected"]["num_boost_round"]

    models: dict[str, lgb.Booster] = {}
    family_report: dict[str, dict] = {}
    print("\n=== LightGBM on Chinchilla targets (6 families, 24 mixes) ===\n")
    for fam in families:
        y = y_by_fam[fam]
        model = fit_family_model(X, y, lgb_params, num_boost_round)
        models[fam] = model
        pred = predict_family(model, X)
        in_rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        loo = loo_cv(X, y, lgb_params, num_boost_round)
        importance = {
            d: float(v)
            for d, v in zip(DOMAINS, model.feature_importance(importance_type="gain"))
        }
        family_report[fam] = {
            "in_sample_rmse": in_rmse,
            "in_sample_mae": float(np.mean(np.abs(pred - y))),
            "observed": {
                "min": float(y.min()),
                "max": float(y.max()),
                "std": float(y.std()),
            },
            "feature_importance_gain": importance,
            **loo,
        }
        print(
            f"{fam:22s}  in-sample RMSE={in_rmse:.4f}  "
            f"LOO RMSE={loo['loo_rmse']:.4f} ({loo['loo_rmse_over_std']:.0%} of std)"
        )

    def objective(r: np.ndarray) -> float:
        return float(macro_predict_batch(models, families, r[None, :])[0])

    def objective_batch(R: np.ndarray) -> np.ndarray:
        return macro_predict_batch(models, families, R)

    optima = {}
    optimization_meta: dict[str, dict] = {}
    constraint_order = [c[0] for c in OPT_CONSTRAINTS if c[0] != "uncapped"] + ["uncapped"]
    constraint_by_name = {name: (caps, floors) for name, caps, floors in OPT_CONSTRAINTS}

    for name in constraint_order:
        caps, floors = constraint_by_name[name]
        print(f"\n=== Optimizing ({name}) ===")
        extra_starts: list[np.ndarray] = []
        if name == "uncapped":
            for other_name, other in optima.items():
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
        print(
            f"  pred={val:.4f}  random_best={meta['random_best_predicted_macro']:.4f}  "
            f"delta={meta['improvement_over_random_best']:.4f}  max_w={optima[name]['max_w']:.3f}"
        )

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

    pred_macro = np.mean(
        np.stack([np.array(family_report[f]["predictions"]) for f in families]), axis=0
    )
    true_macro = np.mean(np.stack([y_by_fam[f] for f in families]), axis=0)
    macro_err = pred_macro - true_macro

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
    near_optimal_balanced_samples = []
    seen: set[tuple[float, ...]] = set()
    for val, r in samples:
        key = tuple(round(float(x), 2) for x in r)
        if key in seen:
            continue
        seen.add(key)
        near_optimal_balanced_samples.append(
            {
                "predicted_macro": float(val),
                "min_w": float(r.min()),
                "max_w": float(r.max()),
                "weights": {d: float(v) for d, v in zip(DOMAINS, r)},
            }
        )
        if len(near_optimal_balanced_samples) >= 8:
            break

    obs_macro = [macro_curve(r["task_loss_families"]) for r in runs]
    rng_simplex = np.random.default_rng(42)
    R_rand = rng_simplex.dirichlet(np.ones(7), 1000)
    fams = sorted(families)
    pred_all = np.stack([predict_family(models[f], R_rand) for f in fams])
    macro_rand = pred_all.mean(0)
    random_simplex_plausibility = {
        "n_samples": 1000,
        "sampling": "dirichlet_alpha=1",
        "seed": 42,
        "observed_pilot_macro_range": [min(obs_macro), max(obs_macro)],
        "macro": {
            "min": float(macro_rand.min()),
            "p5": float(np.percentile(macro_rand, 5)),
            "p50": float(np.percentile(macro_rand, 50)),
            "p95": float(np.percentile(macro_rand, 95)),
            "p99": float(np.percentile(macro_rand, 99)),
            "max": float(macro_rand.max()),
            "mean": float(macro_rand.mean()),
            "std": float(macro_rand.std()),
            "pct_in_pilot_range": float(
                np.mean((macro_rand >= min(obs_macro)) & (macro_rand <= max(obs_macro))) * 100
            ),
            "n_gt_3": int(np.sum(macro_rand > 3)),
            "n_gt_5": int(np.sum(macro_rand > 5)),
            "n_gt_10": int(np.sum(macro_rand > 10)),
        },
        "any_family_gt_10": int(np.sum(pred_all.max(0) > 10)),
        "mmlu_other_p99": float(np.percentile(pred_all[fams.index("mmlu_other")], 99)),
        "mmlu_other_max": float(pred_all[fams.index("mmlu_other")].max()),
    }

    report = {
        "model": "lightgbm",
        "n_runs": len(runs),
        "domain_order": list(DOMAINS),
        "extrapolate_to_step": chin["chinchilla_steps"],
        "families": families,
        "hyperparam_search": hyperparam_search,
        "optimization_method": optimization_meta,
        "lgb_params": {**lgb_params, "num_boost_round": num_boost_round},
        "family_models": family_report,
        "optimization": optima,
        "pilot_ranked": pilot_rows,
        "near_optimal_balanced_samples": near_optimal_balanced_samples,
        "random_simplex_plausibility": random_simplex_plausibility,
        "macro_loo": {
            "loo_rmse": float(np.sqrt(np.mean(macro_err**2))),
            "loo_mae": float(np.mean(np.abs(macro_err))),
            "predictions": pred_macro.tolist(),
            "true": true_macro.tolist(),
        },
        "summary": {
            "mean_in_sample_rmse": float(
                np.mean([family_report[f]["in_sample_rmse"] for f in families])
            ),
            "mean_loo_rmse": float(
                np.mean([family_report[f]["loo_rmse"] for f in families])
            ),
            "mean_loo_rmse_over_std": float(
                np.mean([family_report[f]["loo_rmse_over_std"] for f in families])
            ),
            "macro_loo_rmse": float(np.sqrt(np.mean(macro_err**2))),
        },
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

    print("\n=== Optima (LightGBM) ===")
    for name, o in optima.items():
        w = o["weights"]
        print(
            f"{name:12s}  pred={o['predicted_macro']:.4f}  max_w={o['max_w']:.3f}  "
            + " ".join(f"{d[:4]}={w[d]:.3f}" for d in DOMAINS if w[d] >= 0.01)
        )

    print("\n=== Pilot mixes ranked by LightGBM ===")
    for row in pilot_rows:
        print(
            f"  {row['run_name']} ({row['tag']:10s}) pred={row['predicted_macro']:.4f} "
            f"max_w={row['max_w']:.3f} measured6={row['measured_curve_6']:.4f}"
        )


if __name__ == "__main__":
    main()
