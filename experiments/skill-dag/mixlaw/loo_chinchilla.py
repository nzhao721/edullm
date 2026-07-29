#!/usr/bin/env python3
"""Leave-one-out cross-validation for the Chinchilla mixing-law fit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fit_chinchilla import N_STARTS, SEED, build_targets, fit_family_candidates
from fit_mixing_law import DOMAINS, _predict

DATA = Path("mixlaw_data.json")
CHIN = Path("mixlaw_chinchilla_extrapolated.json")
FIT = Path("mixlaw_fit_chinchilla.json")
OUT = Path("mixlaw_fit_chinchilla_loo.json")

# Slightly fewer starts than the full fit to keep wall time reasonable.
LOO_STARTS = 64


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    chin = json.loads(CHIN.read_text(encoding="utf-8"))
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    families, y_by_fam = build_targets(data, chin)
    runs = data["runs"]
    R = np.array([[run["weights"][d] for d in DOMAINS] for run in runs], dtype=float)
    n = len(runs)

    import fit_chinchilla as fc

    fc.N_STARTS = LOO_STARTS

    report: dict = {
        "n_runs": n,
        "loo_starts": LOO_STARTS,
        "full_fit_starts": N_STARTS,
        "extrapolate_to_step": chin["chinchilla_steps"],
        "families": {},
        "run_names": [run["run_name"] for run in runs],
    }

    t0 = time.time()
    print(f"LOO CV: {n} folds x {len(families)} families, {LOO_STARTS} starts each")
    print(f"(full fit used {N_STARTS} starts)\n")

    for fi, fam in enumerate(families):
        y = y_by_fam[fam]
        preds = np.empty(n, dtype=float)
        fold_rmse = []
        fam_t0 = time.time()
        for i in range(n):
            keep = np.arange(n) != i
            cands = fit_family_candidates(R[keep], y[keep], seed=SEED + 1000 * fi + i)
            theta = cands[0]["theta"]
            preds[i] = float(_predict(theta, R[i : i + 1])[0])
            fold_rmse.append(cands[0]["rmse"])
            if (i + 1) % 6 == 0 or i == 0:
                elapsed = time.time() - fam_t0
                eta = elapsed / (i + 1) * (n - i - 1)
                print(
                    f"  {fam:22s} fold {i+1:2d}/{n}  "
                    f"pred={preds[i]:.4f} true={y[i]:.4f}  "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

        err = preds - y
        in_sample = float(fit["targets"][fam]["fit_rmse"])
        entry = {
            "loo_rmse": float(np.sqrt(np.mean(err**2))),
            "loo_mae": float(np.mean(np.abs(err))),
            "loo_max_abs": float(np.max(np.abs(err))),
            "target_std": float(y.std()),
            "loo_rmse_over_std": float(np.sqrt(np.mean(err**2)) / max(float(y.std()), 1e-12)),
            "in_sample_rmse": in_sample,
            "predictions": preds.tolist(),
            "errors": err.tolist(),
            "mean_train_rmse": float(np.mean(fold_rmse)),
        }
        report["families"][fam] = entry
        ratio = entry["loo_rmse_over_std"]
        print(
            f"{fam:22s}  LOO RMSE={entry['loo_rmse']:.4f}  "
            f"({ratio:.0%} of std)  MAE={entry['loo_mae']:.4f}  "
            f"max|e|={entry['loo_max_abs']:.4f}  "
            f"in-sample={in_sample:.4f}  "
            f"[{time.time()-fam_t0:.0f}s]",
            flush=True,
        )

    pred_macro = np.mean(
        np.stack([np.array(report["families"][f]["predictions"]) for f in families]),
        axis=0,
    )
    true_macro = np.mean(np.stack([y_by_fam[f] for f in families]), axis=0)
    macro_err = pred_macro - true_macro
    report["macro"] = {
        "loo_rmse": float(np.sqrt(np.mean(macro_err**2))),
        "loo_mae": float(np.mean(np.abs(macro_err))),
        "loo_max_abs": float(np.max(np.abs(macro_err))),
        "target_std": float(true_macro.std()),
        "predictions": pred_macro.tolist(),
        "true": true_macro.tolist(),
        "errors": macro_err.tolist(),
    }

    worst = max(report["families"].items(), key=lambda kv: kv[1]["loo_rmse"])
    best = min(report["families"].items(), key=lambda kv: kv[1]["loo_rmse"])
    report["summary"] = {
        "mean_loo_rmse": float(np.mean([v["loo_rmse"] for v in report["families"].values()])),
        "mean_loo_rmse_over_std": float(
            np.mean([v["loo_rmse_over_std"] for v in report["families"].values()])
        ),
        "worst_family": worst[0],
        "worst_loo_rmse": worst[1]["loo_rmse"],
        "best_family": best[0],
        "best_loo_rmse": best[1]["loo_rmse"],
        "macro_loo_rmse": report["macro"]["loo_rmse"],
        "elapsed_sec": time.time() - t0,
    }

    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(
        f"mean LOO RMSE={report['summary']['mean_loo_rmse']:.4f} "
        f"({report['summary']['mean_loo_rmse_over_std']:.0%} of std)  "
        f"macro LOO RMSE={report['macro']['loo_rmse']:.4f}  "
        f"wall={report['summary']['elapsed_sec']/60:.1f} min"
    )


if __name__ == "__main__":
    main()
