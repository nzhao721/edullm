#!/usr/bin/env python3
"""Sample random simplex mixtures and check mixing-law plausibility off the pilot hull."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fit_mixing_law import DOMAINS, _predict
from mixlaw_common import macro_curve

ROOT = Path(__file__).parent
N_SAMPLES = 1000
SEED = 42


def load_thetas(path: Path) -> dict[str, np.ndarray]:
    fit = json.loads(path.read_text(encoding="utf-8"))
    return {
        fam: np.array(
            [math.log(fit["targets"][fam]["c"]), math.log(fit["targets"][fam]["k"])]
            + [fit["targets"][fam]["t"][d] for d in DOMAINS]
        )
        for fam in sorted(fit["targets"])
    }


def summarize(thetas: dict[str, np.ndarray], R: np.ndarray, obs_macro: list[float]) -> dict:
    fams = sorted(thetas)
    pred = np.stack([_predict(thetas[f], R) for f in fams])
    macro = pred.mean(0)
    return {
        "macro": {
            "min": float(macro.min()),
            "p5": float(np.percentile(macro, 5)),
            "p50": float(np.percentile(macro, 50)),
            "p95": float(np.percentile(macro, 95)),
            "p99": float(np.percentile(macro, 99)),
            "max": float(macro.max()),
            "mean": float(macro.mean()),
            "std": float(macro.std()),
            "pct_in_pilot_range": float(
                np.mean((macro >= min(obs_macro)) & (macro <= max(obs_macro))) * 100
            ),
            "n_gt_3": int(np.sum(macro > 3)),
            "n_gt_5": int(np.sum(macro > 5)),
            "n_gt_10": int(np.sum(macro > 10)),
        },
        "any_family_gt_10": int(np.sum(pred.max(0) > 10)),
        "mmlu_other_p99": float(np.percentile(pred[fams.index("mmlu_other")], 99)),
        "mmlu_other_max": float(pred[fams.index("mmlu_other")].max()),
    }


def main() -> None:
    data = json.loads((ROOT / "mixlaw_data.json").read_text(encoding="utf-8"))
    obs_macro = [macro_curve(r["task_loss_families"]) for r in data["runs"]]
    rng = np.random.default_rng(SEED)
    R = rng.dirichlet(np.ones(7), N_SAMPLES)

    stats = summarize(
        load_thetas(ROOT / "mixlaw_fit_chinchilla.json"), R, obs_macro
    )
    out = {
        "n_samples": N_SAMPLES,
        "sampling": "dirichlet_alpha=1",
        "seed": SEED,
        "fit_file": "mixlaw_fit_chinchilla.json",
        "observed_pilot_macro_range": [min(obs_macro), max(obs_macro)],
        **stats,
    }
    out_path = ROOT / "mixlaw_random_simplex_plausibility.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
