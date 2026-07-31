#!/usr/bin/env python3
"""Emit validation_mixtures_10b.json from fit artifacts + pilot mixtures.

All arms train by streaming from a working pool staged from published
``edullm-data`` (``pretrain/olmo-127b``) at the recipe domain weights
(see ``data_source`` in the emitted JSON).
"""
from __future__ import annotations

import json
from pathlib import Path

from mixlaw_common import DOMAINS, OLMO_MIX_1124_WEIGHTS
from stage_validation_pool_from_edullm_data import DEFAULT_DATASET_ID

ROOT = Path(__file__).parent
FIT = json.loads((ROOT / "mixlaw_fit_chinchilla.json").read_text(encoding="utf-8"))
LGB = json.loads((ROOT / "mixlaw_fit_lightgbm_chinchilla.json").read_text(encoding="utf-8"))
MIX = json.loads((ROOT / "mixtures.json").read_text(encoding="utf-8"))

ML_NEAR_OPT_INDEX = 3  # near-opt 4
LGB_NEAR_OPT_INDEX = 7  # near-opt 8


def wlist(d: dict[str, float]) -> list[float]:
    return [float(d[x]) for x in DOMAINS]


def main() -> None:
    rows: list[dict] = [
        {
            "id": 0,
            "run_name": "olmo-mix-1124",
            "tag": "natural",
            "source": "reference",
            "label": "natural",
            "weights": wlist(OLMO_MIX_1124_WEIGHTS),
        }
    ]
    for pid, tag in ((1, "base"), (7, "C1-dclm60"), (18, "C1")):
        m = next(x for x in MIX["mixtures"] if x["id"] == pid)
        w = dict(zip(MIX["domain_order"], m["weights"]))
        rows.append(
            {
                "id": pid,
                "run_name": f"mix{pid:02d}",
                "tag": tag,
                "source": "pilot",
                "label": tag,
                "weights": wlist(w),
            }
        )

    surrogates = [
        (25, "ML-pilot_caps", "mixing-law", FIT["optimization"]["pilot_caps"]["weights"]),
        (
            26,
            "ML-near-opt-4",
            "mixing-law",
            FIT["near_optimal_balanced_samples"][ML_NEAR_OPT_INDEX]["weights"],
        ),
        (27, "LGB-min1pct", "lightgbm", LGB["optimization"]["min1pct"]["weights"]),
        (
            28,
            "LGB-near-opt-8",
            "lightgbm",
            LGB["near_optimal_balanced_samples"][LGB_NEAR_OPT_INDEX]["weights"],
        ),
    ]
    for mid, name, source, weights in surrogates:
        rows.append(
            {
                "id": mid,
                "run_name": name,
                "tag": name,
                "source": source,
                "label": name,
                "weights": wlist(weights),
            }
        )

    out = {
        "schema": 2,
        "description": (
            "370M validation mixtures for skill-dag mixing-law / LightGBM scale-up "
            "(10B tokens each). Train by domain-stratified streaming from published "
            "edullm-data pretrain/olmo-127b at the listed weights — no per-mix "
            "pre-baked corpora."
        ),
        "budget_tokens": 10_000_000_000,
        "domain_order": list(DOMAINS),
        "seed": 6198,
        "data_source": {
            "dataset_id": DEFAULT_DATASET_ID,
            "bucket": "edullm-data",
            "label_key": "source",
            "mode": "domain_stratified_stream",
        },
        "mixtures": rows,
    }
    path = ROOT / "validation_mixtures_10b.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(rows)} mixtures)")


if __name__ == "__main__":
    main()
