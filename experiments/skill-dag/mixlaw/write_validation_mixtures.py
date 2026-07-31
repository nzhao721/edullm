#!/usr/bin/env python3
"""Emit validation_mixtures_10b.json from fit artifacts + pilot mixtures.

mix01 reuses s3://edullm-datasets/regmix/regmix-10b/ by reference (read-only).
"""
from __future__ import annotations

import json
from pathlib import Path

from mixlaw_common import DOMAINS, OLMO_MIX_1124_WEIGHTS

ROOT = Path(__file__).parent
FIT = json.loads((ROOT / "mixlaw_fit_chinchilla.json").read_text(encoding="utf-8"))
LGB = json.loads((ROOT / "mixlaw_fit_lightgbm_chinchilla.json").read_text(encoding="utf-8"))
MIX = json.loads((ROOT / "mixtures.json").read_text(encoding="utf-8"))

# 370M validation surrogate picks (1-based near-opt index into fit JSON samples).
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
            "reuse_s3": None,
        }
    ]
    for pid, tag in ((1, "base"), (7, "C1-dclm60"), (18, "C1")):
        m = next(x for x in MIX["mixtures"] if x["id"] == pid)
        w = dict(zip(MIX["domain_order"], m["weights"]))
        row = {
            "id": pid,
            "run_name": f"mix{pid:02d}",
            "tag": tag,
            "source": "pilot",
            "label": tag,
            "weights": wlist(w),
            "reuse_s3": None,
        }
        if pid == 1:
            row["reuse_s3"] = "s3://edullm-datasets/regmix/regmix-10b/"
        rows.append(row)

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
                "reuse_s3": None,
            }
        )

    out = {
        "schema": 1,
        "description": (
            "370M validation mixtures for skill-dag mixing-law / LightGBM scale-up "
            "(10B tokens each). mix01 is a read-only reference to regmix-10b."
        ),
        "budget_tokens": 10_000_000_000,
        "domain_order": list(DOMAINS),
        "seed": 6198,
        "mixtures": rows,
        "regmix_policy": "read_only_never_modify",
    }
    path = ROOT / "validation_mixtures_10b.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(rows)} mixtures)")


if __name__ == "__main__":
    main()
