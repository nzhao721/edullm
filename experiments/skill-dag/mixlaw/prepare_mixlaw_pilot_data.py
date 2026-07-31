#!/usr/bin/env python3
"""Prepare per-mix recipe sidecars for the 24 DataDecide-60M mixing-law probes.

Writes ``mix_weights.json`` per mix under ``<work>/mixNN/``. Training streams
from a shared working pool staged from published ``edullm-data``
(``pretrain/olmo-127b``) at those weights (no per-mix materialized slices).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mixlaw_common import DEFAULT_TOKENS_PER_PARAM, EDULLM_DATA_DATASET_ID, MIXTURES_JSON
from recipe_data import prepare_from_mixtures

DEFAULT_RECIPE = MIXTURES_JSON


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--tokens-per-param", type=float, default=DEFAULT_TOKENS_PER_PARAM)
    ap.add_argument("--dataset-id", default=EDULLM_DATA_DATASET_ID)
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Optional pin; training/staging default to resolve_latest / pool provenance",
    )
    ap.add_argument("--only", nargs="*", default=None, help="Optional mixNN run names")
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    recipe["_tokens_per_param"] = float(args.tokens_per_param)
    recipe["data_source"] = {
        "dataset_id": args.dataset_id,
        "mode": "domain_stratified_stream",
        "data_bucket": "edullm-data",
    }
    if args.dataset_version:
        recipe["data_source"]["dataset_version"] = args.dataset_version
    wanted = set(args.only) if args.only else None
    arms = prepare_from_mixtures(recipe, args.recipe, args.work, only=wanted)
    if not arms:
        raise SystemExit("no mixtures prepared")

    for arm in arms:
        print(f"prepared {arm['run_name']}")

    index = {
        "recipe": str(args.recipe.resolve()),
        "tokens_per_param": float(args.tokens_per_param),
        "budget_tokens": arms[0]["budget_tokens"],
        "domain_order": recipe.get("domain_order"),
        "data_source": recipe["data_source"],
        "arms": arms,
    }
    out = args.work / "pilot_arms.json"
    out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
