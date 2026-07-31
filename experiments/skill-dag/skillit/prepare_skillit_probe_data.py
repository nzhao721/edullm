#!/usr/bin/env python3
"""Prepare per-probe recipe sidecars for Skill-It 60M one-hot probes.

Writes ``mix_weights.json`` under ``<work>/<probe_id>/``. Training streams from
a working pool staged from published ``edullm-data`` at one-hot weights (no
materialized slices). Does **not** read ``s3://edullm-datasets/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MIXLAW = Path(__file__).resolve().parent.parent / "mixlaw"
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

from mixlaw_common import DEFAULT_TOKENS_PER_PARAM  # noqa: E402
from recipe_data import prepare_from_mixtures  # noqa: E402

DEFAULT_RECIPE = Path(__file__).resolve().parent / "probes.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--tokens-per-param", type=float, default=DEFAULT_TOKENS_PER_PARAM)
    ap.add_argument("--only", nargs="*", default=None, help="Optional probe run_name values")
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    recipe["_tokens_per_param"] = float(args.tokens_per_param)
    wanted = set(args.only) if args.only else None
    arms = prepare_from_mixtures(recipe, args.recipe, args.work, only=wanted)
    if not arms:
        raise SystemExit("no probes prepared")

    for arm in arms:
        print(f"prepared {arm['run_name']}")

    index = {
        "recipe": str(args.recipe.resolve()),
        "tokens_per_param": float(args.tokens_per_param),
        "budget_tokens": arms[0]["budget_tokens"],
        "domain_order": recipe.get("domain_order"),
        "data_source": recipe.get("data_source")
        or {
            "dataset_id": "pretrain/olmo-127b",
            "bucket": "edullm-data",
            "mode": "domain_stratified_stream",
        },
        "arms": arms,
    }
    out = args.work / "probe_arms.json"
    out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(arms)} probes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
