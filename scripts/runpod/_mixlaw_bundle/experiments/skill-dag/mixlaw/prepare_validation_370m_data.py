#!/usr/bin/env python3
"""Prepare per-arm recipe sidecars for mixlaw 370M validation training.

Reads domain weights from ``validation_mixtures_10b.json``. Training streams
from a working pool staged from published+validated
``s3://edullm-data/pretrain/olmo-127b`` (``edullm_data.read.dataset_paths`` /
``resolve_latest``) at those weights. Never reads ``s3://edullm-datasets/``.

Optionally ``--stage-pool`` fetches and builds that pool on this machine so an
ephemeral empty-scratch node does not need pre-existing FarmShare/laptop
corpora or persistent pools.

Writes per mix::

    <work>/<run_name>/mix_weights.json
    <work>/validation_arms.json   # index of all prepared arms
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mixlaw_common import DOMAINS
from stage_validation_pool_from_edullm_data import (
    DEFAULT_DATASET_ID,
    pool_is_ready,
    resolve_dataset,
    stage_pool,
)

DEFAULT_RECIPE = Path(__file__).resolve().parent / "validation_mixtures_10b.json"


def prepare_mix(
    mix: dict,
    recipe: dict,
    work: Path,
    *,
    dataset_id: str,
    dataset_version: str,
) -> dict:
    name = mix["run_name"]
    weights = {d: float(w) for d, w in zip(DOMAINS, mix["weights"])}

    out_dir = work / name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "mix_weights.json"
    recipe_seed = int(recipe.get("seed", 6198))
    ds = recipe.get("data_source") or {}
    payload = {
        "run_name": name,
        "id": mix["id"],
        "tag": mix.get("tag"),
        "source": mix.get("source"),
        "domain_order": list(DOMAINS),
        "weights": weights,
        "recipe": str(DEFAULT_RECIPE.name),
        "recipe_seed": recipe_seed,
        "stream_seed": recipe_seed,
        "budget_tokens": recipe.get("budget_tokens"),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "edullm_data_uri": f"s3://edullm-data/{dataset_id}/{dataset_version}/",
        "sampling": ds.get("mode", "domain_stratified_stream"),
        "label_key": ds.get("label_key", "source"),
    }
    weights_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "run_name": name,
        "id": mix["id"],
        "mix_weights": str(weights_path.resolve()),
        "weights": weights,
        "stream_seed": recipe_seed + int(mix["id"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="validation_mixtures_10b.json (domain-weight source of truth)",
    )
    ap.add_argument("--work", type=Path, required=True, help="Output directory for sidecars")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of run_name values (default: all recipe mixes)",
    )
    ap.add_argument(
        "--dataset-id",
        default=None,
        help=f"Override recipe data_source.dataset_id (default: recipe or {DEFAULT_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin version; default resolve_latest() against edullm-data",
    )
    ap.add_argument(
        "--stage-pool",
        type=Path,
        default=None,
        help="If set, stage a working pool from edullm-data into this directory",
    )
    ap.add_argument("--stage-seed", type=int, default=6198)
    ap.add_argument(
        "--skip-resolve",
        action="store_true",
        help="Do not call edullm-data (write dataset_id only; version may be unset)",
    )
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    ds = recipe.get("data_source") or {}
    dataset_id = args.dataset_id or ds.get("dataset_id") or DEFAULT_DATASET_ID
    dataset_version = args.dataset_version or ds.get("version")

    if args.skip_resolve:
        if not dataset_version:
            dataset_version = "latest"
    else:
        from edullm_data.s3 import Boto3S3

        s3 = Boto3S3.default()
        dataset_id, dataset_version = resolve_dataset(dataset_id, dataset_version, s3=s3)

    pool_meta = None
    if args.stage_pool is not None:
        if pool_is_ready(args.stage_pool):
            print(f"pool already ready: {args.stage_pool}")
            meta_path = args.stage_pool / "pool_meta.json"
            if meta_path.is_file():
                pool_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            pool_meta = stage_pool(
                out_dir=args.stage_pool,
                mixtures_json=args.recipe,
                budget_tokens=int(recipe.get("budget_tokens", 10_000_000_000)),
                dataset_id=dataset_id,
                dataset_version=None if dataset_version == "latest" else dataset_version,
                seed=int(args.stage_seed),
            )
            dataset_id = str(pool_meta["dataset_id"])
            dataset_version = str(pool_meta["version"])

    wanted = set(args.only) if args.only else None
    arms = []
    for mix in recipe["mixtures"]:
        if wanted is not None and mix["run_name"] not in wanted:
            continue
        arms.append(
            prepare_mix(
                mix,
                recipe,
                args.work,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
            )
        )
        print(f"prepared {mix['run_name']}")

    if not arms:
        raise SystemExit("no mixtures prepared")

    index = {
        "recipe": str(args.recipe.resolve()),
        "budget_tokens": recipe.get("budget_tokens"),
        "domain_order": recipe.get("domain_order"),
        "data_source": {
            "dataset_id": dataset_id,
            "version": dataset_version,
            "uri": f"s3://edullm-data/{dataset_id}/{dataset_version}/",
            "mode": "domain_stratified_stream",
            "label_key": "source",
        },
        "pool_dir": str(args.stage_pool.resolve()) if args.stage_pool else None,
        "pool_meta": pool_meta,
        "arms": arms,
    }
    out = args.work / "validation_arms.json"
    out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
