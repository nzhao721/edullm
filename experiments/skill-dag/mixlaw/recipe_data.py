#!/usr/bin/env python3
"""Shared recipe sidecar helpers for domain-stratified streaming."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mixlaw_common import DOMAINS, EDULLM_DATA_DATASET_ID, SEQ_LEN, token_budget

DEFAULT_STREAMING = "domain_stratified_stream"


def weights_list_to_dict(vec: list[float]) -> dict[str, float]:
    return {d: float(w) for d, w in zip(DOMAINS, vec)}


def load_mix_weights(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise SystemExit(f"{path}: missing weights dict")
    out = {d: float(weights[d]) for d in DOMAINS}
    return out, payload


def budget_tokens_from_recipe(recipe: dict, tokens_per_param: float | None) -> int:
    if recipe.get("budget_tokens"):
        return int(recipe["budget_tokens"])
    if tokens_per_param is None:
        raise SystemExit("recipe has no budget_tokens; pass --tokens-per-param")
    return int(token_budget(float(tokens_per_param))[2])


def prepare_arm_sidecar(
    *,
    run_name: str,
    arm_id: int | str,
    weights: dict[str, float],
    recipe: dict,
    recipe_path: Path,
    work: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = work / str(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_seed = int(recipe.get("seed", 6198))
    arm_int = int(arm_id) if str(arm_id).isdigit() else abs(hash(str(arm_id))) % 1_000_000
    budget = budget_tokens_from_recipe(recipe, recipe.get("_tokens_per_param"))
    data_source = recipe.get("data_source") or {}
    payload: dict[str, Any] = {
        "run_name": run_name,
        "id": arm_int,
        "domain_order": list(DOMAINS),
        "weights": weights,
        "recipe": recipe_path.name,
        "recipe_seed": recipe_seed,
        "stream_seed": recipe_seed + arm_int,
        "budget_tokens": budget,
        "length_tokens": budget,
        "total_steps": budget // (SEQ_LEN * 96),  # GLOBAL_BATCH_SEQS=96; overwritten if needed
        "sampling": DEFAULT_STREAMING,
    }
    if data_source.get("dataset_id"):
        payload["dataset_id"] = data_source["dataset_id"]
        if data_source.get("dataset_version"):
            payload["dataset_version"] = data_source["dataset_version"]
    elif data_source.get("pool_s3"):
        # Legacy callers (e.g. 370M validation / skillit) may still pin an S3 prefix.
        payload["olmohq_source"] = data_source["pool_s3"]
    else:
        payload["dataset_id"] = EDULLM_DATA_DATASET_ID
    if extra:
        payload.update(extra)
    weights_path = out_dir / "mix_weights.json"
    weights_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "run_name": run_name,
        "id": arm_int,
        "mix_weights": str(weights_path.resolve()),
        "weights": weights,
        "stream_seed": payload["stream_seed"],
        "budget_tokens": budget,
    }


def prepare_from_mixtures(
    recipe: dict,
    recipe_path: Path,
    work: Path,
    *,
    only: set[str] | None = None,
    name_key: str = "run_name",
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for mix in recipe["mixtures"]:
        run_name = str(mix.get(name_key) or f"mix{int(mix['id']):02d}")
        if only is not None and run_name not in only:
            continue
        weights = weights_list_to_dict(mix["weights"])
        extra = {k: mix[k] for k in ("tag", "source", "label", "a_mode") if k in mix}
        arms.append(
            prepare_arm_sidecar(
                run_name=run_name,
                arm_id=int(mix["id"]),
                weights=weights,
                recipe=recipe,
                recipe_path=recipe_path,
                work=work,
                extra=extra or None,
            )
        )
    return arms
