#!/usr/bin/env python3
"""Deterministic seven-arm AWS Batch entrypoint for MixLaw 370M.

The platform supplies the array index, immutable dataset identity, workload
credentials, and durable S3 prefixes. This entrypoint never resolves ``latest``
and never includes the separately running ``mix01`` control arm.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from platform_artifacts import join_s3_prefix, parse_s3_prefix

ARRAY_ARMS: tuple[tuple[int, str], ...] = (
    (0, "olmo-mix-1124"),
    (7, "mix07"),
    (18, "mix18"),
    (25, "ML-pilot_caps"),
    (26, "ML-near-opt-4"),
    (27, "LGB-min1pct"),
    (28, "LGB-near-opt-8"),
)
EXCLUDED_ARMS = frozenset({"mix01"})
PINNED_DATASET_ID = "pretrain/olmo-127b"
PINNED_DATASET_VERSION = "v1"
PLATFORM_VCPUS = 16
GPU_RANKS = 8
THREADS_PER_RANK = PLATFORM_VCPUS // GPU_RANKS
_INDEX = re.compile(r"0|[1-9][0-9]*")


class PlatformLaunchError(ValueError):
    """The platform environment or checked-in recipe violates the launch contract."""


@dataclass(frozen=True)
class PlatformLaunch:
    arm_index: int
    mix_id: int
    mix_name: str
    scratch_dir: Path
    command: tuple[str, ...]
    environment: dict[str, str]


def array_arm(raw_index: str | None) -> tuple[int, int, str]:
    if raw_index is None:
        raise PlatformLaunchError("AWS_BATCH_JOB_ARRAY_INDEX is required")
    if not _INDEX.fullmatch(raw_index):
        raise PlatformLaunchError(
            f"AWS_BATCH_JOB_ARRAY_INDEX must be a base-10 integer, got {raw_index!r}"
        )
    index = int(raw_index)
    if index >= len(ARRAY_ARMS):
        raise PlatformLaunchError(
            f"AWS_BATCH_JOB_ARRAY_INDEX must be in 0..{len(ARRAY_ARMS) - 1}, got {index}"
        )
    mix_id, mix_name = ARRAY_ARMS[index]
    if mix_name in EXCLUDED_ARMS:
        raise PlatformLaunchError(f"excluded control arm selected: {mix_name}")
    return index, mix_id, mix_name


def _job_scratch(job_id: str, scratch_root: Path) -> Path:
    if not job_id or job_id in {".", ".."} or "/" in job_id or "\x00" in job_id:
        raise PlatformLaunchError(f"unsafe AWS_BATCH_JOB_ID: {job_id!r}")
    return scratch_root / job_id


def _selected_recipe(recipe_path: Path, mix_id: int, mix_name: str) -> tuple[dict, dict]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    domain_order = recipe.get("domain_order")
    if not isinstance(domain_order, list) or not domain_order:
        raise PlatformLaunchError(f"{recipe_path}: missing domain_order")
    matches = [
        mix
        for mix in recipe.get("mixtures", [])
        if mix.get("id") == mix_id and mix.get("run_name") == mix_name
    ]
    if len(matches) != 1:
        raise PlatformLaunchError(
            f"{recipe_path}: expected one id={mix_id} run_name={mix_name!r}, got {len(matches)}"
        )
    mix = matches[0]
    weights = mix.get("weights")
    if not isinstance(weights, list) or len(weights) != len(domain_order):
        raise PlatformLaunchError(f"{recipe_path}: invalid weights for {mix_name}")
    if mix_name in EXCLUDED_ARMS:
        raise PlatformLaunchError(f"{recipe_path}: excluded control arm selected")
    return recipe, mix


def write_selected_sidecar(
    recipe_path: Path,
    destination: Path,
    *,
    mix_id: int,
    mix_name: str,
    dataset_id: str,
    dataset_version: str,
) -> Path:
    """Write exactly one deterministic arm sidecar from the checked-in recipe."""
    recipe, mix = _selected_recipe(recipe_path, mix_id, mix_name)
    domains = list(recipe["domain_order"])
    recipe_seed = int(recipe.get("seed", 6198))
    payload = {
        "run_name": mix_name,
        "id": mix_id,
        "tag": mix.get("tag"),
        "source": mix.get("source"),
        "domain_order": domains,
        "weights": {domain: float(weight) for domain, weight in zip(domains, mix["weights"])},
        "recipe": recipe_path.name,
        "recipe_seed": recipe_seed,
        "stream_seed": recipe_seed + mix_id,
        "budget_tokens": int(recipe["budget_tokens"]),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "edullm_data_uri": f"s3://edullm-data/{dataset_id}/{dataset_version}/",
        "sampling": "domain_stratified_stream",
        "label_key": "source",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(destination)
    return destination


def _require_platform_identity(environ: Mapping[str, str]) -> tuple[str, str, str, str]:
    dataset_id = environ.get("EDULLM_DATASET_ID", "")
    dataset_version = environ.get("EDULLM_DATASET_VERSION", "")
    if dataset_id != PINNED_DATASET_ID:
        raise PlatformLaunchError(
            f"EDULLM_DATASET_ID must be {PINNED_DATASET_ID!r}, got {dataset_id!r}"
        )
    if dataset_version != PINNED_DATASET_VERSION:
        raise PlatformLaunchError(
            f"EDULLM_DATASET_VERSION must be {PINNED_DATASET_VERSION!r}, "
            f"got {dataset_version!r}"
        )
    checkpoint_prefix = environ.get("EDULLM_CHECKPOINT_DIR", "")
    output_prefix = environ.get("EDULLM_OUTPUT_PREFIX", "")
    parse_s3_prefix(checkpoint_prefix)
    parse_s3_prefix(output_prefix)
    project = environ.get("EDULLM_WANDB_PROJECT") or environ.get("WANDB_PROJECT")
    if project != "mixlaw":
        raise PlatformLaunchError(
            f"platform W&B project must be 'mixlaw', got {project!r}"
        )
    return dataset_id, dataset_version, checkpoint_prefix, output_prefix


def bounded_thread_environment() -> dict[str, str]:
    threads = str(THREADS_PER_RANK)
    return {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
        "TOKENIZERS_PARALLELISM": "false",
        "DATA_LOADER_NUM_WORKERS": "0",
        "EDULLM_DATA_LOADER_WORKERS": "0",
    }


def prepare_launch(
    environ: Mapping[str, str],
    *,
    script_dir: Path | None = None,
    scratch_root: Path | None = None,
) -> PlatformLaunch:
    script_dir = Path(script_dir or Path(__file__).resolve().parent)
    repo_root = script_dir.parents[2]
    recipe_path = script_dir / "validation_mixtures_10b.json"
    ladder_config = script_dir / "ladder_base_config.yaml"
    launch_script = script_dir / "launch_validation_370m.sh"
    for required in (recipe_path, ladder_config, launch_script):
        if not required.is_file():
            raise PlatformLaunchError(f"required runtime file is missing: {required}")

    arm_index, mix_id, mix_name = array_arm(environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    job_id = environ.get("AWS_BATCH_JOB_ID", "")
    root = Path(scratch_root or environ.get("MIXLAW_SCRATCH_ROOT", "/scratch"))
    scratch_dir = _job_scratch(job_id, root)
    dataset_id, dataset_version, checkpoint_base, output_base = _require_platform_identity(
        environ
    )

    work_dir = scratch_dir / "work"
    pool_dir = scratch_dir / "pool"
    progress_dir = scratch_dir / "progress"
    task_loss_dir = scratch_dir / "task-loss"
    save_dir = scratch_dir / "checkpoints"
    logs_dir = scratch_dir / "logs"
    for path in (work_dir, pool_dir, progress_dir, task_loss_dir, save_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    weights_path = write_selected_sidecar(
        recipe_path,
        work_dir / mix_name / "mix_weights.json",
        mix_id=mix_id,
        mix_name=mix_name,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    arm_key = f"array/{arm_index:02d}-{mix_name}"
    checkpoint_prefix = join_s3_prefix(checkpoint_base, arm_key)
    output_prefix = join_s3_prefix(output_base, arm_key)
    run_id = f"{environ.get('EDULLM_RUN_ID', job_id)}-{mix_name}"

    child_env = dict(environ)
    child_env.update(bounded_thread_environment())
    child_env.update(
        {
            "MIXLAW_PLATFORM": "1",
            "EDULLM_ROOT": str(repo_root),
            "MIX_NAME": mix_name,
            "MIX_WEIGHTS_JSON": str(weights_path),
            "SAVE_FOLDER": str(save_dir),
            "PROGRESS_DIR": str(progress_dir),
            "POOL_DIR": str(pool_dir),
            "STAGE_DIR": str(pool_dir),
            "TASK_LOSS_RESULTS_DIR": str(task_loss_dir),
            "CHECKPOINT_PREFIX": checkpoint_prefix,
            "OUTPUT_PREFIX": output_prefix,
            "DATASET_ID": dataset_id,
            "DATASET_VERSION": dataset_version,
            "NPROC": str(GPU_RANKS),
            "TASK_LOSS_NPROC": str(GPU_RANKS),
            "LADDER_BASE_CONFIG": str(ladder_config),
            "RECOVERY_MODE": "fail",
            "WANDB_PROJECT": "mixlaw",
            "WANDB_GROUP": "370m-validation",
            "WANDB_RUN_NAME": f"mixlaw-370m-{mix_name}",
            "WANDB_RUN_ID": run_id,
            "NAME": f"mixlaw-370m-{mix_name}",
            "PYTHONUNBUFFERED": "1",
            "PYTHON": sys.executable,
        }
    )
    return PlatformLaunch(
        arm_index=arm_index,
        mix_id=mix_id,
        mix_name=mix_name,
        scratch_dir=scratch_dir,
        command=("/bin/bash", str(launch_script)),
        environment=child_env,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-prefix",
        default=None,
        help="Must equal platform-injected EDULLM_CHECKPOINT_DIR when supplied",
    )
    args = parser.parse_args(argv)
    if (
        args.checkpoint_prefix is not None
        and args.checkpoint_prefix != os.environ.get("EDULLM_CHECKPOINT_DIR")
    ):
        print(
            "[mixlaw-platform] launch refused: --checkpoint-prefix does not match "
            "EDULLM_CHECKPOINT_DIR",
            file=sys.stderr,
        )
        return 2
    try:
        launch = prepare_launch(os.environ)
    except Exception as exc:  # noqa: BLE001 - one fail-closed Batch diagnostic
        print(f"[mixlaw-platform] launch refused: {exc}", file=sys.stderr)
        return 2
    print(
        f"[mixlaw-platform] array_index={launch.arm_index} mix={launch.mix_name} "
        f"scratch={launch.scratch_dir} vcpus={PLATFORM_VCPUS} ranks={GPU_RANKS} "
        f"threads_per_rank={THREADS_PER_RANK}",
        file=sys.stderr,
        flush=True,
    )
    os.execve(launch.command[0], launch.command, launch.environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
