#!/usr/bin/env python3
"""Pure runtime contracts for MixLaw 370M launch and recovery."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

MIXLAW_DEFAULT_LENGTH_TOKENS = 10_000_000_000
MIXLAW_GLOBAL_BATCH_TOKENS = 4_194_304
MIXLAW_DEFAULT_STEPS = MIXLAW_DEFAULT_LENGTH_TOKENS // MIXLAW_GLOBAL_BATCH_TOKENS
RECOVERY_MODES = ("fresh", "resume", "fail")

OLMES_BPB_LABELS = (
    "arc_challenge_val_rc_5shot_bpb",
    "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
)

DEPENDENCY_DISTRIBUTIONS = (
    "torch",
    "ai2-olmo-core",
    "ai2-olmo",
    "edullm-data",
    "datasets",
    "torchmetrics",
    "scikit-learn",
)


class MixLawContractError(ValueError):
    """A launch or runtime contract is ambiguous or unsafe."""


def collect_dependency_versions(
    distributions: Sequence[str] = DEPENDENCY_DISTRIBUTIONS,
) -> dict[str, str]:
    """Return installed distribution versions without importing GPU libraries."""
    versions: dict[str, str] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "<missing>"
    return versions


def dependency_contract_errors(
    versions: Mapping[str, str],
    available_olmes_labels: Sequence[str],
    *,
    ladder_base_config: Path | None,
    eval_script: Path | None,
) -> list[str]:
    """Validate package presence, current edullm-data, OLMES labels, and files."""
    errors: list[str] = []
    for name in DEPENDENCY_DISTRIBUTIONS:
        if versions.get(name, "<missing>") == "<missing>":
            errors.append(f"missing required distribution: {name}")
    edullm_version = str(versions.get("edullm-data", "<missing>"))
    if edullm_version.startswith("0.2."):
        errors.append(
            f"edullm-data {edullm_version} is obsolete; install the newest release/main"
        )
    missing_labels = sorted(set(OLMES_BPB_LABELS).difference(available_olmes_labels))
    if missing_labels:
        errors.append(
            f"OLMES BPB label map is incomplete: missing {len(missing_labels)}/"
            f"{len(OLMES_BPB_LABELS)} labels (first={missing_labels[0]!r})"
        )
    if ladder_base_config is None or not Path(ladder_base_config).is_file():
        errors.append(f"missing LADDER_BASE_CONFIG: {ladder_base_config}")
    if eval_script is None or not Path(eval_script).is_file():
        errors.append(f"missing task-loss evaluator: {eval_script}")
    return errors


def production_contract_errors(
    *,
    durable_export: bool,
    task_loss_on_save: bool,
    task_loss_strict: bool,
) -> list[str]:
    """Production durability (local scratch + W&B) requires strict eval."""
    if not durable_export:
        return []
    errors: list[str] = []
    if not task_loss_on_save:
        errors.append("production durability requires --task-loss-on-save")
    if not task_loss_strict:
        errors.append("production durability requires --task-loss-strict")
    return errors


def _extract_load_path(args: Sequence[str]) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--load-path":
            if index + 1 >= len(args):
                raise MixLawContractError("--load-path requires a value")
            values.append(args[index + 1])
            index += 2
            continue
        if item.startswith("--load-path="):
            values.append(item.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise MixLawContractError("multiple --load-path values are not allowed")
    return values[0] if values else None


def checkpoint_uri_from_durable_metadata(
    metadata_path: Path,
    *,
    mix_name: str,
) -> str:
    """Resolve and validate one checkpoint URI from durable-step metadata."""
    path = Path(metadata_path)
    if not path.is_file():
        raise MixLawContractError(f"resume metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MixLawContractError(f"resume metadata is not an object: {path}")
    step = payload.get("last_durable_step")
    uri = payload.get("checkpoint_uri")
    if not isinstance(step, int) or step < 0:
        raise MixLawContractError(f"invalid last_durable_step in {path}")
    if not isinstance(uri, str) or not uri.strip():
        raise MixLawContractError(f"missing checkpoint_uri in {path}")
    step_name = f"step{step}"
    if uri.startswith("s3://"):
        expected_prefix = (
            f"s3://edullm-checkpoints/mixlaw/370m-validation/{mix_name}/checkpoints/"
        )
        if not uri.startswith(expected_prefix) or uri.rstrip("/").split("/")[-1] != step_name:
            raise MixLawContractError(
                f"durable metadata checkpoint does not match mix={mix_name!r} "
                f"step={step}: {uri}"
            )
    elif Path(uri).name != step_name:
        raise MixLawContractError(
            f"durable metadata checkpoint does not match step={step}: {uri}"
        )
    return uri


def resolve_recovery_args(
    mode: str,
    extra_args: Sequence[str],
    *,
    mix_name: str,
    durable_metadata_path: Path | None = None,
) -> list[str]:
    """Return unambiguous trainer args for ``fresh|resume|fail``."""
    normalized = str(mode).strip().lower()
    if normalized not in RECOVERY_MODES:
        raise MixLawContractError(
            f"RECOVERY_MODE must be one of {'|'.join(RECOVERY_MODES)} (got {mode!r})"
        )
    args = list(extra_args)
    has_fresh = "--fresh" in args
    load_path = _extract_load_path(args)
    if has_fresh and load_path:
        raise MixLawContractError("--fresh and --load-path are mutually exclusive")

    if normalized == "fresh":
        if load_path:
            raise MixLawContractError("RECOVERY_MODE=fresh rejects --load-path")
        if not has_fresh:
            args.append("--fresh")
        return args

    if normalized == "resume":
        if has_fresh:
            raise MixLawContractError("RECOVERY_MODE=resume rejects --fresh")
        if load_path:
            return args
        if durable_metadata_path is None:
            raise MixLawContractError(
                "RECOVERY_MODE=resume requires --load-path or DURABLE_METADATA_PATH"
            )
        uri = checkpoint_uri_from_durable_metadata(
            durable_metadata_path,
            mix_name=mix_name,
        )
        return [*args, "--load-path", uri]

    if has_fresh or load_path:
        raise MixLawContractError(
            "RECOVERY_MODE=fail rejects --fresh/--load-path; it fails on leftovers"
        )
    return args


def _recovery_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=RECOVERY_MODES)
    parser.add_argument("--mix-name", required=True)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--durable-metadata", type=Path)
    args = parser.parse_args(argv)
    resolved = resolve_recovery_args(
        args.mode,
        shlex.split(args.extra_args),
        mix_name=args.mix_name,
        durable_metadata_path=args.durable_metadata,
    )
    for item in resolved:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(_recovery_cli())
