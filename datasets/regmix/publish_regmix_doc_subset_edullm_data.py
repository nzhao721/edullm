#!/usr/bin/env python3
"""Publish one document-filtered RegMix token corpus to edullm-data.

The filter and materialization steps create ``tokenized/<source>/<source>.npy``
memmaps. This wrapper turns those into a valid ``pretrain-tokens/v1`` dataset:
it shards raw uint32 tokens, carves a per-source held-out split, and publishes
through the edullm-data landing airlock.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from publish_regmix_edullm_data import (
    DEFAULT_SHARD_BYTES,
    DEFAULT_TOKENIZER,
    VAL_FRACTION,
    build_manifest_from_tokenized,
    build_sources,
    ensure_edullm_data,
    stage_publish_layout,
)


def _read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-root", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--selection-description",
        required=True,
        help="Exact score, checkpoint, direction, and token-mass selection rule.",
    )
    parser.add_argument(
        "--purpose",
        required=True,
        help="What consumes this corpus and which experiment decision it supports.",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selection = _read_json(args.selection_manifest, label="selection manifest")
    manifest = build_manifest_from_tokenized(args.tokenized_root)
    if not manifest.get("accepted", True):
        raise SystemExit("tokenized manifest is not accepted")

    stage_publish_layout(
        tokenized_root=args.tokenized_root,
        manifest=manifest,
        out_root=args.stage_dir,
        shard_bytes=args.shard_bytes,
        force=args.force,
        val_fraction=args.val_fraction,
    )

    summary = {
        "dataset_id": args.dataset_id,
        "tokenized_root": str(args.tokenized_root),
        "selection_manifest": str(args.selection_manifest),
        "selection": args.selection_description,
        "selection_summary": selection,
        "tokens_before_val_carve": manifest["total_stream_tokens_with_eos"],
        "stage_dir": str(args.stage_dir),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    ensure_edullm_data()
    from edullm_data.contracts import validate_dataset_id
    from edullm_data.publish import publish
    from edullm_data.s3 import Boto3S3

    try:
        validate_dataset_id(args.dataset_id)
    except Exception as exc:
        raise SystemExit(f"invalid dataset id {args.dataset_id!r}: {exc}") from exc

    about = (
        "A document-filtered, re-tokenized subset of the RegMix 10B corpus. "
        f"{args.selection_description} "
        "Tokens are grouped by original RegMix source; held-out data is separate from train."
    )
    notes = (
        "Scores are finalized document-level RegMix LM labels from the RefHQ-on-5.5B "
        "late-checkpoint average. The source corpus is re-tokenized with "
        f"{args.tokenizer}; validation is a {args.val_fraction:.4%} per-source carve "
        "from the selected subset. Selection summary: "
        + json.dumps(selection, sort_keys=True)
    )
    plan = publish(
        args.stage_dir,
        dataset_id=args.dataset_id,
        purpose=args.purpose,
        profile="pretrain-tokens/v1",
        tokenizer=args.tokenizer,
        s3=Boto3S3.default(),
        created_at=datetime.now(timezone.utc).isoformat(),
        hash_workers=args.hash_workers,
        copy_workers=args.copy_workers,
        about=about,
        sources=build_sources(manifest),
        notes=notes,
    )
    print(
        json.dumps(
            {
                "dataset_id": plan.dataset_id,
                "version": plan.version,
                "payload_objects": len(plan.payload_keys),
                "landing_prefix": f"s3://edullm-landing/{plan.dataset_id}/{plan.version}/",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
