#!/usr/bin/env python3
"""Publish RegMix 10B curriculum token-order rankings to edullm-data.

Reads ``ranked_chunks_<metric>.npy`` from a local ``build_curriculum_index.py``
staging tree and publishes **one** curriculum dataset with four
``token-order/v1`` groups (compression, flesch, mtld, learnability), each a
permutation over ``pretrain/regmix-10b``.

Layout written under ``--stage-dir`` before ``publish()``::

  compression/train-00000.u32le.bin   # easy→hard global_chunk_idx (train split)
  compression/val-00000.u32le.bin     # identity permutation (held-out partition only)
  flesch/...
  mtld/...
  learnability/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Import metric names from the curriculum experiment package.
_SCRIPT_DIR = Path(__file__).resolve().parent
_CUR_CANDIDATES = (
    _SCRIPT_DIR.parent / "curriculum",  # FarmShare: RUN_DIR/scripts/curriculum
    _SCRIPT_DIR.parents[1] / "experiments" / "curriculum",  # repo checkout
)
for _cur in _CUR_CANDIDATES:
    if (_cur / "curriculum_pacing.py").is_file():
        if str(_cur) not in sys.path:
            sys.path.insert(0, str(_cur))
        break
else:
    raise SystemExit(
        "could not import curriculum_pacing: expected "
        f"{_CUR_CANDIDATES[0]} or {_CUR_CANDIDATES[1]} on sys.path"
    )

from curriculum_pacing import (  # noqa: E402
    CURRICULUM_DATASET_ID,
    CURRICULUM_ORDER_GROUP_FOR_METRIC,
    DIFFICULTY_METRICS,
)

DEFAULT_PARENT_DATASET_ID = "pretrain/regmix-10b"
DEFAULT_PURPOSE = (
    "RegMix 10B easy→hard token-order rankings (compression, flesch, mtld, "
    "learnability) for OLMo2-370M curriculum arms; indices into pretrain/regmix-10b"
)


def _load_index_manifest(index_dir: Path) -> dict:
    path = index_dir / "curriculum_manifest.json"
    if not path.is_file():
        raise SystemExit(f"missing curriculum build manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent")
    if payload.get("version") != 2 or not isinstance(parent, dict):
        raise SystemExit(
            f"{path}: legacy/document-local curriculum manifest is not publishable; "
            "rebuild with parent_pool_flat_chunks_v1"
        )
    required = (
        "dataset_id",
        "version",
        "manifest_sha256",
        "coordinate_model",
        "coordinate_sha256",
        "seq_len",
        "n_chunks",
    )
    missing = [key for key in required if parent.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"{path}: parent identity missing {missing}")
    if parent["coordinate_model"] != "parent_pool_flat_chunks_v1":
        raise SystemExit(
            f"{path}: unsupported coordinate model {parent['coordinate_model']!r}"
        )
    return payload


def _load_ranked(index_dir: Path, metric: str, *, expected_count: int) -> np.ndarray:
    path = index_dir / f"ranked_chunks_{metric}.npy"
    if not path.is_file():
        raise SystemExit(f"missing ranked order array: {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 1:
        raise SystemExit(f"{path}: expected 1-D order vector, got shape {arr.shape}")
    if arr.dtype.kind not in ("i", "u"):
        raise SystemExit(f"{path}: order dtype must be integer, got {arr.dtype}")
    if int(arr.size) != int(expected_count):
        raise SystemExit(
            f"{path}: wrong length {arr.size}; expected parent block_count={expected_count}"
        )
    if arr.size == 0:
        raise SystemExit(f"{path}: empty order vector")
    if arr.dtype.kind == "i" and int(arr.min()) < 0:
        raise SystemExit(f"{path}: negative parent chunk id {int(arr.min())}")
    if int(arr.max()) >= int(expected_count):
        raise SystemExit(
            f"{path}: out-of-range parent chunk id {int(arr.max())}; "
            f"valid range is [0, {expected_count})"
        )
    order = np.asarray(arr, dtype=np.uint32)
    if np.unique(order).size != order.size:
        raise SystemExit(f"{path}: indices are not unique — expected a permutation")
    if not np.array_equal(np.sort(order), np.arange(expected_count, dtype=np.uint32)):
        raise SystemExit(f"{path}: order is not a complete parent-pool permutation")
    return order


def _write_u32le(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(values, dtype="<u4").tobytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"internal error: {path} size {len(data)} not uint32-aligned")
    path.write_bytes(data)


def stage_curriculum_orders(
    *,
    index_dir: Path,
    out_root: Path,
    expected_block_count: int,
    metrics: tuple[str, ...] = DIFFICULTY_METRICS,
) -> dict[str, int]:
    """Write ``<group>/train-00000.u32le.bin`` (+ val identity) per metric."""
    lengths: dict[str, int] = {}
    for metric in metrics:
        group = CURRICULUM_ORDER_GROUP_FOR_METRIC[metric]
        order = _load_ranked(
            index_dir, metric, expected_count=int(expected_block_count)
        )
        n = int(order.size)
        lengths[group] = n
        group_dir = out_root / group
        _write_u32le(group_dir / "train-00000.u32le.bin", order)
        # Satisfy curriculum-family held-out partition requirement; trainers read train only.
        _write_u32le(group_dir / "val-00000.u32le.bin", np.arange(n, dtype=np.uint32))
        print(f"staged {group}: n_chunks={n:,} ({metric})", flush=True)
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise SystemExit(
            f"metric order lengths disagree across groups: {lengths}; "
            "expected one shared parent chunk count"
        )
    return lengths


def _resolve_parent_dep(
    *,
    parent_dataset_id: str,
    parent_version: str | None,
    block_count: int,
    seq_len: int,
    expected_manifest_sha256: str,
    expected_coordinate_sha256: str,
):
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    ver = parent_version or resolve_latest(parent_dataset_id, s3=s3)
    if not ver:
        raise SystemExit(
            f"parent {parent_dataset_id!r} is not published on edullm-data; "
            "publish pretrain/regmix-10b before curriculum orders"
        )
    import json as _json

    pds = _json.loads(
        s3.get("edullm-data", f"{parent_dataset_id}/{ver}/dataset.json").decode("utf-8")
    )
    groups = pds.get("groups") or []
    if not groups:
        raise SystemExit(f"parent {parent_dataset_id}/{ver} has no groups")
    token_groups = [
        group
        for group in groups
        if group.get("profile") == "pretrain-tokens/v1"
        or str(group.get("name") or "").lower() in {"tokens", "train"}
    ]
    if len(token_groups) != 1:
        raise SystemExit(
            f"parent {parent_dataset_id}/{ver}: expected exactly one "
            f"pretrain-tokens/v1 group, found {len(token_groups)}"
        )
    parent_group = token_groups[0]
    man_sha = parent_group.get("manifest_sha256")
    if not man_sha:
        raise SystemExit(f"parent {parent_dataset_id}/{ver} missing manifest_sha256")
    if man_sha != expected_manifest_sha256:
        raise SystemExit(
            "parent manifest mismatch: curriculum index was built for "
            f"{expected_manifest_sha256}, published {parent_dataset_id}/{ver} has {man_sha}"
        )
    manifest_path = parent_group.get("manifest")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise SystemExit(f"parent {parent_dataset_id}/{ver} missing group manifest path")
    manifest = _json.loads(
        s3.get(
            "edullm-data",
            f"{parent_dataset_id}/{ver}/{manifest_path}",
        ).decode("utf-8")
    )
    resolved = dataset_paths(
        parent_dataset_id,
        ver,
        split="train",
        group=parent_group.get("name"),
        s3=s3,
    )
    prefix = f"s3://edullm-data/{parent_dataset_id}/{ver}/"
    ordered_paths = [
        uri[len(prefix) :] if uri.startswith(prefix) else uri for uri in resolved.paths
    ]
    entries_by_path = {
        str(entry.get("path")): entry for entry in manifest.get("entries") or []
    }
    coordinate_rows = []
    source_starts: dict[str, int] = {}
    parent_blocks = 0
    for entry_path in ordered_paths:
        entry = entries_by_path.get(entry_path)
        if entry is None:
            raise SystemExit(
                f"dataset_paths returned {entry_path!r}, absent from parent manifest"
            )
        count = (entry.get("count") or {}).get("value")
        if not isinstance(count, int) or count < 0:
            raise SystemExit(
                f"parent {parent_dataset_id}/{ver} has an invalid train token count: {entry!r}"
            )
        labels = entry.get("labels") or {}
        source = labels.get("source")
        if not isinstance(source, str) or not source:
            parts = Path(entry_path).as_posix().split("/")
            if "tokens" not in parts or parts.index("tokens") + 1 >= len(parts):
                raise SystemExit(
                    f"parent entry {entry_path!r} lacks labels.source provenance"
                )
            source = parts[parts.index("tokens") + 1]
        source_start = source_starts.get(source, 0)
        n_chunks = max(0, (count - 1) // int(seq_len))
        coordinate_rows.append(
            {
                "path": entry_path,
                "count": count,
                "source": source,
                "source_token_start": source_start,
                "n_chunks": n_chunks,
            }
        )
        source_starts[source] = source_start + count
        parent_blocks += n_chunks
    coordinate_sha256 = hashlib.sha256(
        json.dumps(coordinate_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if coordinate_sha256 != expected_coordinate_sha256:
        raise SystemExit(
            "parent coordinate layout mismatch: curriculum index was built for "
            f"{expected_coordinate_sha256}, published layout is {coordinate_sha256}"
        )
    if parent_blocks != int(block_count):
        raise SystemExit(
            "curriculum order does not align with the published parent chunk layout: "
            f"ranking has {block_count:,} blocks, but {parent_dataset_id}/{ver} has "
            f"{parent_blocks:,} train blocks at seq_len={seq_len}. "
            "Build rankings from the parent shard layout; do not publish a re-tokenized "
            "document-local index as a token-order/v1 parent permutation."
        )
    return {
        "role": "token_pool",
        "dataset_id": parent_dataset_id,
        "version": ver,
        "manifest_sha256": man_sha,
        "coordinate_sha256": coordinate_sha256,
        "block_count": int(block_count),
    }


def _group_meta(*, parent_dep: dict, block_count: int) -> dict[str, dict]:
    partitions = [
        {"name": "train", "by": "path", "glob": "train-*.u32le.bin"},
        {"name": "val", "by": "path", "glob": "val-*.u32le.bin"},
    ]
    common = {
        "depends_on": [parent_dep],
        "block_count": int(block_count),
        "ordering": "permutation",
        "coverage": "overlapping",
        "partitions": partitions,
    }
    return {group: dict(common) for group in CURRICULUM_ORDER_GROUP_FOR_METRIC.values()}


def ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "edullm-data is not installed. Install from "
            "git+https://github.com/edu-llm/edullm-data@main "
            f"({e})"
        ) from e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Local build_curriculum_index.py output (ranked_chunks_*.npy)",
    )
    ap.add_argument("--stage-dir", type=Path, required=True, help="edullm-data publish layout root")
    ap.add_argument("--dataset-id", default=CURRICULUM_DATASET_ID)
    ap.add_argument("--parent-dataset-id", default=DEFAULT_PARENT_DATASET_ID)
    ap.add_argument(
        "--parent-version",
        default=None,
        help="Pin parent pool version (default: resolve_latest on edullm-data)",
    )
    ap.add_argument("--purpose", default=DEFAULT_PURPOSE)
    ap.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Parent training sequence length used to validate ranked block indices",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Replace existing stage-dir contents")
    args = ap.parse_args()

    if not args.index_dir.is_dir():
        raise SystemExit(f"--index-dir not found: {args.index_dir}")
    manifest = _load_index_manifest(args.index_dir)
    expected_parent = manifest["parent"]
    if args.parent_dataset_id != expected_parent["dataset_id"]:
        raise SystemExit(
            f"--parent-dataset-id {args.parent_dataset_id!r} != index parent "
            f"{expected_parent['dataset_id']!r}"
        )
    if args.parent_version and args.parent_version != expected_parent["version"]:
        raise SystemExit(
            f"--parent-version {args.parent_version!r} != index parent "
            f"{expected_parent['version']!r}"
        )
    if int(args.seq_len) != int(expected_parent["seq_len"]):
        raise SystemExit(
            f"--seq-len {args.seq_len} != index parent seq_len "
            f"{expected_parent['seq_len']}"
        )
    print(json.dumps({"curriculum_index": manifest.get("n_ranked")}, indent=2), flush=True)

    stage_dir = args.stage_dir
    if stage_dir.exists() and args.force:
        import shutil

        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    lengths = stage_curriculum_orders(
        index_dir=args.index_dir,
        out_root=stage_dir,
        expected_block_count=int(expected_parent["n_chunks"]),
    )
    block_count = next(iter(lengths.values()))

    profile = {group: "token-order/v1" for group in CURRICULUM_ORDER_GROUP_FOR_METRIC.values()}

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "dataset_id": args.dataset_id,
                    "parent": args.parent_dataset_id,
                    "parent_version": expected_parent["version"],
                    "parent_manifest_sha256": expected_parent["manifest_sha256"],
                    "coordinate_sha256": expected_parent["coordinate_sha256"],
                    "block_count": block_count,
                    "groups": sorted(profile),
                    "stage_dir": str(stage_dir.resolve()),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    ensure_edullm_data()
    from edullm_data.contracts import validate_dataset_id
    from edullm_data.publish import publish
    from edullm_data.s3 import Boto3S3

    try:
        validate_dataset_id(args.dataset_id)
    except Exception as exc:
        raise SystemExit(f"invalid dataset_id {args.dataset_id!r}: {exc}") from exc

    parent_dep = _resolve_parent_dep(
        parent_dataset_id=args.parent_dataset_id,
        parent_version=args.parent_version or expected_parent["version"],
        block_count=block_count,
        seq_len=args.seq_len,
        expected_manifest_sha256=expected_parent["manifest_sha256"],
        expected_coordinate_sha256=expected_parent["coordinate_sha256"],
    )
    group_meta = _group_meta(parent_dep=parent_dep, block_count=block_count)

    about = (
        "Four easy→hard token-order permutations over the RegMix 10B pretrain pool "
        f"({args.parent_dataset_id}/{parent_dep['version']}): compression ratio, "
        "Flesch reading ease, MTLD, and RefHQ learnability. Each group is a "
        "``token-order/v1`` index vector; curriculum trainers select a group and "
        "apply pacing in-process."
    )
    notes = (
        "Train split carries the published ranking; val split is an identity "
        "permutation over the same block count (curriculum-family held-out "
        "partition requirement). Trainers read split=train only."
    )

    created_at = datetime.now(timezone.utc).isoformat()
    plan = publish(
        stage_dir,
        dataset_id=args.dataset_id,
        purpose=args.purpose,
        profile=profile,
        s3=Boto3S3.default(),
        created_at=created_at,
        group_meta=group_meta,
        about=about,
        notes=notes,
    )
    print(
        json.dumps(
            {
                "dataset_id": plan.dataset_id,
                "version": plan.version,
                "payload_objects": len(plan.payload_keys),
                "block_count": block_count,
                "parent": parent_dep,
                "groups": sorted(profile),
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        f"published to s3://edullm-landing/{plan.dataset_id}/{plan.version}/ "
        f"(validator promotes to edullm-data)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
