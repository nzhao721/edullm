#!/usr/bin/env python3
"""Prepare Skill-It 370M working pool + per-arm recipe sidecars from edullm-data.

Fetches published+validated domain shards via ``edullm_data.read.dataset_paths`` /
``resolve_latest`` (pinned to ``pretrain/olmo-127b/v1``), stages them under
``--pool-dir`` on a clean machine, and writes per-arm weight sidecars::

    <pool-dir>/tokens/<domain>/train-*.u32le.bin
    <pool-dir>/edullm_data_source.json
    <pool-dir>/_EDULLM_DATA_SOURCE.json  # compatibility alias
    <work>/<arm_id>/arm_weights.json
    <work>/skillit_arms.json

Does **not** assume FarmShare scratch or laptop-local pools already exist, and
does **not** read ``s3://edullm-datasets/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

_SKILLIT = Path(__file__).resolve().parent
_MIXLAW = _SKILLIT.parent / "mixlaw"
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

from mixlaw_common import DOMAINS  # noqa: E402

DEFAULT_RECIPE = _SKILLIT / "skillit_train_recipe.json"
DEFAULT_DATASET_ID = "pretrain/olmo-127b"
DEFAULT_DATASET_VERSION = "v1"
SOURCE_MARKER = "edullm_data_source.json"
SOURCE_MARKER_ALIASES = (SOURCE_MARKER, "_EDULLM_DATA_SOURCE.json")
_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def _initial_weights(recipe: dict) -> dict[str, float]:
    if "initial_weights" in recipe:
        vec = recipe["initial_weights"]
        return {d: float(w) for d, w in zip(DOMAINS, vec)}
    base = recipe.get("base_weights")
    if isinstance(base, dict):
        return {d: float(base[d]) for d in DOMAINS}
    raise SystemExit("recipe missing initial_weights or base_weights")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    m = _S3_URI_RE.match(uri.strip())
    if not m:
        raise ValueError(f"not an s3 uri: {uri!r}")
    return m.group(1), m.group(2)


def _dtype_nbytes(dtype: str | None) -> int:
    if dtype is None:
        return 4
    d = dtype.lower().replace("numpy.", "")
    if d in ("uint32", "int32", "<u4", "u4"):
        return 4
    if d in ("uint16", "int16", "<u2", "u2"):
        return 2
    if d in ("uint64", "int64", "<u8", "u8"):
        return 8
    return 4


def load_pool_source(pool_dir: Path) -> tuple[dict[str, Any], Path]:
    """Load either SkillIt/mixlaw provenance marker convention."""
    found: list[tuple[dict[str, Any], Path]] = []
    for name in SOURCE_MARKER_ALIASES:
        path = pool_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid pool provenance JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"pool provenance must be an object: {path}")
        found.append((payload, path))
    if not found:
        expected = " or ".join(str(pool_dir / name) for name in SOURCE_MARKER_ALIASES)
        raise SystemExit(f"missing pool provenance marker: {expected}")
    first, first_path = found[0]
    first_identity = (
        first.get("dataset_id"),
        first.get("version") or first.get("dataset_version"),
    )
    for payload, path in found[1:]:
        identity = (
            payload.get("dataset_id"),
            payload.get("version") or payload.get("dataset_version"),
        )
        if identity != first_identity:
            raise SystemExit(
                "conflicting pool provenance markers: "
                f"{first_path}={first_identity!r}, {path}={identity!r}"
            )
    return first, first_path


def validate_pool_source(
    pool_dir: Path,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    version: str = DEFAULT_DATASET_VERSION,
    require_370m_layout: bool = False,
) -> dict[str, Any]:
    """Fail closed unless a staged pool has the pinned published identity."""
    source, marker = load_pool_source(pool_dir)
    got = (
        source.get("dataset_id"),
        source.get("version") or source.get("dataset_version"),
    )
    expected = (dataset_id, version)
    if got != expected:
        raise SystemExit(
            f"{marker}: staged pool source {got!r} does not match pinned "
            f"edullm-data source {expected!r}"
        )
    if (source.get("data_bucket") or source.get("bucket")) != "edullm-data":
        raise SystemExit(f"{marker}: pool is not sourced from s3://edullm-data/")
    if require_370m_layout:
        missing = [
            domain
            for domain in DOMAINS
            if not any((pool_dir / "tokens" / domain).glob("train-*.u32le.bin"))
        ]
        if missing:
            raise SystemExit(
                f"{marker}: pool lacks 370M tokens/<domain>/train-*.u32le.bin "
                f"for {missing}"
            )
    source = dict(source)
    source["version"] = got[1]
    return source


def resolve_dataset(
    dataset_id: str,
    version: Optional[str] = None,
) -> tuple[str, Any]:
    """Return ``(version, s3_client)`` after resolving latest published version."""
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    if version:
        return version, s3
    from edullm_data.read import resolve_latest

    ver = resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(f"no published versions for {dataset_id!r} in s3://edullm-data/")
    return ver, s3


def domain_train_paths(
    dataset_id: str,
    version: str,
    domain: str,
    *,
    s3: Any,
) -> Any:
    from edullm_data.read import dataset_paths

    return dataset_paths(
        dataset_id,
        version,
        split="train",
        s3=s3,
        labels={"source": domain},
    )


def stage_domain_shards(
    *,
    dataset_id: str,
    version: str,
    domain: str,
    pool_dir: Path,
    s3: Any,
    max_tokens: Optional[int],
    boto3_client: Any,
) -> dict[str, Any]:
    """Download train shards for one domain into ``pool_dir/tokens/<domain>/``."""
    resolved = domain_train_paths(dataset_id, version, domain, s3=s3)
    if not resolved.paths:
        raise SystemExit(
            f"{dataset_id}/{version}: no train shards for source={domain!r}"
        )
    dtype = resolved.dtype or "uint32"
    nbytes = _dtype_nbytes(dtype)
    out_dir = pool_dir / "tokens" / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []
    tokens_kept = 0
    for uri in resolved.paths:
        bucket, key = _parse_s3_uri(uri)
        name = Path(key).name
        dest = out_dir / name
        head = boto3_client.head_object(Bucket=bucket, Key=key)
        size = int(head["ContentLength"])
        shard_tokens = size // nbytes
        if dest.is_file() and dest.stat().st_size == size:
            pass  # reuse existing
        else:
            tmp = dest.with_suffix(dest.suffix + ".partial")
            boto3_client.download_file(bucket, key, str(tmp))
            tmp.replace(dest)
        downloaded.append(
            {
                "uri": uri,
                "path": str(dest.relative_to(pool_dir)).replace("\\", "/"),
                "bytes": size,
                "tokens": shard_tokens,
            }
        )
        tokens_kept += shard_tokens
        if max_tokens is not None and tokens_kept >= int(max_tokens):
            break

    return {
        "domain": domain,
        "dtype": dtype,
        "rows": int(resolved.rows) if resolved.rows is not None else None,
        "tokens_staged": tokens_kept,
        "shards": downloaded,
    }


def stage_working_pool(
    *,
    pool_dir: Path,
    dataset_id: str,
    version: Optional[str],
    max_tokens_per_domain: Optional[int],
    skip_if_complete: bool = True,
) -> dict[str, Any]:
    """Stage domain train shards from edullm-data into ``pool_dir``."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    marker_path = pool_dir / SOURCE_MARKER
    ver, s3 = resolve_dataset(dataset_id, version)
    if dataset_id != DEFAULT_DATASET_ID or ver != DEFAULT_DATASET_VERSION:
        raise SystemExit(
            "SkillIt 370M source is pinned to "
            f"{DEFAULT_DATASET_ID}/{DEFAULT_DATASET_VERSION}; got {dataset_id}/{ver}"
        )

    if skip_if_complete and any((pool_dir / name).is_file() for name in SOURCE_MARKER_ALIASES):
        prev = validate_pool_source(
            pool_dir,
            dataset_id=dataset_id,
            version=ver,
            require_370m_layout=True,
        )
        print(f"pool already staged: {marker_path}")
        return prev

    import boto3

    boto3_client = boto3.client("s3")
    domains_meta = []
    for domain in DOMAINS:
        print(f"staging {dataset_id}/{ver} source={domain} …")
        meta = stage_domain_shards(
            dataset_id=dataset_id,
            version=ver,
            domain=domain,
            pool_dir=pool_dir,
            s3=s3,
            max_tokens=max_tokens_per_domain,
            boto3_client=boto3_client,
        )
        print(
            f"  {domain}: {meta['tokens_staged']} tokens "
            f"({len(meta['shards'])} shards, dtype={meta['dtype']})"
        )
        domains_meta.append(meta)

    dtypes = {m["dtype"] for m in domains_meta}
    if len(dtypes) != 1:
        raise SystemExit(f"inconsistent dtypes across domains: {sorted(dtypes)}")

    payload = {
        "dataset_id": dataset_id,
        "version": ver,
        "data_bucket": "edullm-data",
        "split": "train",
        "label_key": "source",
        "domains": list(DOMAINS),
        "dtype": next(iter(dtypes)),
        "max_tokens_per_domain": max_tokens_per_domain,
        "layout": "tokens/<domain>/train-*.u32le.bin",
        "domain_meta": domains_meta,
        "reader": "edullm_data.read.dataset_paths + resolve_latest",
    }
    marker_text = json.dumps(payload, indent=2) + "\n"
    for name in SOURCE_MARKER_ALIASES:
        (pool_dir / name).write_text(marker_text, encoding="utf-8")
    print(f"wrote {marker_path}")
    return payload


def prepare_arm(arm: dict, recipe: dict, work: Path, weights: dict[str, float], source: dict) -> dict:
    arm_id = arm["arm_id"]
    out_dir = work / arm_id
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "arm_weights.json"
    recipe_seed = int(recipe.get("seed", 42))
    payload = {
        "arm_id": arm_id,
        "a_mode": arm["a_mode"],
        "id": arm["id"],
        "label": arm.get("label"),
        "domain_order": list(DOMAINS),
        "weights": weights,
        "recipe": str(DEFAULT_RECIPE.name),
        "recipe_seed": recipe_seed,
        "stream_seed": recipe_seed + int(arm["id"]),
        "budget_tokens": recipe.get("budget_tokens"),
        "edullm_data": {
            "dataset_id": source.get("dataset_id"),
            "version": source.get("version"),
            "dtype": source.get("dtype"),
        },
        "sampling": "domain_stratified_stream",
        "skillit": recipe.get("skillit", {}),
    }
    weights_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "arm_id": arm_id,
        "a_mode": arm["a_mode"],
        "id": arm["id"],
        "arm_weights": str(weights_path.resolve()),
        "weights": weights,
        "stream_seed": recipe_seed + int(arm["id"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="skillit_train_recipe.json (domain-weight source of truth)",
    )
    ap.add_argument("--work", type=Path, default=None, help="Output directory for arm sidecars")
    ap.add_argument(
        "--pool-dir",
        type=Path,
        required=True,
        help="Staging root for edullm-data domain shards (created if missing)",
    )
    ap.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help=f"edullm-data dataset id (default: recipe data_source.dataset_id or {DEFAULT_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        type=str,
        default=DEFAULT_DATASET_VERSION,
        help=f"Pinned version (required contract: {DEFAULT_DATASET_VERSION})",
    )
    ap.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=None,
        help="Cap staged tokens per domain (default: recipe budget_tokens)",
    )
    ap.add_argument(
        "--skip-stage",
        action="store_true",
        help="Only write arm sidecars; require an already-staged pool with marker",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of arm_id values (default: all recipe arms)",
    )
    ap.add_argument(
        "--validate-pool-only",
        action="store_true",
        help="Validate pinned pool provenance/layout and exit without staging",
    )
    ap.add_argument(
        "--pool-layout",
        choices=("370m", "probe"),
        default="370m",
        help="Expected local payload layout when validating an existing pool",
    )
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    data_src = recipe.get("data_source") or {}
    dataset_id = (
        args.dataset_id
        or data_src.get("dataset_id")
        or DEFAULT_DATASET_ID
    )
    version = args.dataset_version or data_src.get("version") or DEFAULT_DATASET_VERSION
    if dataset_id != DEFAULT_DATASET_ID or version != DEFAULT_DATASET_VERSION:
        raise SystemExit(
            "SkillIt source is pinned to "
            f"{DEFAULT_DATASET_ID}/{DEFAULT_DATASET_VERSION}; got {dataset_id}/{version}"
        )
    if args.validate_pool_only:
        source = validate_pool_source(
            args.pool_dir,
            dataset_id=dataset_id,
            version=version,
            require_370m_layout=args.pool_layout == "370m",
        )
        print(
            f"validated pool source {source['dataset_id']}/{source['version']} "
            f"at {args.pool_dir}"
        )
        return 0
    if args.work is None:
        raise SystemExit("--work is required unless --validate-pool-only is used")
    max_tok = args.max_tokens_per_domain
    if max_tok is None and recipe.get("budget_tokens") is not None:
        max_tok = int(recipe["budget_tokens"])

    if args.skip_stage:
        source = validate_pool_source(
            args.pool_dir,
            dataset_id=dataset_id,
            version=version,
            require_370m_layout=True,
        )
    else:
        source = stage_working_pool(
            pool_dir=args.pool_dir,
            dataset_id=dataset_id,
            version=version,
            max_tokens_per_domain=max_tok,
            skip_if_complete=True,
        )

    weights = _initial_weights(recipe)
    wanted = set(args.only) if args.only else None
    arms = []
    for arm in recipe["arms"]:
        if wanted is not None and arm["arm_id"] not in wanted:
            continue
        arms.append(prepare_arm(arm, recipe, args.work, weights, source))
        print(f"prepared {arm['arm_id']} ({arm['a_mode']})")

    if not arms:
        raise SystemExit("no arms prepared")

    index = {
        "recipe": str(args.recipe.resolve()),
        "budget_tokens": recipe.get("budget_tokens"),
        "domain_order": recipe.get("domain_order") or list(DOMAINS),
        "data_source": {
            "dataset_id": source.get("dataset_id", dataset_id),
            "version": source.get("version"),
            "bucket": "edullm-data",
            "mode": "domain_stratified_stream",
            "pool_dir": str(args.pool_dir.resolve()),
            "dtype": source.get("dtype"),
        },
        "initial_weights": weights,
        "skillit": recipe.get("skillit", {}),
        "arms": arms,
    }
    out = args.work / "skillit_arms.json"
    args.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(arms)} arms)")
    print(f"POOL_DIR={args.pool_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
