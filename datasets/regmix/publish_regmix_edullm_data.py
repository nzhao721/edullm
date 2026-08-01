#!/usr/bin/env python3
"""Stage RegMix 10B token memmaps for edullm-data and call publish()."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ID = "pretrain/regmix-10b"
DEFAULT_TOKENIZER = "tokenizer/dolma2-bpe"
DEFAULT_PURPOSE = (
    "RegMix-weighted 10B-token OLMo-mix pretraining corpus for 370M ladder runs, "
    "token-selection arms, curriculum, and MixLaw validation"
)
# Hard cap: every train/val shard is at most 1 GiB (last shard may be smaller).
MAX_SHARD_BYTES = 1_073_741_824
DEFAULT_SHARD_BYTES = MAX_SHARD_BYTES
# Same fraction from every source → val mix weights == full-corpus mix weights.
VAL_FRACTION = 0.0015
DEFAULT_TOKENIZED_ROOT = Path(
    "/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/tokenized"
)


def _align_shard_bytes(n: int) -> int:
    n -= n % 4
    if n < 4:
        raise ValueError("shard_bytes must be at least 4")
    return n


def build_manifest_from_tokenized(tokenized_root: Path) -> dict:
    """Build a RefHQ-shaped tokenized manifest from per-domain *.json + on-disk sizes."""
    domains: dict[str, dict] = {}
    total_tokens = 0
    for source_dir in sorted(p for p in tokenized_root.iterdir() if p.is_dir()):
        source = source_dir.name
        npy = source_dir / f"{source}.npy"
        meta_path = source_dir / f"{source}.json"
        if not npy.is_file():
            raise SystemExit(f"missing {npy}")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        actual_bytes = npy.stat().st_size
        if actual_bytes % 4 != 0:
            raise SystemExit(f"{source}: npy size {actual_bytes} not uint32-aligned")
        declared = int(meta.get("bytes") or 0)
        if declared and declared != actual_bytes:
            raise SystemExit(f"{source}: meta bytes {declared} != on-disk {actual_bytes}")
        tokens = int(
            meta.get("tokens_with_eos")
            or meta.get("stream_tokens_with_eos")
            or (actual_bytes // 4)
        )
        domains[source] = {
            **meta,
            "bytes": actual_bytes,
            "stream_tokens_with_eos": tokens,
        }
        total_tokens += tokens
    return {
        "accepted": True,
        "tokenizer_id": "allenai/dolma2-tokenizer",
        "eos_token_id": 100257,
        "total_stream_tokens_with_eos": total_tokens,
        "domains": domains,
    }


def split_npy_to_shards(
    npy_path: Path,
    out_dir: Path,
    *,
    shard_bytes: int,
) -> list[Path]:
    """Write tokens/<source>/train-*.u32le.bin shards, each <= MAX_SHARD_BYTES."""
    if shard_bytes > MAX_SHARD_BYTES:
        raise ValueError(f"shard_bytes {shard_bytes} exceeds max {MAX_SHARD_BYTES}")
    shard_bytes = _align_shard_bytes(shard_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = npy_path.stat().st_size
    if total % 4 != 0:
        raise ValueError(f"{npy_path}: size {total} is not a multiple of 4 (uint32)")

    written: list[Path] = []
    shard_idx = 0
    with npy_path.open("rb") as src:
        while True:
            chunk = src.read(shard_bytes)
            if not chunk:
                break
            if len(chunk) % 4 != 0:
                raise ValueError(f"{npy_path}: trailing partial token in shard {shard_idx}")
            out = out_dir / f"train-{shard_idx:05d}.u32le.bin"
            out.write_bytes(chunk)
            written.append(out)
            shard_idx += 1
    if not written:
        raise ValueError(f"{npy_path}: empty input")
    return written


def _source_train_bytes(source_dir: Path) -> int:
    return sum(p.stat().st_size for p in source_dir.glob("train-*.u32le.bin"))


def carve_val_holdout(out_root: Path, *, fraction: float = VAL_FRACTION) -> dict[str, int]:
    """Carve the same fraction from every source so val mix weights match the full corpus."""
    if not (0.0 < fraction < 0.5):
        raise SystemExit(f"val fraction must be in (0, 0.5); got {fraction}")

    tokens_root = out_root / "tokens"
    sources = sorted(p.name for p in tokens_root.iterdir() if p.is_dir())
    if not sources:
        raise SystemExit(f"no sources under {tokens_root}")

    carved: dict[str, int] = {}
    for source in sources:
        source_dir = tokens_root / source
        train_shards = sorted(source_dir.glob("train-*.u32le.bin"))
        if not train_shards:
            raise SystemExit(f"no train shards under {source_dir} for val carve")
        source_bytes = _source_train_bytes(source_dir)
        if source_bytes % 4 != 0:
            raise SystemExit(f"{source}: train bytes {source_bytes} not uint32-aligned")
        val_bytes = int(source_bytes * fraction)
        val_bytes -= val_bytes % 4
        if val_bytes < 4:
            raise SystemExit(
                f"{source}: val carve too small ({val_bytes} bytes at fraction={fraction})"
            )

        remaining = val_bytes
        chunks: list[bytes] = []
        for shard in reversed(train_shards):
            if remaining <= 0:
                break
            data = shard.read_bytes()
            take = min(remaining, len(data))
            take -= take % 4
            if take <= 0:
                continue
            if take >= len(data):
                chunks.append(data)
                shard.unlink()
                remaining -= len(data)
            else:
                shard.write_bytes(data[:-take])
                chunks.append(data[-take:])
                remaining -= take
        if remaining != 0:
            raise SystemExit(
                f"{source}: could not carve {val_bytes} bytes (short by {remaining})"
            )
        val_path = source_dir / "val-00000.u32le.bin"
        val_path.write_bytes(b"".join(reversed(chunks)))
        carved[source] = val_bytes // 4
        print(
            f"carved val: {val_path.relative_to(out_root)} "
            f"({carved[source]:,} tokens, {fraction:.4%} of source)",
            flush=True,
        )

    total_val = sum(carved.values())
    print(
        f"val carve total: {total_val:,} tokens across {len(carved)} sources "
        f"(fraction={fraction:.4%} per source)",
        flush=True,
    )
    return carved


def stage_publish_layout(
    *,
    tokenized_root: Path,
    manifest: dict,
    out_root: Path,
    shard_bytes: int,
    force: bool,
    val_fraction: float = VAL_FRACTION,
) -> dict[str, list[str]]:
    """Stage as tokens/<source>/… only → entry.labels = {source: …}, domain omitted."""
    if shard_bytes > MAX_SHARD_BYTES:
        raise SystemExit(f"--shard-bytes {shard_bytes} exceeds max {MAX_SHARD_BYTES}")
    if out_root.exists():
        if not force:
            raise SystemExit(f"staging dir exists: {out_root} (pass --force to replace)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    staged: dict[str, list[str]] = {}
    for source in sorted(manifest["domains"]):
        meta = manifest["domains"][source]
        npy_path = tokenized_root / source / f"{source}.npy"
        if not npy_path.is_file():
            raise SystemExit(f"missing tokenized memmap: {npy_path}")
        expected_bytes = int(meta.get("bytes") or 0)
        actual_bytes = npy_path.stat().st_size
        if expected_bytes and expected_bytes != actual_bytes:
            raise SystemExit(
                f"{source}: manifest bytes {expected_bytes} != on-disk {actual_bytes}"
            )
        # One nest level only: tokens/<source>/train-*.u32le.bin (no domain segment).
        rel_dir = Path("tokens") / source
        shards = split_npy_to_shards(
            npy_path,
            out_root / rel_dir,
            shard_bytes=shard_bytes,
        )
        for shard in shards:
            if shard.stat().st_size > MAX_SHARD_BYTES:
                raise SystemExit(f"shard exceeds 1 GiB: {shard}")
        staged[source] = [str(rel_dir / p.name) for p in shards]
        print(
            f"staged {source}: {len(shards)} shard(s), "
            f"{actual_bytes:,} bytes -> {rel_dir}/ (labels.source={source!r})",
            flush=True,
        )
    carve_val_holdout(out_root, fraction=val_fraction)
    return staged


def build_sources(manifest: dict) -> list[dict]:
    total = int(manifest.get("total_stream_tokens_with_eos") or 0)
    sources: list[dict] = []
    for domain in sorted(manifest["domains"]):
        meta = manifest["domains"][domain]
        tokens = int(meta.get("stream_tokens_with_eos") or 0)
        share = (tokens / total) if total else None
        row: dict = {
            "name": domain,
            "tokens": tokens,
            "scope": "measured-in-this-dataset",
        }
        if share is not None:
            row["share"] = round(share, 6)
        sources.append(row)
    return sources


def ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError:
        raise SystemExit(
            "edullm-data is not installed. Clone main and pip install -e it "
            "(see publish_regmix_edullm_data.sbatch)."
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional prebuilt tokenized manifest; else built from tokenized-root",
    )
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--hash-workers", type=int, default=16)
    parser.add_argument("--copy-workers", type=int, default=16)
    parser.add_argument(
        "--text-run-dir",
        type=Path,
        default=None,
        help="run dir with trim/<source>/*-trimmed.json.gz (default: parent of tokenized-root)",
    )
    parser.add_argument("--skip-text-stage", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _datasets_root = Path(__file__).resolve().parents[1]
    if str(_datasets_root) not in sys.path:
        sys.path.insert(0, str(_datasets_root))
    from edullm_text_companion import PUBLISH_PROFILE, TEXT_GROUP_META, stage_text_companion

    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest_from_tokenized(args.tokenized_root)
        print(
            f"built manifest: {len(manifest['domains'])} sources, "
            f"{manifest['total_stream_tokens_with_eos']:,} tokens",
            flush=True,
        )
    if not manifest.get("accepted", True):
        raise SystemExit("tokenized manifest is not accepted")

    if not args.skip_stage:
        stage_publish_layout(
            tokenized_root=args.tokenized_root,
            manifest=manifest,
            out_root=args.stage_dir,
            shard_bytes=args.shard_bytes,
            force=args.force,
            val_fraction=args.val_fraction,
        )
    elif not args.stage_dir.is_dir():
        raise SystemExit(f"--skip-stage but stage-dir missing: {args.stage_dir}")

    text_run_dir = args.text_run_dir or args.tokenized_root.parent
    if not args.skip_text_stage:
        stage_text_companion(
            sources=sorted(manifest["domains"]),
            run_dir=text_run_dir,
            out_root=args.stage_dir,
            shard_bytes=args.shard_bytes,
        )

    if args.dry_run:
        print(f"dry-run: staged under {args.stage_dir}", flush=True)
        return 0

    ensure_edullm_data()
    from edullm_data.contracts import validate_dataset_id
    from edullm_data.publish import publish
    from edullm_data.s3 import Boto3S3

    try:
        validate_dataset_id(args.dataset_id)
    except Exception as exc:
        raise SystemExit(f"invalid dataset_id {args.dataset_id!r}: {exc}") from exc

    about = (
        "RegMix-weighted 10B-token dolma2 corpus sampled from OLMo-mix-1124-30b across seven "
        "sources (dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki). "
        "Shards are nested tokens/<source>/ so each mix source is carried in the object key "
        "as entry.labels.source. Per-source token counts are measured from the published objects."
    )
    notes = (
        f"Validation split: {args.val_fraction:.4%} of each source carved into "
        f"tokens/<source>/val-00000.u32le.bin so val source weights match the full mix "
        f"(~{args.val_fraction:.2%} of corpus). Companion raw documents under text/<source>/ "
        f"(text-corpus/v1) retain their complete selected document stream. Legacy path on edullm-datasets: "
        "regmix/regmix-10b/tokenized/*.npy headerless uint32 memmaps."
    )

    created_at = datetime.now(timezone.utc).isoformat()
    plan = publish(
        args.stage_dir,
        dataset_id=args.dataset_id,
        purpose=args.purpose,
        profile=PUBLISH_PROFILE,
        tokenizer=args.tokenizer,
        group_meta=TEXT_GROUP_META,
        s3=Boto3S3.default(),
        created_at=created_at,
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
                "source_kind": plan.source_kind,
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        f"published to s3://edullm-landing/{plan.dataset_id}/{plan.version}/ "
        f"(validator will promote to edullm-data)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
