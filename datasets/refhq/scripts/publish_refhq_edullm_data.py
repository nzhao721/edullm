#!/usr/bin/env python3
"""Stage RefHQ RegMix 5.5B token memmaps for edullm-data and call publish()."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ID = "pretrain/refhq-regmix-5p5b"
DEFAULT_TOKENIZER = "tokenizer/dolma2-bpe"
DEFAULT_PURPOSE = (
    "HQ-filtered RegMix-weighted 5.5B-token pretraining corpus for 370M reference runs, "
    "BLADE/RHO validation, and MixLaw probes"
)
DEFAULT_SHARD_BYTES = 1_073_741_824  # 1 GiB, uint32-aligned
# Same fraction from every source so val mix weights match the full corpus
# (~0.15%, matching olmo-150b-dolma2's held-out fraction).
VAL_FRACTION = 0.0015


def _align_shard_bytes(n: int) -> int:
    n -= n % 4
    if n < 4:
        raise ValueError("shard_bytes must be at least 4")
    return n


def split_npy_to_shards(
    npy_path: Path,
    out_dir: Path,
    *,
    shard_bytes: int,
) -> list[Path]:
    """Split a headerless uint32 memmap (.npy suffix) into train-*.u32le.bin shards."""
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
    """Carve the same fraction of tokens from every source into val-00000.u32le.bin.

    Layout is one nest level: tokens/<source>/… → labels={"source": <source>} only.
    A uniform fraction keeps val source weights identical to the full corpus.
    """
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
    """Build tokens/<source>/train-*.u32le.bin under out_root (source label only)."""
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
        rel_dir = Path("tokens") / source
        shards = split_npy_to_shards(
            npy_path,
            out_root / rel_dir,
            shard_bytes=shard_bytes,
        )
        staged[source] = [str(rel_dir / p.name) for p in shards]
        print(
            f"staged {source}: {len(shards)} shard(s), "
            f"{actual_bytes:,} bytes -> {rel_dir}/",
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
        filt = meta.get("filter")
        if filt:
            row["note"] = str(filt)
        sources.append(row)
    return sources


def ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError:
        raise SystemExit(
            "edullm-data is not installed. Run:\n"
            '  pip install "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"'
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        default=Path("/scratch/users/nzhao2/refhq-regmix-5p5b-v1/tokenized"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/scratch/users/nzhao2/refhq-regmix-5p5b-v1/manifests/tokenized_manifest.json"),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        help="writable dir for tokens/<source>/train-*.u32le.bin layout",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help="fraction of each source carved into val (same fraction → matching mix weights)",
    )
    parser.add_argument("--hash-workers", type=int, default=16)
    parser.add_argument("--copy-workers", type=int, default=16)
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="stage only; do not publish")
    parser.add_argument("--force", action="store_true", help="replace existing stage-dir")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("accepted"):
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
        "HQ-filtered RegMix-weighted 5.5B-token dolma2 corpus: seven independent source pulls "
        "(dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki) tokenized with "
        "allenai/dolma2-tokenizer. Shards are nested tokens/<source>/ so each mix source is "
        "carried in the object key as entry.labels.source (no domain level). Per-source token "
        "counts are measured from the published objects."
    )
    notes = (
        f"Validation split: {args.val_fraction:.4%} of each source carved into "
        f"tokens/<source>/val-00000.u32le.bin so val source weights match the full mix "
        f"(~{args.val_fraction:.2%} of corpus, olmo-150b-scale). Legacy path on edullm-datasets: "
        "refhq/refhq-regmix-5p5b-v1/tokenized/*.npy headerless uint32 memmaps."
    )

    created_at = datetime.now(timezone.utc).isoformat()
    plan = publish(
        args.stage_dir,
        dataset_id=args.dataset_id,
        purpose=args.purpose,
        profile="pretrain-tokens/v1",
        tokenizer=args.tokenizer,
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
