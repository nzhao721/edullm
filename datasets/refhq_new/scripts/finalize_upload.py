#!/usr/bin/env python3
"""Verify dolma2 tokenized memmaps and upload working store to edullm-datasets.

Assumptions (sibling tokenize_source.py / plan_refhq_new.py contract):
  tokenized/<source>/<domain>/{train,val}.npy   headerless uint32 memmaps
  tokenized/<source>/<domain>/{train,val}.json  meta with stream_tokens_with_eos

Also accepts tok/ as an alias, and already-sharded
  tokenized/<source>/<domain>/{train,val}-NNNNN.u32le.bin (uploaded as-is under tokens/).

Working store: s3://edullm-datasets/refhq/refhq-new/
Holdout is document-level before tokenize — train/val npy already exist; no carve here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BUCKET = "edullm-datasets"
DEFAULT_PREFIX = "refhq/refhq-new"
DEFAULT_SHARD_BYTES = 1_073_741_824
SPLITS = ("train", "val")


def _aws_s3_sync(local: Path, uri: str) -> None:
    cmd = ["aws", "s3", "sync", str(local), uri, "--only-show-errors"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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
    split: str,
) -> list[Path]:
    """Split a headerless uint32 memmap into {split}-NNNNN.u32le.bin shards."""
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
            out = out_dir / f"{split}-{shard_idx:05d}.u32le.bin"
            out.write_bytes(chunk)
            written.append(out)
            shard_idx += 1
    if not written:
        raise ValueError(f"{npy_path}: empty input")
    return written


def resolve_tok_root(scratch_root: Path) -> Path:
    for name in ("tokenized", "tok"):
        cand = scratch_root / name
        if cand.is_dir():
            return cand
    raise SystemExit(f"missing tokenized/ or tok/ under {scratch_root}")

def iter_source_domain_dirs(tok_root: Path) -> list[tuple[str, str, Path]]:
    pairs: list[tuple[str, str, Path]] = []
    for source_dir in sorted(p for p in tok_root.iterdir() if p.is_dir()):
        source = source_dir.name
        for domain_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
            pairs.append((source, domain_dir.name, domain_dir))
    return pairs


def collect_pair_report(domain_dir: Path) -> dict:
    """Collect npy and/or u32le shard stats for one (source, domain)."""
    splits: dict[str, dict] = {}
    for split in SPLITS:
        npy = domain_dir / f"{split}.npy"
        meta_path = domain_dir / f"{split}.json"
        shards = sorted(domain_dir.glob(f"{split}-*.u32le.bin"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if npy.is_file():
            nbytes = npy.stat().st_size
            if nbytes % 4 != 0:
                raise SystemExit(f"{npy}: size {nbytes} not uint32-aligned")
            tokens = int(
                meta.get("stream_tokens_with_eos")
                or meta.get("tokens_with_eos")
                or (nbytes // 4)
            )
            splits[split] = {
                **meta,
                "npy": str(npy),
                "bytes": nbytes,
                "stream_tokens_with_eos": tokens,
                "kind": "npy",
            }
        elif shards:
            nbytes = sum(p.stat().st_size for p in shards)
            if nbytes % 4 != 0:
                raise SystemExit(f"{domain_dir}/{split}-*: bytes {nbytes} not uint32-aligned")
            tokens = int(
                meta.get("stream_tokens_with_eos")
                or meta.get("tokens_with_eos")
                or (nbytes // 4)
            )
            splits[split] = {
                **meta,
                "shards": [p.name for p in shards],
                "bytes": nbytes,
                "stream_tokens_with_eos": tokens,
                "kind": "u32le",
            }
    return splits


def stage_tokens_from_tok(
    *,
    tok_root: Path,
    out_root: Path,
    shard_bytes: int,
    force: bool,
) -> dict[str, list[str]]:
    """Build tokens/<source>/<domain>/{train,val}-*.u32le.bin (no val carve)."""
    if out_root.exists():
        if not force:
            raise SystemExit(f"staging dir exists: {out_root} (pass --force to replace)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    staged: dict[str, list[str]] = {}
    for source, domain, domain_dir in iter_source_domain_dirs(tok_root):
        rel_dir = Path("tokens") / source / domain
        dest = out_root / rel_dir
        dest.mkdir(parents=True, exist_ok=True)
        key = f"{source}/{domain}"
        staged[key] = []
        report = collect_pair_report(domain_dir)
        if not report:
            raise SystemExit(f"no train/val outputs under {domain_dir}")
        for split, meta in sorted(report.items()):
            if meta["kind"] == "u32le":
                for name in meta["shards"]:
                    src = domain_dir / name
                    dst = dest / name
                    shutil.copy2(src, dst)
                    staged[key].append(str(rel_dir / name))
            else:
                shards = split_npy_to_shards(
                    Path(meta["npy"]),
                    dest,
                    shard_bytes=shard_bytes,
                    split=split,
                )
                staged[key].extend(str(rel_dir / p.name) for p in shards)
            print(
                f"staged {key} {split}: {meta['stream_tokens_with_eos']:,} tokens -> {rel_dir}/",
                flush=True,
            )
        if "train" not in report:
            raise SystemExit(f"{key}: missing train split")
    return staged


def build_manifest(*, scratch_root: Path, tok_root: Path, bucket: str, prefix: str) -> dict:
    reports: dict[str, dict] = {}
    failures: list[str] = []
    total_stream_tokens = 0

    pairs = iter_source_domain_dirs(tok_root)
    if not pairs:
        failures.append(f"no source/domain dirs under {tok_root}")

    for source, domain, domain_dir in pairs:
        key = f"{source}/{domain}"
        try:
            splits = collect_pair_report(domain_dir)
        except SystemExit as exc:
            failures.append(str(exc))
            continue
        if "train" not in splits:
            failures.append(f"{key}: missing train.npy or train-*.u32le.bin")
            continue
        # val may be empty for tiny smoke runs; warn but accept if present or absent.
        pair_tokens = sum(int(s["stream_tokens_with_eos"]) for s in splits.values())
        reports[key] = {
            "source": source,
            "domain": domain,
            "splits": splits,
            "stream_tokens_with_eos": pair_tokens,
            "bytes": sum(int(s["bytes"]) for s in splits.values()),
        }
        total_stream_tokens += pair_tokens

    # Aggregate per HF source for publish() sources[].
    by_source: dict[str, int] = {}
    for key, meta in reports.items():
        src = meta["source"]
        by_source[src] = by_source.get(src, 0) + int(meta["stream_tokens_with_eos"])

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_id": "allenai/dolma2-tokenizer",
        "eos_token_id": 100257,
        "s3_bucket": bucket,
        "s3_prefix": prefix,
        "tok_root": str(tok_root),
        "total_stream_tokens_with_eos": total_stream_tokens,
        "by_source": by_source,
        "pairs": reports,
        "failures": failures,
        "accepted": not failures,
        "notes": (
            "Doc-level 0.15% holdout was applied before tokenize; "
            "train/val memmaps are independent (no token carve at finalize)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="if set, also write tokens/<source>/<domain>/*.u32le.bin and sync to S3",
    )
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bucket = args.bucket or plan.get("s3_bucket") or DEFAULT_BUCKET
    prefix = args.prefix or plan.get("s3_prefix") or DEFAULT_PREFIX
    scratch_root = Path(plan.get("scratch_root") or plan.get("run_dir") or args.plan.parent.parent)
    tok_root = resolve_tok_root(scratch_root)
    manifests_dir = scratch_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        scratch_root=scratch_root,
        tok_root=tok_root,
        bucket=bucket,
        prefix=prefix,
    )
    manifest_path = manifests_dir / "tokenized_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if not manifest["accepted"]:
        print(f"ACCEPTANCE FAILED: {len(manifest['failures'])} error(s)", flush=True)
        for line in manifest["failures"]:
            print(f"  - {line}", flush=True)
        return 1

    stage_dir = args.stage_dir
    if stage_dir is not None:
        stage_tokens_from_tok(
            tok_root=tok_root,
            out_root=stage_dir,
            shard_bytes=args.shard_bytes,
            force=args.force,
        )

    if args.dry_run or args.skip_upload:
        print("skipping S3 upload", flush=True)
        return 0

    # Upload tokenized/ memmaps (RefHQ-style working store) + manifests + staged shards.
    _aws_s3_sync(tok_root, f"s3://{bucket}/{prefix}/tokenized/")
    _aws_s3_sync(manifests_dir, f"s3://{bucket}/{prefix}/manifests/")
    if stage_dir is not None and (stage_dir / "tokens").is_dir():
        _aws_s3_sync(stage_dir / "tokens", f"s3://{bucket}/{prefix}/tokens/")
    print(f"upload complete -> s3://{bucket}/{prefix}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
