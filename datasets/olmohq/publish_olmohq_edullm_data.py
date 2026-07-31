#!/usr/bin/env python3
"""Stage olmohq (~127B) token shards for edullm-data and call publish().

Source: s3://edullm-datasets/olmo100b/olmo-mix-1124-30b/ (active tokenized_manifest).
Layout: tokens/<source>/train-*.u32le.bin  → labels={source} only (domain omitted).
Shards: max 1 GiB. Val: same fraction from every source (mix weights match).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ID = "pretrain/olmo-127b"
DEFAULT_TOKENIZER = "tokenizer/dolma2-bpe"
DEFAULT_PURPOSE = (
    "Default reservoir corpus mirroring olmo-mix-1124 (~127B dolma2 tokens) "
    "for sampling training mixes, curricula, and ladder runs"
)
MAX_SHARD_BYTES = 1_073_741_824
DEFAULT_SHARD_BYTES = MAX_SHARD_BYTES
VAL_FRACTION = 0.0015
DEFAULT_BUCKET = "edullm-datasets"
DEFAULT_PREFIX = "olmo100b/olmo-mix-1124-30b"
DEFAULT_MANIFEST_KEY = f"{DEFAULT_PREFIX}/plan/tokenized_manifest.json"
# Reload AWS_SESSION_ENV this often so a login-node refresher can rotate STS keys.
SESSION_RELOAD_SECONDS = 5 * 60


def _align_shard_bytes(n: int) -> int:
    n -= n % 4
    if n < 4:
        raise ValueError("shard_bytes must be at least 4")
    return n


def load_shard_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("accepted", True):
        raise SystemExit("tokenized manifest is not accepted")
    shards = manifest.get("shards") or []
    if not shards:
        raise SystemExit(f"no shards in {path}")
    by_source: dict[str, list[dict]] = defaultdict(list)
    totals: dict[str, dict] = {}
    for row in shards:
        source = str(row.get("domain") or row.get("source") or "").strip()
        if not source:
            raise SystemExit(f"shard missing domain/source: {row!r}")
        rel = str(row.get("path") or "").lstrip("/")
        if not rel:
            raise SystemExit(f"shard missing path: {row!r}")
        # Manifest paths are like shards/00000__….npy relative to tokenized/.
        if not rel.startswith("shards/"):
            rel = f"shards/{rel}" if not rel.startswith("tokenized/") else rel
        if rel.startswith("tokenized/"):
            rel = rel[len("tokenized/") :]
        bytes_ = int(row.get("bytes") or 0)
        tokens = int(row.get("tokens") or row.get("tokens_with_eos") or (bytes_ // 4))
        if bytes_ and bytes_ % 4 != 0:
            raise SystemExit(f"{rel}: bytes {bytes_} not uint32-aligned")
        by_source[source].append({"path": rel, "bytes": bytes_, "tokens": tokens})
    for source, rows in by_source.items():
        rows.sort(key=lambda r: r["path"])
        totals[source] = {
            "bytes": sum(r["bytes"] for r in rows),
            "stream_tokens_with_eos": sum(r["tokens"] for r in rows),
            "shard_count": len(rows),
            "shards": rows,
        }
    total_tokens = sum(v["stream_tokens_with_eos"] for v in totals.values())
    declared = int(manifest.get("total_content_tokens") or 0)
    if declared and declared != total_tokens:
        print(
            f"warning: manifest total_content_tokens={declared:,} != "
            f"sum(shards)={total_tokens:,}; using sum(shards)",
            flush=True,
        )
    return {
        "accepted": True,
        "tokenizer_id": manifest.get("tokenizer_id") or "allenai/dolma2-tokenizer",
        "eos_token_id": int(manifest.get("eos_token_id") or 100257),
        "total_stream_tokens_with_eos": total_tokens,
        "domains": totals,
        "raw": manifest,
    }


class ShardWriter:
    """Append raw uint32 bytes into train-*.u32le.bin files capped at shard_bytes."""

    def __init__(self, out_dir: Path, *, shard_bytes: int, resume: bool = False) -> None:
        if shard_bytes > MAX_SHARD_BYTES:
            raise ValueError(f"shard_bytes {shard_bytes} exceeds max {MAX_SHARD_BYTES}")
        self.shard_bytes = _align_shard_bytes(shard_bytes)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self.current_path: Path | None = None
        self.current_fh = None
        self.current_size = 0
        self.written: list[Path] = []
        self.total_bytes = 0
        if resume:
            existing = sorted(out_dir.glob("train-*.u32le.bin"))
            self.written = list(existing)
            self.total_bytes = sum(p.stat().st_size for p in existing)
            if existing:
                last = existing[-1]
                last_size = last.stat().st_size
                if 0 < last_size < self.shard_bytes and last_size % 4 == 0:
                    self.current_path = last
                    self.current_fh = last.open("ab")
                    self.current_size = last_size
                    self.shard_idx = len(existing)  # next NEW shard index
                else:
                    self.shard_idx = len(existing)

    def _open_next(self) -> None:
        if self.current_fh is not None:
            self.current_fh.close()
        self.current_path = self.out_dir / f"train-{self.shard_idx:05d}.u32le.bin"
        self.current_fh = self.current_path.open("wb")
        self.current_size = 0
        self.written.append(self.current_path)
        self.shard_idx += 1

    def write(self, data: bytes) -> None:
        if len(data) % 4 != 0:
            raise ValueError("chunk length not uint32-aligned")
        offset = 0
        while offset < len(data):
            if self.current_fh is None or self.current_size >= self.shard_bytes:
                self._open_next()
            room = self.shard_bytes - self.current_size
            take = min(room, len(data) - offset)
            take -= take % 4
            if take <= 0:
                self._open_next()
                continue
            chunk = data[offset : offset + take]
            assert self.current_fh is not None
            self.current_fh.write(chunk)
            self.current_size += len(chunk)
            self.total_bytes += len(chunk)
            offset += take

    def close(self) -> list[Path]:
        if self.current_fh is not None:
            self.current_fh.close()
            self.current_fh = None
        if not self.written:
            raise ValueError(f"no shards written under {self.out_dir}")
        for path in self.written:
            size = path.stat().st_size
            if size > MAX_SHARD_BYTES:
                raise SystemExit(f"shard exceeds 1 GiB: {path} ({size})")
            if size % 4 != 0:
                raise SystemExit(f"shard not uint32-aligned: {path}")
        return self.written


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
        # Val may exceed 1 GiB for large sources (~0.15% of ~30B ≈ 45M tokens ≈ 180 MiB),
        # but enforce the same max if needed by splitting — for 0.15% of 127B total val
        # is ~190M tokens ≈ 760 MiB, per-source max is dclm ≈ 45M tokens ≈ 178 MiB.
        val_path = source_dir / "val-00000.u32le.bin"
        blob = b"".join(reversed(chunks))
        if len(blob) > MAX_SHARD_BYTES:
            raise SystemExit(f"{source}: val shard exceeds 1 GiB ({len(blob)})")
        val_path.write_bytes(blob)
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


def resolve_local_shard(tokenized_root: Path, rel: str) -> Path:
    path = tokenized_root / rel
    if path.is_file():
        return path
    alt = tokenized_root / Path(rel).name
    if alt.is_file():
        return alt
    raise FileNotFoundError(path)


def _apply_session_env_file(path: Path) -> bool:
    """Load export KEY=VAL lines from a FarmShare aws-session.env into os.environ."""
    if not path.is_file():
        return False
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("unset "):
            for name in line.split()[1:]:
                os.environ.pop(name, None)
            continue
        if not line.startswith("export "):
            continue
        assign = line[len("export ") :]
        if "=" not in assign:
            continue
        key, val = assign.split("=", 1)
        os.environ[key] = shlex.split(val)[0] if val else ""
        loaded += 1
    return loaded > 0


class RefreshingS3:
    """boto3 S3 client that reloads AWS_SESSION_ENV periodically (login-node refresher)."""

    def __init__(self, session_env: Path | None = None) -> None:
        import boto3

        self._boto3 = boto3
        self.session_env = session_env or Path(os.environ.get("AWS_SESSION_ENV") or "")
        self._client = None
        self._loaded_at = 0.0
        self._mtime = 0.0
        self._fingerprint = ""
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> None:
        now = time.time()
        mtime = 0.0
        if self.session_env and self.session_env.is_file():
            mtime = self.session_env.stat().st_mtime
        stale = (now - self._loaded_at) >= SESSION_RELOAD_SECONDS
        changed = mtime > self._mtime
        if not force and self._client is not None and not stale and not changed:
            return
        if self.session_env and self.session_env.is_file():
            _apply_session_env_file(self.session_env)
            print(f"reloaded AWS session from {self.session_env}", flush=True)
        key = os.environ.get("AWS_ACCESS_KEY_ID") or ""
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
        token = os.environ.get("AWS_SESSION_TOKEN") or ""
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
        if not (key and secret and token):
            raise RuntimeError("AWS session env missing access key / secret / token")
        fingerprint = f"{key[-4:]}:{token[-8:]}:{mtime}"
        self._client = self._boto3.client(
            "s3",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            aws_session_token=token,
            region_name=region,
        )
        if fingerprint != self._fingerprint:
            print(f"S3 client credentials rotated (...{key[-4:]})", flush=True)
            self._fingerprint = fingerprint
        self._loaded_at = now
        self._mtime = mtime

    @property
    def client(self):
        self.refresh()
        return self._client


def _open_shard_stream(
    *,
    rel: str,
    tokenized_root: Path | None,
    s3_bucket: str | None,
    s3_tokenized_prefix: str | None,
    s3: RefreshingS3 | None,
):
    """Open a local file or an S3 StreamingBody for one input .npy shard."""
    if tokenized_root is not None:
        path = resolve_local_shard(tokenized_root, rel)
        return path.open("rb"), path.stat().st_size, str(path)
    assert s3_bucket and s3_tokenized_prefix and s3 is not None
    key = f"{s3_tokenized_prefix.rstrip('/')}/{rel.lstrip('/')}"
    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            s3.refresh(force=(attempt > 1))
            # Prefer GET only — avoids some HeadObject 400s with stale/odd signing.
            obj = s3.client.get_object(Bucket=s3_bucket, Key=key)
            size = int(obj["ContentLength"])
            return obj["Body"], size, f"s3://{s3_bucket}/{key}"
        except Exception as exc:  # noqa: BLE001 — retry after credential refresh
            last_exc = exc
            print(f"S3 open failed attempt {attempt}/5 for {key}: {exc}", flush=True)
            time.sleep(min(30, 3 * attempt))
            # Wait for login-node refresher to rewrite aws-session.env
            if s3.session_env:
                deadline = time.time() + 90
                base_mtime = s3.session_env.stat().st_mtime if s3.session_env.is_file() else 0
                while time.time() < deadline:
                    if s3.session_env.is_file() and s3.session_env.stat().st_mtime > base_mtime:
                        break
                    time.sleep(5)
    assert last_exc is not None
    raise last_exc


def stream_source_to_shards(
    *,
    source: str,
    shard_rows: list[dict],
    out_dir: Path,
    shard_bytes: int,
    tokenized_root: Path | None = None,
    s3_bucket: str | None = None,
    s3_tokenized_prefix: str | None = None,
    s3: RefreshingS3 | None = None,
) -> list[Path]:
    progress_path = out_dir / "_ingest_progress.json"
    completed: set[str] = set()
    resume = False
    if progress_path.is_file() and out_dir.is_dir():
        try:
            completed = set(
                json.loads(progress_path.read_text(encoding="utf-8")).get("completed") or []
            )
            resume = bool(completed)
        except json.JSONDecodeError:
            completed = set()
            resume = False
    elif out_dir.is_dir() and any(out_dir.glob("train-*.u32le.bin")):
        # Crash without progress file: keep whole input shards already covered by
        # existing train bytes; truncate any partial trailing input.
        existing_bytes = sum(p.stat().st_size for p in out_dir.glob("train-*.u32le.bin"))
        kept = 0
        for row in shard_rows:
            nxt = kept + int(row["bytes"])
            if nxt <= existing_bytes:
                completed.add(row["path"])
                kept = nxt
            else:
                break
        if completed and kept > 0:
            resume = True
            _truncate_train_dir_to_bytes(out_dir, kept)
            progress_path.write_text(
                json.dumps({"completed": sorted(completed), "bytes_written": kept}, indent=2),
                encoding="utf-8",
            )
            print(
                f"  bootstrapped mid-source resume {source}: "
                f"{len(completed)}/{len(shard_rows)} inputs (~{kept:,} bytes)",
                flush=True,
            )
    if out_dir.exists() and not resume:
        shutil.rmtree(out_dir)
    writer = ShardWriter(out_dir, shard_bytes=shard_bytes, resume=resume)
    expected = sum(int(r["bytes"]) for r in shard_rows)
    bufsize = 64 * 1024 * 1024
    if resume and completed:
        print(
            f"  mid-source resume {source}: {len(completed)}/{len(shard_rows)} input shards done, "
            f"{writer.total_bytes:,} bytes already staged",
            flush=True,
        )
    for row in shard_rows:
        rel = row["path"]
        if rel in completed:
            continue
        src, actual, label = _open_shard_stream(
            rel=rel,
            tokenized_root=tokenized_root,
            s3_bucket=s3_bucket,
            s3_tokenized_prefix=s3_tokenized_prefix,
            s3=s3,
        )
        declared = int(row["bytes"] or 0)
        if declared and declared != actual:
            raise SystemExit(f"{label}: manifest bytes {declared} != size {actual}")
        if actual % 4 != 0:
            raise SystemExit(f"{label}: size {actual} not uint32-aligned")
        try:
            while True:
                chunk = src.read(bufsize)
                if not chunk:
                    break
                writer.write(chunk)
        finally:
            close = getattr(src, "close", None)
            if callable(close):
                close()
        completed.add(rel)
        progress_path.write_text(
            json.dumps(
                {"completed": sorted(completed), "bytes_written": writer.total_bytes},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  ingested {label} ({actual:,} bytes)", flush=True)
    written = writer.close()
    if writer.total_bytes != expected:
        raise SystemExit(
            f"{source}: staged {writer.total_bytes:,} bytes != expected {expected:,}"
        )
    if progress_path.exists():
        progress_path.unlink()
    return written


def _truncate_train_dir_to_bytes(out_dir: Path, keep_bytes: int) -> None:
    """Keep a prefix of train-*.u32le.bin totaling exactly keep_bytes."""
    keep_bytes -= keep_bytes % 4
    remaining = keep_bytes
    for path in sorted(out_dir.glob("train-*.u32le.bin")):
        if remaining <= 0:
            path.unlink()
            continue
        size = path.stat().st_size
        if size <= remaining:
            remaining -= size
            continue
        data = path.read_bytes()[:remaining]
        path.write_bytes(data)
        remaining = 0
    if remaining != 0:
        raise SystemExit(f"{out_dir}: could not truncate to {keep_bytes} bytes")


def _source_complete(source_dir: Path, expected_bytes: int) -> bool:
    if not source_dir.is_dir():
        return False
    if any(source_dir.glob("val-*.u32le.bin")):
        # val already carved — treat as done only if train+val == expected
        total = sum(p.stat().st_size for p in source_dir.glob("*.u32le.bin"))
        return total == expected_bytes
    train = sum(p.stat().st_size for p in source_dir.glob("train-*.u32le.bin"))
    return train == expected_bytes


def stage_publish_layout(
    *,
    manifest: dict,
    out_root: Path,
    shard_bytes: int,
    force: bool,
    resume: bool = False,
    val_fraction: float = VAL_FRACTION,
    tokenized_root: Path | None = None,
    s3_bucket: str | None = None,
    s3_tokenized_prefix: str | None = None,
) -> dict[str, list[str]]:
    """Stage as tokens/<source>/… only → entry.labels = {source: …}, domain omitted."""
    if shard_bytes > MAX_SHARD_BYTES:
        raise SystemExit(f"--shard-bytes {shard_bytes} exceeds max {MAX_SHARD_BYTES}")
    if tokenized_root is None and not (s3_bucket and s3_tokenized_prefix):
        raise SystemExit("need --tokenized-root or --s3-bucket/--s3-tokenized-prefix")
    if out_root.exists():
        if force and not resume:
            shutil.rmtree(out_root)
        elif not resume and not force:
            raise SystemExit(
                f"staging dir exists: {out_root} (pass --force to replace or --resume)"
            )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "tokens").mkdir(parents=True, exist_ok=True)

    s3 = None
    if tokenized_root is None:
        s3 = RefreshingS3()

    staged: dict[str, list[str]] = {}
    for source in sorted(manifest["domains"]):
        meta = manifest["domains"][source]
        rel_dir = Path("tokens") / source
        out_dir = out_root / rel_dir
        if resume and _source_complete(out_dir, int(meta["bytes"])):
            shards = sorted(out_dir.glob("train-*.u32le.bin"))
            staged[source] = [str(rel_dir / p.name) for p in shards]
            print(
                f"resume skip {source}: already complete "
                f"({meta['bytes']:,} bytes, {len(shards)} train shard(s))",
                flush=True,
            )
            continue
        print(
            f"staging {source}: {meta['shard_count']} input shard(s), "
            f"{meta['bytes']:,} bytes -> {rel_dir}/ (labels.source={source!r})",
            flush=True,
        )
        shards = stream_source_to_shards(
            source=source,
            shard_rows=meta["shards"],
            out_dir=out_dir,
            shard_bytes=shard_bytes,
            tokenized_root=tokenized_root,
            s3_bucket=s3_bucket,
            s3_tokenized_prefix=s3_tokenized_prefix,
            s3=s3,
        )
        staged[source] = [str(rel_dir / p.name) for p in shards]
        print(
            f"staged {source}: {len(shards)} output shard(s) <= 1 GiB",
            flush=True,
        )
    # Only carve val when missing (resume-safe).
    tokens_root = out_root / "tokens"
    need_val = False
    for source_dir in tokens_root.iterdir():
        if source_dir.is_dir() and not any(source_dir.glob("val-*.u32le.bin")):
            need_val = True
            break
    if need_val:
        # Drop any partial val and carve fresh from train tails.
        for source_dir in tokens_root.iterdir():
            if not source_dir.is_dir():
                continue
            for vp in source_dir.glob("val-*.u32le.bin"):
                vp.unlink()
        carve_val_holdout(out_root, fraction=val_fraction)
    else:
        print("resume: all sources already have val shards; skipping carve", flush=True)
    return staged


def build_sources(manifest: dict) -> list[dict]:
    total = int(manifest.get("total_stream_tokens_with_eos") or 0)
    sources: list[dict] = []
    for name in sorted(manifest["domains"]):
        meta = manifest["domains"][name]
        tokens = int(meta.get("stream_tokens_with_eos") or 0)
        share = (tokens / total) if total else None
        row: dict = {
            "name": name,
            "tokens": tokens,
            "scope": "measured-in-this-dataset",
            "uri": f"s3://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}/tokenized/",
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
            "(see publish_olmohq_edullm_data.sbatch)."
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        default=None,
        help="optional local mirror of …/tokenized/ (contains shards/*.npy)",
    )
    parser.add_argument(
        "--s3-bucket",
        default=DEFAULT_BUCKET,
        help="read tokenized shards from this bucket when --tokenized-root is omitted",
    )
    parser.add_argument(
        "--s3-tokenized-prefix",
        default=f"{DEFAULT_PREFIX}/tokenized",
        help="S3 prefix containing shards/ (FarmShare streams from here)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="active plan/tokenized_manifest.json (post top-up trim)",
    )
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument(
        "--about",
        default=None,
        help="README about paragraph; default is derived from measured token count",
    )
    parser.add_argument(
        "--legacy-uri",
        default=None,
        help="notes provenance URI (defaults to s3://<bucket>/<prefix>/)",
    )
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--hash-workers", type=int, default=16)
    parser.add_argument("--copy-workers", type=int, default=16)
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep existing stage-dir; skip complete sources; restage incomplete ones",
    )
    args = parser.parse_args()

    manifest = load_shard_manifest(args.manifest)
    print(
        f"loaded manifest: {len(manifest['domains'])} sources, "
        f"{manifest['total_stream_tokens_with_eos']:,} tokens, "
        f"{sum(m['shard_count'] for m in manifest['domains'].values())} input shards",
        flush=True,
    )

    if not args.skip_stage:
        stage_publish_layout(
            manifest=manifest,
            out_root=args.stage_dir,
            shard_bytes=args.shard_bytes,
            force=args.force,
            resume=args.resume,
            val_fraction=args.val_fraction,
            tokenized_root=args.tokenized_root,
            s3_bucket=None if args.tokenized_root else args.s3_bucket,
            s3_tokenized_prefix=None if args.tokenized_root else args.s3_tokenized_prefix,
        )
    elif not args.stage_dir.is_dir():
        raise SystemExit(f"--skip-stage but stage-dir missing: {args.stage_dir}")

    if args.dry_run:
        print(f"dry-run: staged under {args.stage_dir}", flush=True)
        return 0

    # Reload laptop-pushed aws-session.env before landing upload (staging may have
    # finished under an older STS window; hash/upload can outlast remaining TTL).
    session_env = Path(os.environ.get("AWS_SESSION_ENV", "") or "")
    if session_env.is_file() and _apply_session_env_file(session_env):
        print(f"reloaded AWS session before publish from {session_env}", flush=True)

    ensure_edullm_data()
    from edullm_data.contracts import validate_dataset_id
    from edullm_data.publish import publish
    from edullm_data.s3 import Boto3S3

    try:
        validate_dataset_id(args.dataset_id)
    except Exception as exc:
        raise SystemExit(f"invalid dataset_id {args.dataset_id!r}: {exc}") from exc

    about = args.about or (
        "Corpus mirroring allenai/olmo-mix-1124 domain mix, dolma2-tokenized to "
        f"~{manifest['total_stream_tokens_with_eos'] / 1e9:.1f}B tokens across seven sources "
        "(dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki). "
        "Shards nest as tokens/<source>/ so each mix source is carried as entry.labels.source "
        "(domain omitted). Per-source counts are measured from the published objects."
    )
    legacy = args.legacy_uri or (
        f"s3://{args.s3_bucket}/{args.s3_tokenized_prefix.rstrip('/').removesuffix('/tokenized')}/"
        if args.s3_tokenized_prefix
        else f"s3://{DEFAULT_BUCKET}/{DEFAULT_PREFIX}/"
    )
    notes = (
        f"Validation split: {args.val_fraction:.4%} of each source carved into "
        f"tokens/<source>/val-00000.u32le.bin so val source weights match the full mix. "
        f"Legacy path: {legacy}"
    )

    # Stamp source URIs from the actual S3 prefix used for this publish.
    sources = build_sources(manifest)
    tok_uri = (
        f"s3://{args.s3_bucket}/{args.s3_tokenized_prefix.rstrip('/')}/"
        if args.s3_tokenized_prefix
        else None
    )
    if tok_uri:
        for row in sources:
            row["uri"] = tok_uri

    created_at = datetime.now(timezone.utc).isoformat()
    # Force a fresh boto3 client after session reload (do not reuse stale default).
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
        sources=sources,
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
