#!/usr/bin/env python3
"""Stage FineWeb-Edu 1B (SmolLM2 tokens + raw text) and publish to edullm-landing.

Sources on FarmShare:
  - tokenized: fineweb-edu-1b-smollm2-tokenized/train_tokens.bin (HuggingFaceFW/fineweb_edu_100BT-shuffled)
  - raw text:  fineweb-edu-1b-smollm2-raw/shards/*.jsonl.gz (HuggingFaceFW/fineweb-edu sample-100BT)

These are same-budget companions, not document-aligned (different HF repos / doc counts).
Layout: tokens/fineweb-edu/{train,val}-*.u32le.bin + text/fineweb-edu/train-*.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shlex
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ID = "pretrain/fineweb-edu-1b"
DEFAULT_TOKENIZER_ID = "tokenizer/smollm2-bpe"
DEFAULT_PURPOSE = (
    "1B-token FineWeb-Edu corpus (SmolLM2-tokenized + raw text companion) "
    "for SmolLM2-135M ladder runs and FineWeb baselines"
)
DEFAULT_SHARD_BYTES = 1_073_741_824  # 1 GiB, uint32-aligned
VAL_FRACTION = 0.0015
SOURCE_LABEL = "fineweb-edu"
SESSION_RELOAD_SECONDS = 5 * 60
PUBLISH_PROFILE = {"tokens": "pretrain-tokens/v1", "text": "text-corpus/v1"}
TEXT_GROUP_META = {"text": {"record_schema": {"text": "str", "id": "str"}}}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)


def _align_shard_bytes(n: int) -> int:
    n -= n % 4
    if n < 4:
        raise ValueError("shard_bytes must be at least 4")
    return n


def split_bin_to_shards(
    bin_path: Path,
    out_dir: Path,
    *,
    shard_bytes: int,
) -> list[Path]:
    """Split a headerless uint32 memmap into train-*.u32le.bin shards."""
    shard_bytes = _align_shard_bytes(shard_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = bin_path.stat().st_size
    if total % 4 != 0:
        raise ValueError(f"{bin_path}: size {total} is not a multiple of 4 (uint32)")

    written: list[Path] = []
    shard_idx = 0
    with bin_path.open("rb") as src:
        while True:
            chunk = src.read(shard_bytes)
            if not chunk:
                break
            if len(chunk) % 4 != 0:
                raise ValueError(f"{bin_path}: trailing partial token in shard {shard_idx}")
            out = out_dir / f"train-{shard_idx:05d}.u32le.bin"
            out.write_bytes(chunk)
            written.append(out)
            shard_idx += 1
    if not written:
        raise ValueError(f"{bin_path}: empty input")
    return written


def carve_val_holdout(source_dir: Path, *, fraction: float = VAL_FRACTION) -> int:
    """Carve ``fraction`` of train tokens into val-00000.u32le.bin (from shard tails)."""
    if not (0.0 < fraction < 0.5):
        raise SystemExit(f"val fraction must be in (0, 0.5); got {fraction}")

    train_shards = sorted(source_dir.glob("train-*.u32le.bin"))
    if not train_shards:
        raise SystemExit(f"no train shards under {source_dir}")
    source_bytes = sum(p.stat().st_size for p in train_shards)
    if source_bytes % 4 != 0:
        raise SystemExit(f"train bytes {source_bytes} not uint32-aligned")
    val_bytes = int(source_bytes * fraction)
    val_bytes -= val_bytes % 4
    if val_bytes < 4:
        raise SystemExit(f"val carve too small ({val_bytes} bytes at fraction={fraction})")

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
        raise SystemExit(f"could not carve {val_bytes} bytes (short by {remaining})")

    val_path = source_dir / "val-00000.u32le.bin"
    val_path.write_bytes(b"".join(reversed(chunks)))
    n_tokens = val_bytes // 4
    print(
        f"carved val: {val_path} ({n_tokens:,} tokens, {fraction:.4%} of source)",
        flush=True,
    )
    return n_tokens


def stage_text_from_raw(
    *,
    raw_shards_dir: Path,
    out_dir: Path,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> dict[str, int]:
    """Rewrite FineWeb raw jsonl.gz (doc_id/text/…) → text-corpus records (id/text)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl.gz"):
        stale.unlink()

    inputs = sorted(raw_shards_dir.glob("train-*.jsonl.gz"))
    if not inputs:
        raise SystemExit(f"no train-*.jsonl.gz under {raw_shards_dir}")

    shard_idx = 0
    current_fh = None
    current_size = 0
    docs = 0
    skipped = 0
    written: list[Path] = []

    def open_next() -> None:
        nonlocal shard_idx, current_fh, current_size
        if current_fh is not None:
            current_fh.close()
        path = out_dir / f"train-{shard_idx:05d}.jsonl.gz"
        current_fh = gzip.open(path, "wb")
        current_size = 0
        written.append(path)
        shard_idx += 1

    for path in inputs:
        with gzip.open(path, "rt", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    skipped += 1
                    continue
                doc_id = row.get("doc_id")
                if doc_id is None:
                    doc_id = row.get("id")
                record = {"id": str(doc_id), "text": text}
                payload = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
                if current_fh is None or current_size + len(payload) > shard_bytes:
                    open_next()
                assert current_fh is not None
                current_fh.write(payload)
                current_size += len(payload)
                docs += 1

    if current_fh is not None:
        current_fh.close()
    if not docs:
        raise SystemExit("no non-empty text documents staged")
    stats = {"train_docs": docs, "train_shards": len(written), "skipped_whitespace": skipped}
    print(
        f"staged text/{SOURCE_LABEL}: docs={docs:,} shards={len(written)}"
        + (f" skipped_whitespace={skipped:,}" if skipped else ""),
        flush=True,
    )
    return stats


def stage_publish_layout(
    *,
    tokenized_root: Path,
    raw_root: Path,
    out_root: Path,
    shard_bytes: int,
    force: bool,
    val_fraction: float,
) -> None:
    if out_root.exists():
        if not force:
            raise SystemExit(f"staging dir exists: {out_root} (pass --force to replace)")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tokens_bin = tokenized_root / "train_tokens.bin"
    if not tokens_bin.is_file():
        raise SystemExit(f"missing {tokens_bin}")
    tokens_dir = out_root / "tokens" / SOURCE_LABEL
    shards = split_bin_to_shards(tokens_bin, tokens_dir, shard_bytes=shard_bytes)
    print(f"staged tokens/{SOURCE_LABEL}: {len(shards)} train shards from {tokens_bin}", flush=True)
    carve_val_holdout(tokens_dir, fraction=val_fraction)

    raw_shards = raw_root / "shards"
    if not raw_shards.is_dir():
        raise SystemExit(f"missing raw shards dir: {raw_shards}")
    stage_text_from_raw(
        raw_shards_dir=raw_shards,
        out_dir=out_root / "text" / SOURCE_LABEL,
        shard_bytes=shard_bytes,
    )


def stage_tokenizer_dir(*, hf_snapshot: Path, out_dir: Path, force: bool) -> Path:
    if out_dir.exists():
        if not force:
            raise SystemExit(f"tokenizer stage exists: {out_dir} (pass --force)")
        shutil.rmtree(out_dir)
    tok_group = out_dir / "tokenizer"
    tok_group.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        src = hf_snapshot / name
        if not src.is_file():
            raise SystemExit(f"tokenizer file missing in snapshot: {src}")
        # Resolve HF hub symlinks to real blob content.
        shutil.copyfile(src.resolve(), tok_group / name)
    print(f"staged tokenizer files from {hf_snapshot} → {tok_group}", flush=True)
    return out_dir


def _apply_session_env_file(path: Path) -> bool:
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


class RefreshingBoto3S3:
    """edullm_data S3 wrapper that rotates STS from laptop-pushed aws-session.env."""

    def __init__(
        self,
        session_env: Path | None = None,
        *,
        region: str = "us-east-1",
        reload_seconds: float = SESSION_RELOAD_SECONDS,
    ) -> None:
        import threading

        import boto3
        from edullm_data.s3 import Boto3S3

        self._boto3 = boto3
        self._Boto3S3 = Boto3S3
        self.session_env = session_env or Path(os.environ.get("AWS_SESSION_ENV") or "")
        self.region = region
        self.reload_seconds = reload_seconds
        self._lock = threading.Lock()
        self._inner: Boto3S3 | None = None
        self._loaded_at = 0.0
        self._mtime = 0.0
        self._fingerprint = ""
        self._rebuild(force=True)

    def _rebuild(self, *, force: bool = False) -> None:
        now = time.time()
        mtime = 0.0
        if self.session_env and self.session_env.is_file():
            mtime = self.session_env.stat().st_mtime
        stale = (now - self._loaded_at) >= self.reload_seconds
        changed = mtime > self._mtime
        if not force and self._inner is not None and not stale and not changed:
            return
        if self.session_env and self.session_env.is_file():
            _apply_session_env_file(self.session_env)
        key = os.environ.get("AWS_ACCESS_KEY_ID") or ""
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
        token = os.environ.get("AWS_SESSION_TOKEN") or ""
        if not (key and secret and token):
            raise RuntimeError("AWS session env missing access key / secret / token")
        fingerprint = f"{key[-4:]}:{token[-8:]}:{mtime}"
        client = self._boto3.client(
            "s3",
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            aws_session_token=token,
            region_name=self.region,
        )
        self._inner = self._Boto3S3(client)
        if fingerprint != self._fingerprint:
            print(f"publish S3 credentials rotated (...{key[-4:]})", flush=True)
            self._fingerprint = fingerprint
        self._loaded_at = now
        self._mtime = mtime

    def _call(self, method: str, *args, **kwargs):
        with self._lock:
            self._rebuild()
            fn = getattr(self._inner, method)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "ExpiredToken" not in msg and "InvalidToken" not in msg and "expired" not in msg.lower():
                raise
            with self._lock:
                self._rebuild(force=True)
                fn = getattr(self._inner, method)
            print(f"publish S3 retry after credential refresh ({method})", flush=True)
            return fn(*args, **kwargs)

    def get(self, bucket: str, key: str) -> bytes:
        return self._call("get", bucket, key)

    def get_range(self, bucket: str, key: str, start: int, length: int) -> bytes:
        return self._call("get_range", bucket, key, start, length)

    def head(self, bucket: str, key: str) -> dict:
        return self._call("head", bucket, key)

    def list(self, bucket: str, prefix: str) -> list:
        return self._call("list", bucket, prefix)

    def hash_object(self, bucket: str, key: str) -> tuple[str, int]:
        return self._call("hash_object", bucket, key)

    def put(self, bucket: str, key: str, body: bytes, *, content_type: str | None = None) -> None:
        return self._call("put", bucket, key, body, content_type=content_type)

    def put_file(self, bucket: str, key: str, local_path: str) -> None:
        return self._call("put_file", bucket, key, local_path)

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
        return self._call("copy", src_bucket, src_key, dst_bucket, dst_key)

    def delete(self, bucket: str, key: str) -> None:
        return self._call("delete", bucket, key)


def wait_for_validated(
    s3: RefreshingBoto3S3,
    *,
    dataset_id: str,
    version: str,
    timeout_s: float = 1800.0,
    poll_s: float = 20.0,
) -> None:
    """Block until the airlock validator promotes a dataset to edullm-data."""
    key = f"{dataset_id}/{version}/_VALIDATED.json"
    deadline = time.time() + timeout_s
    print(f"waiting for s3://edullm-data/{key} (timeout={timeout_s:.0f}s)", flush=True)
    while time.time() < deadline:
        try:
            s3.head("edullm-data", key)
            print(f"validated: s3://edullm-data/{key}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 — keep polling until timeout
            print(f"  not yet validated ({exc.__class__.__name__}); sleep {poll_s:.0f}s", flush=True)
            time.sleep(poll_s)
            if s3.session_env and s3.session_env.is_file():
                _apply_session_env_file(s3.session_env)
    raise SystemExit(f"timed out waiting for s3://edullm-data/{key}")


def ensure_edullm_data() -> None:
    try:
        import edullm_data  # noqa: F401
    except ImportError:
        raise SystemExit(
            "edullm-data is not installed. Run:\n"
            '  pip install "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"'
        ) from None


def build_sources(*, tok_meta: dict, raw_meta: dict) -> list[dict]:
    return [
        {
            "name": "FineWeb-Edu 100BT-shuffled (tokenized prefix)",
            "tokens": int(tok_meta.get("num_tokens") or 1_000_000_000),
            "license": "ODC-By-1.0",
            "scope": "upstream-full-collection",
            "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT-shuffled",
        },
        {
            "name": "FineWeb-Edu sample-100BT (raw text companion)",
            "tokens": int(raw_meta.get("num_tokens") or 1_000_000_000),
            "license": "ODC-By-1.0",
            "scope": "upstream-full-collection",
            "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        default=Path("/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-tokenized"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/scratch/users/nzhao2/agent-runs/fineweb-edu-1b-smollm2-raw"),
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=Path(
            "/scratch/users/nzhao2/hf-cache/hub/models--HuggingFaceTB--SmolLM2-135M/"
            "snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
        ),
    )
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-stage-dir", type=Path, default=None)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--hash-workers", type=int, default=8)
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--skip-tokenizer-publish", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tok_meta = json.loads((args.tokenized_root / "meta.json").read_text(encoding="utf-8"))
    raw_meta = json.loads((args.raw_root / "meta.json").read_text(encoding="utf-8"))

    tokenizer_stage = args.tokenizer_stage_dir or (args.stage_dir.parent / "tokenizer-stage")
    if not args.skip_tokenizer_publish:
        stage_tokenizer_dir(
            hf_snapshot=args.tokenizer_snapshot,
            out_dir=tokenizer_stage,
            force=args.force,
        )

    if not args.skip_stage:
        stage_publish_layout(
            tokenized_root=args.tokenized_root,
            raw_root=args.raw_root,
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

    session_env = Path(os.environ.get("AWS_SESSION_ENV", "") or "")
    if session_env.is_file() and _apply_session_env_file(session_env):
        print(f"reloaded AWS session before publish from {session_env}", flush=True)
    s3 = RefreshingBoto3S3(session_env if session_env.is_file() else None)

    if not args.skip_tokenizer_publish:
        try:
            validate_dataset_id(args.tokenizer_id)
        except Exception as exc:
            raise SystemExit(f"invalid tokenizer_id {args.tokenizer_id!r}: {exc}") from exc
        created_at = datetime.now(timezone.utc).isoformat()
        tok_plan = publish(
            tokenizer_stage,
            dataset_id=args.tokenizer_id,
            purpose=(
                "Published SmolLM2 tokenizer so FineWeb-Edu and SmolLM2 corpora "
                "own the tokenizer they were built with"
            ),
            profile="tokenizer/v1",
            s3=s3,
            created_at=created_at,
            about=(
                "HuggingFaceTB/SmolLM2-135M tokenizer files (BPE) vendored for eduLLM "
                "pretrain corpora. Shared across SmolLM2 size variants."
            ),
            sources=[
                {
                    "name": "HuggingFaceTB/SmolLM2-135M",
                    "uri": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M",
                    "scope": "upstream-full-collection",
                }
            ],
        )
        print(
            json.dumps(
                {
                    "dataset_id": tok_plan.dataset_id,
                    "version": tok_plan.version,
                    "payload_objects": len(tok_plan.payload_keys),
                },
                indent=2,
            ),
            flush=True,
        )
        print(
            f"published tokenizer to s3://edullm-landing/{tok_plan.dataset_id}/{tok_plan.version}/",
            flush=True,
        )
        wait_for_validated(
            s3,
            dataset_id=tok_plan.dataset_id,
            version=tok_plan.version,
        )

    try:
        validate_dataset_id(args.dataset_id)
    except Exception as exc:
        raise SystemExit(f"invalid dataset_id {args.dataset_id!r}: {exc}") from exc

    about = (
        "FineWeb-Edu ~1B-token SmolLM2-tokenized stream plus a raw text-corpus companion. "
        f"Token group: first {int(tok_meta.get('num_tokens') or 0):,} tokens from "
        f"{tok_meta.get('hf_path')} ({int(tok_meta.get('num_docs') or 0):,} docs). "
        f"Text group: {int(raw_meta.get('num_docs') or 0):,} documents (~1B-token budget) from "
        f"{raw_meta.get('hf_path')} / {raw_meta.get('hf_name')}. "
        "Shards nest as tokens/fineweb-edu and text/fineweb-edu."
    )
    notes = (
        f"Validation split: {args.val_fraction:.4%} of the token stream carved into "
        f"tokens/{SOURCE_LABEL}/val-00000.u32le.bin. "
        "IMPORTANT: tokenized bytes and raw text come from different FineWeb-Edu HF artifacts "
        "(fineweb_edu_100BT-shuffled vs fineweb-edu sample-100BT) and are NOT document-aligned "
        f"(tok docs={tok_meta.get('num_docs')}, raw docs={raw_meta.get('num_docs')}). "
        "Do not assume text[i] corresponds to tokens for the same document. "
        "FarmShare legacy paths: fineweb-edu-1b-smollm2-tokenized/, fineweb-edu-1b-smollm2-raw/."
    )

    if session_env.is_file() and _apply_session_env_file(session_env):
        print(f"reloaded AWS session before corpus publish from {session_env}", flush=True)
        s3 = RefreshingBoto3S3(session_env)

    created_at = datetime.now(timezone.utc).isoformat()
    plan = publish(
        args.stage_dir,
        dataset_id=args.dataset_id,
        purpose=args.purpose,
        profile=PUBLISH_PROFILE,
        tokenizer=args.tokenizer_id,
        group_meta=TEXT_GROUP_META,
        s3=s3,
        created_at=created_at,
        hash_workers=args.hash_workers,
        copy_workers=args.copy_workers,
        about=about,
        sources=build_sources(tok_meta=tok_meta, raw_meta=raw_meta),
        notes=notes,
        license={"id": "ODC-By-1.0", "basis": "declared"},
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
