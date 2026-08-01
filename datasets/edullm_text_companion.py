#!/usr/bin/env python3
"""Stage ``text/<source>/`` JSONL companions for multi-group pretrain publishes.

Documents are read from the same selected source files that produced the packed token
group (trimmed json.gz or RefHQ ``out/<source>/documents-*.json.gz``). The token group
retains its existing token-level train/val split; this raw companion is a complete
``train`` document stream and is not re-tokenized merely to recreate that byte-level
split.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from olmo_shard_utils import doc_text, iter_docs

MAX_TEXT_SHARD_BYTES = 1_073_741_824

PUBLISH_PROFILE = {"tokens": "pretrain-tokens/v1", "text": "text-corpus/v1"}
TEXT_GROUP_META = {"text": {"record_schema": {"text": "str", "id": "str"}}}


def resolve_text_paths(*, source: str, run_dir: Path) -> list[Path]:
    """Ordered raw-document paths for one mix source under a FarmShare run tree."""
    trimmed = run_dir / "trim" / source / f"{source}-trimmed.json.gz"
    if trimmed.is_file():
        return [trimmed]
    upsampled = run_dir / "trim" / source / f"{source}-upsampled.json.gz"
    if upsampled.is_file():
        return [upsampled]
    data_trimmed = run_dir / "data" / source / f"{source}-trimmed.json.gz"
    if data_trimmed.is_file():
        return [data_trimmed]
    out_dir = run_dir / "out" / source
    if out_dir.is_dir():
        shards = sorted(out_dir.glob("documents-*.json.gz"))
        if shards:
            return shards

    manifest_path = run_dir / "plan" / "manifest.jsonl"
    if manifest_path.is_file():
        paths: list[Path] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            domain = str(row.get("domain") or row.get("source") or "").strip()
            if domain != source:
                continue
            rel = str(row.get("path") or row.get("rel_path") or "").lstrip("/")
            if not rel:
                continue
            for candidate in (
                run_dir / rel,
                run_dir / "data" / rel,
                run_dir / "data" / source / Path(rel).name,
            ):
                if candidate.is_file():
                    paths.append(candidate)
                    break
            else:
                raise FileNotFoundError(f"manifest row path missing on disk: {rel} ({source})")
        if paths:
            return paths

    raise FileNotFoundError(
        f"no text companion for source {source!r} under {run_dir} "
        f"(expected trim/{source}/*-trimmed.json.gz, out/{source}/documents-*.json.gz, "
        f"or plan/manifest.jsonl rows)"
    )


def normalize_record(obj: dict[str, Any], *, source: str, seq: int) -> dict[str, str] | None:
    text = doc_text(obj)
    if not text.strip():
        return None
    doc_id = obj.get("id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        digest = hashlib.sha256()
        digest.update(source.encode())
        digest.update(b"\0")
        digest.update(str(seq).encode())
        digest.update(b"\0")
        digest.update(text.encode("utf-8", errors="surrogatepass"))
        doc_id = digest.hexdigest()
    return {"id": doc_id, "text": text}


class JsonlShardWriter:
    def __init__(self, out_dir: Path, *, shard_bytes: int = MAX_TEXT_SHARD_BYTES) -> None:
        self.out_dir = out_dir
        self.shard_bytes = min(shard_bytes, MAX_TEXT_SHARD_BYTES)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = 0
        self.current_path: Path | None = None
        self.current_fh = None
        self.current_size = 0
        self.written: list[Path] = []

    def _open_next(self) -> None:
        if self.current_fh is not None:
            self.current_fh.close()
        self.current_path = self.out_dir / f"train-{self.shard_idx:05d}.jsonl.gz"
        self.current_fh = gzip.open(self.current_path, "wb")
        self.current_size = 0
        self.written.append(self.current_path)
        self.shard_idx += 1

    def write_record(self, record: dict[str, str]) -> None:
        line = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        if self.current_fh is None or self.current_size + len(line) > self.shard_bytes:
            self._open_next()
        assert self.current_fh is not None
        self.current_fh.write(line)
        self.current_size += len(line)

    def close(self) -> list[Path]:
        if self.current_fh is not None:
            self.current_fh.close()
            self.current_fh = None
        return self.written


def stage_source_text(
    *,
    source: str,
    text_paths: list[Path],
    out_dir: Path,
    shard_bytes: int = MAX_TEXT_SHARD_BYTES,
) -> dict[str, int]:
    """Write every selected source document once without re-tokenizing it."""
    print(f"staging text/{source}: {len(text_paths)} input path(s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl.gz"):
        stale.unlink()
    writer = JsonlShardWriter(out_dir, shard_bytes=shard_bytes)
    docs = 0
    skipped = 0
    for path in text_paths:
        for obj in iter_docs(path):
            record = normalize_record(obj, source=source, seq=docs)
            if record is None:
                skipped += 1
                continue
            writer.write_record(record)
            docs += 1
    train_shards = writer.close()
    if not docs:
        raise ValueError(f"{source}: no non-empty documents")
    stats = {
        "train_docs": docs,
        "train_shards": len(train_shards),
        "skipped_whitespace": skipped,
    }
    print(
        f"staged text/{source}: docs={stats['train_docs']:,} "
        f"shards={stats['train_shards']}"
        + (f" skipped_whitespace={skipped:,}" if skipped else ""),
        flush=True,
    )
    return stats


def stage_text_companion(
    *,
    sources: list[str],
    run_dir: Path,
    out_root: Path,
    shard_bytes: int = MAX_TEXT_SHARD_BYTES,
    text_paths_by_source: dict[str, list[Path]] | None = None,
) -> dict[str, dict[str, int]]:
    """Populate ``out_root/text/<source>/`` for every mix source."""
    text_root = out_root / "text"
    text_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}
    for source in sorted(sources):
        paths = (text_paths_by_source or {}).get(source) or resolve_text_paths(
            source=source, run_dir=run_dir
        )
        stats[source] = stage_source_text(
            source=source,
            text_paths=paths,
            out_dir=text_root / source,
            shard_bytes=shard_bytes,
        )
    return stats
