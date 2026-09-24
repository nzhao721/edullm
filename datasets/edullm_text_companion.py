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
import os
from concurrent.futures import ProcessPoolExecutor
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
        self.current_fh = gzip.open(self.current_path, "wb", compresslevel=1)
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


def resolve_dclm_chunk_paths(run_dir: Path) -> list[Path]:
    """Return pre-split DCLM chunk paths when available."""
    chunks_dir = run_dir / "trim" / "dclm" / "label_chunks"
    if not chunks_dir.is_dir():
        return []
    paths = sorted(chunks_dir.glob("dclm-label-*.json.gz"))
    if not paths:
        return []
    return paths


def merge_part_shards(part_dirs: list[Path], out_dir: Path) -> int:
    """Rename per-part shards into one contiguous ``train-*.jsonl.gz`` sequence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("train-*.jsonl.gz"):
        stale.unlink()
    shard_idx = 0
    for part_dir in part_dirs:
        for shard in sorted(part_dir.glob("train-*.jsonl.gz")):
            dest = out_dir / f"train-{shard_idx:05d}.jsonl.gz"
            shard.replace(dest)
            shard_idx += 1
    return shard_idx


def stage_source_text(
    *,
    source: str,
    text_paths: list[Path],
    out_dir: Path,
    shard_bytes: int = MAX_TEXT_SHARD_BYTES,
    seq_offsets: list[int] | None = None,
) -> dict[str, int]:
    """Write every selected source document once without re-tokenizing it."""
    print(f"staging text/{source}: {len(text_paths)} input path(s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl.gz"):
        stale.unlink()
    writer = JsonlShardWriter(out_dir, shard_bytes=shard_bytes)
    docs = 0
    skipped = 0
    offsets = seq_offsets or [0] * len(text_paths)
    if len(offsets) != len(text_paths):
        raise ValueError(f"{source}: seq_offsets length mismatch")
    for path, seq_offset in zip(text_paths, offsets):
        seq = int(seq_offset)
        for obj in iter_docs(path):
            record = normalize_record(obj, source=source, seq=seq)
            if record is None:
                skipped += 1
                continue
            writer.write_record(record)
            seq += 1
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


def _stage_source_part(
    *,
    source: str,
    text_path: Path,
    out_dir: Path,
    shard_bytes: int,
    seq_offset: int,
) -> dict[str, int]:
    return stage_source_text(
        source=source,
        text_paths=[text_path],
        out_dir=out_dir,
        shard_bytes=shard_bytes,
        seq_offsets=[seq_offset],
    )


def stage_source_text_parallel(
    *,
    source: str,
    text_paths: list[Path],
    out_dir: Path,
    shard_bytes: int = MAX_TEXT_SHARD_BYTES,
    workers: int = 32,
    seq_offsets: list[int] | None = None,
) -> dict[str, int]:
    """Stage one source from multiple input shards in parallel, then merge."""
    if len(text_paths) < 2:
        return stage_source_text(
            source=source,
            text_paths=text_paths,
            out_dir=out_dir,
            shard_bytes=shard_bytes,
            seq_offsets=seq_offsets,
        )
    workers = max(1, min(int(workers), len(text_paths)))
    offsets = seq_offsets or [0] * len(text_paths)
    if len(offsets) != len(text_paths):
        raise ValueError(f"{source}: seq_offsets length mismatch")
    parts_root = out_dir.parent / f".{source}-text-parts"
    if parts_root.exists():
        import shutil

        shutil.rmtree(parts_root)
    parts_root.mkdir(parents=True, exist_ok=True)
    print(
        f"staging text/{source}: {len(text_paths)} chunk(s) with {workers} workers",
        flush=True,
    )
    totals = {"train_docs": 0, "train_shards": 0, "skipped_whitespace": 0}
    part_dirs: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for index, (path, seq_offset) in enumerate(zip(text_paths, offsets)):
            part_dir = parts_root / f"part-{index:05d}"
            part_dirs.append(part_dir)
            futures.append(
                pool.submit(
                    _stage_source_part,
                    source=source,
                    text_path=path,
                    out_dir=part_dir,
                    shard_bytes=shard_bytes,
                    seq_offset=int(seq_offset),
                )
            )
        for future in futures:
            stats = future.result()
            totals["train_docs"] += int(stats["train_docs"])
            totals["train_shards"] += int(stats["train_shards"])
            totals["skipped_whitespace"] += int(stats.get("skipped_whitespace", 0))
    totals["train_shards"] = merge_part_shards(part_dirs, out_dir)
    import shutil

    shutil.rmtree(parts_root)
    print(
        f"staged text/{source}: docs={totals['train_docs']:,} "
        f"shards={totals['train_shards']}",
        flush=True,
    )
    return totals


def stage_text_companion(
    *,
    sources: list[str],
    run_dir: Path,
    out_root: Path,
    shard_bytes: int = MAX_TEXT_SHARD_BYTES,
    text_paths_by_source: dict[str, list[Path]] | None = None,
    parallel_sources: set[str] | None = None,
    text_workers: int = 32,
    skip_sources: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Populate ``out_root/text/<source>/`` for every mix source."""
    text_root = out_root / "text"
    text_root.mkdir(parents=True, exist_ok=True)
    ordered_sources = sorted(sources)
    skip = skip_sources or set()
    parallel = parallel_sources or set()
    paths_by_source: dict[str, list[Path]] = {}
    offsets_by_source: dict[str, list[int]] = {}
    for source in ordered_sources:
        if source in (text_paths_by_source or {}):
            paths_by_source[source] = list(text_paths_by_source[source])
        elif source == "dclm":
            chunk_paths = resolve_dclm_chunk_paths(run_dir)
            if chunk_paths:
                paths_by_source[source] = chunk_paths
                manifest = run_dir / "labels" / "dclm_chunk_manifest.jsonl"
                if manifest.is_file():
                    rows = [
                        json.loads(line)
                        for line in manifest.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    offsets_by_source[source] = [
                        int(row.get("line_offset", 0)) for row in rows
                    ]
            else:
                paths_by_source[source] = resolve_text_paths(
                    source=source, run_dir=run_dir
                )
        else:
            paths_by_source[source] = resolve_text_paths(
                source=source, run_dir=run_dir
            )

    pending = [source for source in ordered_sources if source not in skip]
    parallel_pending = [s for s in pending if s in parallel and len(paths_by_source[s]) > 1]
    serial_pending = [s for s in pending if s not in parallel_pending]
    stats: dict[str, dict[str, int]] = {}

    # Parallel multi-chunk sources (e.g. DCLM) run in this process so their
    # own ProcessPoolExecutors are not nested inside another pool.
    for source in parallel_pending:
        stats[source] = stage_source_text_parallel(
            source=source,
            text_paths=paths_by_source[source],
            out_dir=text_root / source,
            shard_bytes=shard_bytes,
            workers=text_workers,
            seq_offsets=offsets_by_source.get(source),
        )

    max_workers = min(len(serial_pending), max(1, os.cpu_count() or 1)) or 1
    if serial_pending:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures: dict[str, Any] = {
                source: pool.submit(
                    stage_source_text,
                    source=source,
                    text_paths=paths_by_source[source],
                    out_dir=text_root / source,
                    shard_bytes=shard_bytes,
                    seq_offsets=offsets_by_source.get(source),
                )
                for source in serial_pending
            }
            for source in serial_pending:
                stats[source] = futures[source].result()
    for source in sorted(skip):
        out_dir = text_root / source
        shard_count = len(list(out_dir.glob("train-*.jsonl.gz")))
        if shard_count == 0:
            raise FileNotFoundError(
                f"skip_sources requested {source!r} but no staged text shards exist"
            )
        print(f"skipped text/{source}: reusing {shard_count} shard(s)", flush=True)
    return stats
