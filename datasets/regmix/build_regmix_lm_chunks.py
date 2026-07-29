#!/usr/bin/env python3
"""Build chunked RegMix JSONL.GZ work items for document-level LM scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterator


def open_text_stream(path: Path):
    name = path.name
    if name.endswith(".jsonl.zstd") or name.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), raw
    if name.endswith(".json.gz") or name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace"), None
    if name.endswith(".jsonl") or name.endswith(".json"):
        return path.open("rt", encoding="utf-8", errors="replace"), None
    raise ValueError(f"unsupported format: {path}")


def doc_text(obj: dict[str, Any]) -> str:
    for key in ("text", "content", "code", "body"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    parts = [v for v in obj.values() if isinstance(v, str) and v]
    return "\n".join(parts)


def iter_docs(path: Path) -> Iterator[tuple[int, str]]:
    stream, raw = open_text_stream(path)
    try:
        for line_index, line in enumerate(stream):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = doc_text(obj)
            if text:
                yield line_index, text
    finally:
        stream.close()
        if raw is not None:
            raw.close()


def shard_stem(index: int, rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"shard-{index:05d}-{digest}"


def write_chunk_manifest_row(
    handle,
    *,
    index: int,
    domain: str,
    source_rel_path: str,
    source_shard_index: int,
    chunk_index: int,
    path: Path,
    docs: int,
    source_start_doc: int,
    source_end_doc: int,
    est_tokens: int,
) -> None:
    handle.write(
        json.dumps(
            {
                "index": index,
                "domain": domain,
                "source_rel_path": source_rel_path,
                "source_shard_index": source_shard_index,
                "chunk_index": chunk_index,
                "path": str(path),
                "docs": docs,
                "source_start_doc": source_start_doc,
                "source_end_doc": source_end_doc,
                "est_tokens": est_tokens,
                "size": path.stat().st_size,
            },
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Default: OUT_ROOT/lm_work_manifest.jsonl",
    )
    parser.add_argument("--target-tokens-per-chunk", type=int, default=25_000_000)
    parser.add_argument("--min-docs-per-chunk", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root
    chunk_root = out_root / "source_chunks"
    manifest_out = args.manifest_out or (out_root / "lm_work_manifest.jsonl")
    summary_out = manifest_out.with_suffix(".summary.json")
    if manifest_out.exists() and summary_out.exists() and not args.force:
        print(summary_out.read_text(encoding="utf-8").strip())
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    chunk_root.mkdir(parents=True, exist_ok=True)
    tmp_manifest = Path(str(manifest_out) + ".tmp")

    shard_rows = [
        json.loads(line)
        for line in args.shard_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_docs = 0
    total_chunks = 0
    by_domain: dict[str, dict[str, int]] = {}

    with tmp_manifest.open("w", encoding="utf-8") as manifest_handle:
        for shard in shard_rows:
            domain = shard["domain"]
            source = Path(shard["path"])
            rel_path = shard["rel_path"]
            docs_est = max(1, int(shard.get("docs") or 1))
            tokens_est = max(docs_est, int(shard.get("est_tokens") or docs_est))
            tokens_per_doc = tokens_est / docs_est
            docs_per_chunk = max(
                args.min_docs_per_chunk,
                int(args.target_tokens_per_chunk / max(tokens_per_doc, 1.0)),
            )
            stem = shard_stem(int(shard["index"]), rel_path)
            domain_dir = chunk_root / domain
            domain_dir.mkdir(parents=True, exist_ok=True)

            chunk_index = 0
            docs_in_chunk = 0
            source_start_doc = 0
            source_doc_ordinal = 0
            chunk_path = domain_dir / f"{stem}-part-{chunk_index:05d}.jsonl.gz"
            tmp_chunk = Path(str(chunk_path) + ".tmp")
            chunk_handle = gzip.open(tmp_chunk, "wt", encoding="utf-8")

            def close_chunk() -> None:
                nonlocal chunk_handle, tmp_chunk, chunk_path, docs_in_chunk
                nonlocal source_start_doc, total_chunks, total_docs
                if docs_in_chunk == 0:
                    chunk_handle.close()
                    tmp_chunk.unlink(missing_ok=True)
                    return
                chunk_handle.close()
                tmp_chunk.replace(chunk_path)
                est_tokens = int(round(docs_in_chunk * tokens_per_doc))
                write_chunk_manifest_row(
                    manifest_handle,
                    index=total_chunks,
                    domain=domain,
                    source_rel_path=rel_path,
                    source_shard_index=int(shard["index"]),
                    chunk_index=chunk_index,
                    path=chunk_path,
                    docs=docs_in_chunk,
                    source_start_doc=source_start_doc,
                    source_end_doc=source_start_doc + docs_in_chunk,
                    est_tokens=est_tokens,
                )
                total_chunks += 1
                total_docs += docs_in_chunk
                stats = by_domain.setdefault(domain, {"chunks": 0, "docs": 0, "est_tokens": 0})
                stats["chunks"] += 1
                stats["docs"] += docs_in_chunk
                stats["est_tokens"] += est_tokens

            for line_index, text in iter_docs(source):
                if docs_in_chunk >= docs_per_chunk:
                    close_chunk()
                    chunk_index += 1
                    source_start_doc = source_doc_ordinal
                    docs_in_chunk = 0
                    chunk_path = domain_dir / f"{stem}-part-{chunk_index:05d}.jsonl.gz"
                    tmp_chunk = Path(str(chunk_path) + ".tmp")
                    chunk_handle = gzip.open(tmp_chunk, "wt", encoding="utf-8")

                chunk_handle.write(
                    json.dumps(
                        {"source_line": line_index, "source_doc": source_doc_ordinal, "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                docs_in_chunk += 1
                source_doc_ordinal += 1

            close_chunk()

    tmp_manifest.replace(manifest_out)
    summary = {
        "event": "lm_chunks_ready",
        "chunk_root": str(chunk_root),
        "manifest": str(manifest_out),
        "n_chunks": total_chunks,
        "n_docs": total_docs,
        "target_tokens_per_chunk": args.target_tokens_per_chunk,
        "by_domain": by_domain,
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
