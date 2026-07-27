#!/usr/bin/env python3
"""Label one OLMo-mix shard with compression ratio, Flesch, and MTLD.

Writes two portable artifacts for the same documents:
  1) docs JSONL.GZ  — text + metrics (self-contained for off-cluster training)
  2) metrics JSONL.GZ — metrics only (cheap sort/filter index)

Stable document ids are content-addressed so sorting/filtering is joinable
across environments without FarmShare paths.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from text_difficulty_metrics import compute_difficulty_metrics


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


def iter_docs(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
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
            if not text:
                continue
            yield line_index, obj, text
    finally:
        stream.close()
        if raw is not None:
            raw.close()


def stable_doc_id(domain: str, rel_path: str, line_index: int, text: str) -> str:
    digest = hashlib.sha1()
    digest.update(domain.encode())
    digest.update(b"\0")
    digest.update(rel_path.encode())
    digest.update(b"\0")
    digest.update(str(line_index).encode())
    digest.update(b"\0")
    digest.update(text.encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()


def _label_one(payload: tuple[str, str, int, str]) -> dict[str, Any]:
    domain, rel_path, line_index, text = payload
    metrics = compute_difficulty_metrics(text).as_dict()
    doc_id = stable_doc_id(domain, rel_path, line_index, text)
    return {
        "id": doc_id,
        "domain": domain,
        "source_path": rel_path,
        "source_line": line_index,
        "text": text,
        **metrics,
    }


def shard_stem(index: int, rel_path: str) -> str:
    """Short, filesystem-safe stem. Avoids ENAMETOOLONG on deep DCLM paths."""
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"shard-{index:05d}-{digest}"


def legacy_stem(rel_path: str) -> str:
    return rel_path.replace("\\", "/").replace("/", "__").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    lines = args.manifest.read_text(encoding="utf-8").splitlines()
    if args.index < 0 or args.index >= len(lines):
        print(f"index {args.index} out of range 0..{len(lines) - 1}", file=sys.stderr)
        return 2
    item = json.loads(lines[args.index])
    domain = item["domain"]
    rel_path = item["rel_path"]
    source = Path(item["path"])
    if not source.exists():
        print(f"missing source shard: {source}", file=sys.stderr)
        return 1

    stem = shard_stem(args.index, rel_path)
    docs_dir = args.out_root / "docs" / domain
    metrics_dir = args.out_root / "metrics" / domain
    map_dir = args.out_root / "path_map"
    docs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    docs_out = docs_dir / f"{stem}.jsonl.gz"
    metrics_out = metrics_dir / f"{stem}.metrics.jsonl.gz"
    done_marker = docs_dir / f"{stem}.done"
    map_out = map_dir / f"{stem}.json"
    if done_marker.exists() and docs_out.exists() and metrics_out.exists():
        print(json.dumps({"event": "skip_existing", "index": args.index, "docs": str(docs_out)}))
        return 0

    tmp_docs = Path(str(docs_out) + ".tmp")
    tmp_metrics = Path(str(metrics_out) + ".tmp")
    n_docs = 0
    workers = max(1, args.workers)

    with gzip.open(tmp_docs, "wt", encoding="utf-8") as docs_handle, gzip.open(
        tmp_metrics, "wt", encoding="utf-8"
    ) as metrics_handle, ProcessPoolExecutor(max_workers=workers) as pool:
        batch: list[tuple[str, str, int, str]] = []

        def flush(batch_items: list[tuple[str, str, int, str]]) -> int:
            if not batch_items:
                return 0
            written = 0
            for labeled in pool.map(_label_one, batch_items, chunksize=8):
                docs_handle.write(json.dumps(labeled, ensure_ascii=False) + "\n")
                metrics_row = {k: v for k, v in labeled.items() if k != "text"}
                metrics_handle.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")
                written += 1
            return written

        for line_index, _obj, text in iter_docs(source):
            batch.append((domain, rel_path, line_index, text))
            if len(batch) >= args.batch_size:
                n_docs += flush(batch)
                batch = []
        n_docs += flush(batch)

    tmp_docs.replace(docs_out)
    tmp_metrics.replace(metrics_out)
    shard_summary = {
        "index": args.index,
        "domain": domain,
        "source_path": rel_path,
        "stem": stem,
        "docs": n_docs,
        "docs_out": str(docs_out),
        "metrics_out": str(metrics_out),
        "workers": workers,
    }
    map_payload = {
        "index": args.index,
        "domain": domain,
        "rel_path": rel_path,
        "stem": stem,
        "docs_out": str(docs_out),
        "metrics_out": str(metrics_out),
        "docs": n_docs,
    }
    map_out.write_text(json.dumps(map_payload, indent=2) + "\n", encoding="utf-8")
    done_marker.write_text(json.dumps(shard_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "labeled_shard", **shard_summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
