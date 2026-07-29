#!/usr/bin/env python3
"""Finalize OLMo difficulty labels into a portable training package.

Produces:
  - labels/SCHEMA.json
  - labels/shard_status.jsonl
  - labels/metrics_index.jsonl.gz  (all docs, metrics only, for sort/filter)
  - labels/READY
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def shard_stem(index: int, rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"shard-{index:05d}-{digest}"


def legacy_stem(rel_path: str) -> str:
    return rel_path.replace("\\", "/").replace("/", "__").replace(" ", "_")


SCHEMA = {
    "version": 1,
    "corpus": "allenai/olmo-mix-1124 stratified ~30B sample",
    "unit": "document",
    "metrics": {
        "compression_ratio": {
            "definition": "utf8_bytes / zlib.compress(utf8, level=6) bytes",
            "notes": "Higher means more compressible/redundant.",
            "suggested_easy_to_hard": "ascending",
        },
        "flesch_reading_ease": {
            "definition": "Kincaid (1975) Flesch Reading Ease",
            "notes": "Higher means easier English readability.",
            "suggested_easy_to_hard": "descending",
        },
        "mtld": {
            "definition": "Bidirectional MTLD, TTR threshold 0.72 (McCarthy & Jarvis 2010)",
            "notes": "Higher means greater lexical diversity.",
            "suggested_easy_to_hard": "ascending",
        },
    },
    "artifacts": {
        "docs": "labels/docs/<domain>/*.jsonl.gz — full text + metrics (portable training rows)",
        "metrics": "labels/metrics/<domain>/*.metrics.jsonl.gz — metrics-only rows",
        "metrics_index": "labels/metrics_index.jsonl.gz — concatenated metrics-only index",
    },
    "join_key": "id",
    "offline_training": [
        "Sort or filter metrics_index.jsonl.gz by a metric column.",
        "Join selected ids back to docs/*.jsonl.gz via id, or stream docs and keep ids in a set.",
        "Or use materialize_curriculum.py to emit ordered/filtered text shards.",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        default=None,
        help="Override SCHEMA corpus description (default: OLMo ~30B sample).",
    )
    args = parser.parse_args()

    labels_root = args.labels_root
    lines = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    status_path = labels_root / "shard_status.jsonl"
    index_path = labels_root / "metrics_index.jsonl.gz"
    schema_path = labels_root / "SCHEMA.json"
    ready_path = labels_root / "READY"

    missing: list[int] = []
    total_docs = 0
    with status_path.open("w", encoding="utf-8") as status_handle, gzip.open(
        index_path, "wt", encoding="utf-8"
    ) as index_handle:
        for item in lines:
            domain = item["domain"]
            rel_path = item["rel_path"]
            candidates = [shard_stem(item["index"], rel_path), legacy_stem(rel_path)]
            done = None
            metrics_file = None
            docs_file = None
            stem = None
            for candidate in candidates:
                candidate_done = labels_root / "docs" / domain / f"{candidate}.done"
                candidate_metrics = (
                    labels_root / "metrics" / domain / f"{candidate}.metrics.jsonl.gz"
                )
                candidate_docs = labels_root / "docs" / domain / f"{candidate}.jsonl.gz"
                if (
                    candidate_done.exists()
                    and candidate_metrics.exists()
                    and candidate_docs.exists()
                ):
                    done = candidate_done
                    metrics_file = candidate_metrics
                    docs_file = candidate_docs
                    stem = candidate
                    break
            if done is None or metrics_file is None or docs_file is None:
                missing.append(item["index"])
                status_handle.write(
                    json.dumps(
                        {
                            "index": item["index"],
                            "domain": domain,
                            "status": "missing",
                            "rel_path": rel_path,
                        }
                    )
                    + "\n"
                )
                continue
            summary = json.loads(done.read_text(encoding="utf-8"))
            total_docs += int(summary.get("docs", 0))
            status_handle.write(
                json.dumps(
                    {
                        "index": item["index"],
                        "domain": domain,
                        "status": "ok",
                        "stem": stem,
                        "docs": summary.get("docs"),
                        "docs_out": str(docs_file),
                        "metrics_out": str(metrics_file),
                        "rel_path": rel_path,
                    }
                )
                + "\n"
            )
            with gzip.open(metrics_file, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        index_handle.write(line if line.endswith("\n") else line + "\n")

    schema = dict(SCHEMA)
    if args.corpus:
        schema["corpus"] = args.corpus
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    if missing:
        print(
            json.dumps(
                {
                    "event": "finalize_incomplete",
                    "missing_shards": missing,
                    "n_missing": len(missing),
                    "docs_so_far": total_docs,
                },
                sort_keys=True,
            )
        )
        return 1

    ready = {
        "event": "labels_ready",
        "n_shards": len(lines),
        "n_docs": total_docs,
        "metrics_index": str(index_path),
        "schema": str(schema_path),
    }
    ready_path.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
