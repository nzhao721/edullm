#!/usr/bin/env python3
"""Finalize RegMix document-level LM labels into metrics indexes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def shard_stem(index: int, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"lm-{index:05d}-{digest}"


SCHEMA = {
    "version": 1,
    "corpus": "RegMix 10B trimmed corpus",
    "unit": "document",
    "join_key": "id",
    "metrics": {
        "avg_nll": {
            "definition": "Mean next-token negative log-likelihood per document under the averaged late RefHQ checkpoint.",
            "source": "average weights of RefHQ checkpoints step1000, step1125, step1315",
        },
        "avg_perplexity": {
            "definition": "exp(avg_nll), using mean document NLL so document length does not confound the score.",
        },
        "early_step250_avg_nll": {
            "definition": "Mean next-token NLL per document under RefHQ checkpoint step250.",
        },
        "late_avg_steps_1000_1125_1315_avg_nll": {
            "definition": "Mean next-token NLL per document under average weights of RefHQ checkpoints step1000, step1125, step1315.",
        },
        "learnability_late_minus_early_avg_nll": {
            "definition": "late_avg_steps_1000_1125_1315_avg_nll - early_step250_avg_nll.",
            "notes": "Negative values mean the late model assigns lower loss than the early model.",
        },
    },
    "artifacts": {
        "docs": "lm_labels/docs/<domain>/*.jsonl.gz - full text + metrics",
        "metrics": "lm_labels/metrics/<domain>/*.metrics.jsonl.gz - metrics only",
        "metrics_index": "lm_labels/metrics_index.jsonl.gz - concatenated metrics-only index",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--work-manifest", type=Path, required=True)
    args = parser.parse_args()

    labels_root = args.labels_root
    rows = [
        json.loads(line)
        for line in args.work_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    status_path = labels_root / "shard_status.jsonl"
    index_path = labels_root / "metrics_index.jsonl.gz"
    schema_path = labels_root / "SCHEMA.json"
    ready_path = labels_root / "READY"

    missing: list[int] = []
    total_docs = 0
    total_tokens = 0
    labels_root.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as status_handle, gzip.open(
        index_path, "wt", encoding="utf-8"
    ) as index_handle:
        for item in rows:
            domain = item["domain"]
            stem = shard_stem(int(item["index"]), item["path"])
            done = labels_root / "docs" / domain / f"{stem}.done"
            metrics_file = labels_root / "metrics" / domain / f"{stem}.metrics.jsonl.gz"
            docs_file = labels_root / "docs" / domain / f"{stem}.jsonl.gz"
            if not (done.exists() and metrics_file.exists() and docs_file.exists()):
                missing.append(int(item["index"]))
                status_handle.write(
                    json.dumps(
                        {
                            "index": item["index"],
                            "domain": domain,
                            "status": "missing",
                            "source_chunk": item["path"],
                        }
                    )
                    + "\n"
                )
                continue

            summary = json.loads(done.read_text(encoding="utf-8"))
            total_docs += int(summary.get("docs", 0))
            total_tokens += int(summary.get("scored_tokens", 0))
            status_handle.write(
                json.dumps(
                    {
                        "index": item["index"],
                        "domain": domain,
                        "status": "ok",
                        "docs": summary.get("docs"),
                        "scored_tokens": summary.get("scored_tokens"),
                        "docs_out": str(docs_file),
                        "metrics_out": str(metrics_file),
                        "source_chunk": item["path"],
                    }
                )
                + "\n"
            )
            with gzip.open(metrics_file, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        index_handle.write(line if line.endswith("\n") else line + "\n")

    schema_path.write_text(json.dumps(SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if missing:
        print(
            json.dumps(
                {
                    "event": "lm_finalize_incomplete",
                    "missing_chunks": missing,
                    "n_missing": len(missing),
                    "docs_so_far": total_docs,
                    "scored_tokens_so_far": total_tokens,
                },
                sort_keys=True,
            )
        )
        return 1

    ready = {
        "event": "lm_labels_ready",
        "n_chunks": len(rows),
        "n_docs": total_docs,
        "scored_tokens": total_tokens,
        "metrics_index": str(index_path),
        "schema": str(schema_path),
    }
    ready_path.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(ready, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
