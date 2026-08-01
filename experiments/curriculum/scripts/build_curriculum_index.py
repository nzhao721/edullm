#!/usr/bin/env python3
"""Build parent-pool RegMix-10B curriculum permutations.

The output coordinates are the exact flat chunks exposed by published
``pretrain/regmix-10b``: for every train shard in ``dataset_paths()`` order,
``(shard_tokens - 1) // 2048`` chunks are appended to the flat coordinate
space. This script never re-tokenizes documents for production output.

Exact mapping requires a local parent-layout descriptor captured from the
published parent manifest. It binds dataset id, version, manifest hash,
``dataset_paths()`` shard order, source labels, source-stream token offsets,
and token counts. Labeled rows must retain ``source_doc`` and ``n_tokens``.
Missing or inconsistent provenance fails closed.

Each parent chunk is owned by the document containing its first token. Per
metric, chunks are sorted by that owner's document rank and then by flat chunk
id, yielding a complete deterministic permutation of the parent pool.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# Allow importing sibling pacing helpers when run as a script.
_CUR_ROOT = Path(__file__).resolve().parents[1]
if str(_CUR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUR_ROOT))

from curriculum_pacing import DIFFICULTY_METRICS, METRIC_SORT  # noqa: E402

log = logging.getLogger("build_curriculum_index")

SEQ_LEN = 2048
TOKENIZER_ID = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100_257
DEFAULT_PARENT_DATASET_ID = "pretrain/regmix-10b"
# Documentation / dry-run listing only — publish via edullm-data, not a legacy bucket.
DEFAULT_DST_URI = "s3://edullm-data/curriculum/"


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open_maybe_gzip(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def load_metrics_index(path: Path) -> Dict[str, dict]:
    """Load metrics_index.jsonl.gz keyed by doc id."""
    rows: Dict[str, dict] = {}
    for obj in iter_jsonl(path):
        doc_id = obj.get("id")
        if not doc_id:
            continue
        rows[str(doc_id)] = obj
    return rows


def join_label_indexes(
    heuristic: Mapping[str, dict],
    lm: Mapping[str, dict],
) -> Tuple[List[dict], dict]:
    """Inner-join on id; keep heuristic fields and attach LM learnability columns."""
    lm_keys = (
        "avg_nll",
        "avg_perplexity",
        "early_step250_avg_nll",
        "late_avg_steps_1000_1125_1315_avg_nll",
        "learnability_late_minus_early_avg_nll",
        "source_path",
        "source_line",
        "source_doc",
        "n_tokens",
    )
    heur_keys = (
        "compression_ratio",
        "flesch_reading_ease",
        "mtld",
        "n_chars",
        "raw_bytes",
        "zlib_bytes",
        "n_words",
        "n_sentences",
        "n_syllables",
    )
    joined: List[dict] = []
    heur_only = 0
    lm_only = 0
    both = 0
    all_ids = set(heuristic) | set(lm)
    for doc_id in sorted(all_ids):
        h = heuristic.get(doc_id)
        l = lm.get(doc_id)
        if h is None:
            lm_only += 1
            continue
        if l is None:
            heur_only += 1
            continue
        both += 1
        row = {
            "id": doc_id,
            "domain": h.get("domain") or l.get("domain"),
        }
        for k in heur_keys:
            if k in h:
                row[k] = h[k]
        for k in lm_keys:
            if k in l:
                row[k] = l[k]
        for k in ("source_path", "source_line", "source_doc", "n_tokens"):
            if k not in row and k in h:
                row[k] = h[k]
        # Carry n_chars from either side.
        if "n_chars" not in row:
            for src in (h, l):
                if "n_chars" in src:
                    row["n_chars"] = src["n_chars"]
                    break
        joined.append(row)

    coverage = {
        "n_heuristic": len(heuristic),
        "n_lm": len(lm),
        "n_joined": both,
        "n_heuristic_only": heur_only,
        "n_lm_only": lm_only,
        "join_rate_vs_heuristic": (both / len(heuristic)) if heuristic else 0.0,
        "join_rate_vs_lm": (both / len(lm)) if lm else 0.0,
    }
    return joined, coverage


def _metric_value(row: Mapping[str, Any], column: str) -> Optional[float]:
    v = row.get(column)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def assign_ranks(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Return ``{metric_alias: {doc_id: rank}}`` with easy=0."""
    ranks: Dict[str, Dict[str, int]] = {}
    for alias, (column, reverse) in METRIC_SORT.items():
        scored: List[Tuple[float, str]] = []
        for row in rows:
            val = _metric_value(row, column)
            if val is None:
                continue
            scored.append((val, str(row["id"])))
        scored.sort(key=lambda t: t[0], reverse=reverse)
        ranks[alias] = {doc_id: i for i, (_, doc_id) in enumerate(scored)}
    return ranks


def write_doc_manifest(path: Path, rows: Sequence[Mapping[str, Any]], ranks: Mapping[str, Mapping[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            obj = dict(row)
            doc_id = str(row["id"])
            for alias in DIFFICULTY_METRICS:
                r = ranks.get(alias, {}).get(doc_id)
                if r is not None:
                    obj[f"difficulty_rank_{alias}"] = r
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_doc_chunks(n_tokens: int, seq_len: int) -> int:
    """Match ``MemmapTokenDataset`` shard-local training chunk count."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    return max(0, (int(n_tokens) - 1) // int(seq_len))


def _source_from_path(path: str) -> str:
    parts = Path(path).as_posix().strip("/").split("/")
    if "tokens" in parts:
        idx = parts.index("tokens")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(f"cannot derive source label from parent shard path {path!r}")


def load_parent_layout(
    *,
    path: Path,
    dataset_id: str,
    version: str,
    manifest_sha256: str,
    seq_len: int,
) -> dict:
    """Load a captured ``dataset_paths`` layout and verify its identity."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_id": dataset_id,
        "version": version,
        "manifest_sha256": manifest_sha256,
        "seq_len": int(seq_len),
        "tokenizer_id": TOKENIZER_ID,
        "eos_token_id": EOS_TOKEN_ID,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"{path}: parent {key} mismatch: expected {value!r}, "
                f"got {payload.get(key)!r}"
            )
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise SystemExit(f"{path}: expected non-empty shards in dataset_paths order")
    normalized: List[dict] = []
    expected_starts: Dict[str, int] = {}
    for index, raw in enumerate(shards):
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}: shard {index} is not an object")
        shard_path = raw.get("path")
        count = raw.get("count")
        start = raw.get("source_token_start")
        if not isinstance(shard_path, str) or not shard_path:
            raise SystemExit(f"{path}: shard {index} missing path")
        if not isinstance(count, int) or count < 0:
            raise SystemExit(f"{path}: shard {shard_path!r} has invalid count")
        if not isinstance(start, int) or start < 0:
            raise SystemExit(
                f"{path}: shard {shard_path!r} missing non-negative "
                "source_token_start provenance"
            )
        source = raw.get("source") or _source_from_path(shard_path)
        source = str(source)
        if start != expected_starts.get(source, 0):
            raise SystemExit(
                f"{path}: non-contiguous source offsets for {source!r}: "
                f"expected {expected_starts.get(source, 0)}, got {start}"
            )
        expected_starts[source] = start + count
        normalized.append(
            {
                "path": shard_path,
                "count": count,
                "source": source,
                "source_token_start": start,
                "n_chunks": count_doc_chunks(count, seq_len),
            }
        )
    source_totals = payload.get("source_total_tokens")
    if not isinstance(source_totals, dict) or not source_totals:
        raise SystemExit(
            f"{path}: source_total_tokens is required to prove labeled stream identity"
        )
    return {**payload, "shards": normalized, "source_total_tokens": source_totals}


def build_parent_pool_orders(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    ranks: Mapping[str, Mapping[str, int]],
    parent_layout: Mapping[str, Any],
    out_dir: Path,
    seq_len: int,
    eos_tokens_per_doc: int = 1,
) -> dict:
    """Map parent chunk starts to owner documents and emit full permutations."""
    import numpy as np

    docs_by_source: Dict[str, List[dict]] = {}
    source_paths: Dict[str, str] = {}
    for row in joined_rows:
        source = row.get("domain")
        source_path = row.get("source_path")
        source_doc = row.get("source_doc")
        n_tokens = row.get("n_tokens")
        if not isinstance(source, str) or not source:
            raise SystemExit(f"joined row {row.get('id')!r} missing domain/source")
        if not isinstance(source_doc, int) or source_doc < 0:
            raise SystemExit(
                f"joined row {row.get('id')!r} missing non-negative source_doc provenance"
            )
        if not isinstance(n_tokens, int) or n_tokens < 0:
            raise SystemExit(
                f"joined row {row.get('id')!r} missing non-negative n_tokens provenance"
            )
        if not isinstance(source_path, str) or not source_path:
            raise SystemExit(
                f"joined row {row.get('id')!r} missing source_path provenance"
            )
        normalized_source_path = Path(source_path).as_posix()
        if source not in normalized_source_path.split("/"):
            raise SystemExit(
                f"joined row {row.get('id')!r} source_path {source_path!r} "
                f"does not identify source {source!r}"
            )
        prior_path = source_paths.setdefault(source, normalized_source_path)
        if prior_path != normalized_source_path:
            raise SystemExit(
                f"source {source!r} spans multiple label source paths "
                f"({prior_path!r}, {normalized_source_path!r}); "
                "parent stream provenance is ambiguous"
            )
        docs_by_source.setdefault(source, []).append(
            {
                "id": str(row["id"]),
                "source_doc": source_doc,
                "n_stream_tokens": n_tokens + int(eos_tokens_per_doc),
            }
        )

    intervals: Dict[str, Tuple[List[int], List[int], List[str]]] = {}
    for source, docs in docs_by_source.items():
        docs.sort(key=lambda d: d["source_doc"])
        ordinals = [d["source_doc"] for d in docs]
        if ordinals != list(range(len(docs))):
            raise SystemExit(
                f"source {source!r}: joined labels do not cover contiguous source_doc "
                f"ordinals 0..{len(docs) - 1}; cannot map parent bytes exactly"
            )
        starts: List[int] = []
        ends: List[int] = []
        ids: List[str] = []
        cursor = 0
        for doc in docs:
            starts.append(cursor)
            cursor += int(doc["n_stream_tokens"])
            ends.append(cursor)
            ids.append(str(doc["id"]))
        expected_total = parent_layout["source_total_tokens"].get(source)
        if not isinstance(expected_total, int) or expected_total != cursor:
            raise SystemExit(
                f"source {source!r}: labeled stream has {cursor} tokens including EOS, "
                f"parent layout declares {expected_total!r}; exact mapping unavailable"
            )
        intervals[source] = (starts, ends, ids)

    chunk_rows: List[dict] = []
    global_chunk_idx = 0
    for shard_index, shard in enumerate(parent_layout["shards"]):
        source = shard["source"]
        if source not in intervals:
            raise SystemExit(f"parent source {source!r} has no joined label provenance")
        starts, ends, ids = intervals[source]
        for local_chunk_idx in range(int(shard["n_chunks"])):
            source_offset = int(shard["source_token_start"]) + local_chunk_idx * int(seq_len)
            doc_pos = bisect.bisect_right(starts, source_offset) - 1
            if doc_pos < 0 or source_offset >= ends[doc_pos]:
                raise SystemExit(
                    f"parent chunk {global_chunk_idx} start {source}:{source_offset} "
                    "is not covered by labeled document provenance"
                )
            doc_id = ids[doc_pos]
            row = {
                "global_chunk_idx": global_chunk_idx,
                "parent_shard_index": shard_index,
                "parent_shard_path": shard["path"],
                "parent_local_chunk_idx": local_chunk_idx,
                "source": source,
                "source_token_offset": source_offset,
                "owner_doc_id": doc_id,
            }
            for alias in DIFFICULTY_METRICS:
                rank = ranks.get(alias, {}).get(doc_id)
                if rank is None:
                    raise SystemExit(
                        f"parent chunk {global_chunk_idx} owner {doc_id!r} lacks "
                        f"a finite {alias} score; refusing a partial order"
                    )
                row[f"difficulty_rank_{alias}"] = int(rank)
            chunk_rows.append(row)
            global_chunk_idx += 1

    if not chunk_rows:
        raise SystemExit("parent layout exposes zero train chunks")
    with gzip.open(out_dir / "parent_chunk_index.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in chunk_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    for alias in DIFFICULTY_METRICS:
        ordered = sorted(
            chunk_rows,
            key=lambda row: (
                int(row[f"difficulty_rank_{alias}"]),
                int(row["global_chunk_idx"]),
            ),
        )
        arr = np.asarray([row["global_chunk_idx"] for row in ordered], dtype=np.uint32)
        expected = np.arange(len(chunk_rows), dtype=np.uint32)
        if not np.array_equal(np.sort(arr), expected):
            raise SystemExit(f"internal error: {alias} order is not a parent permutation")
        np.save(out_dir / f"ranked_chunks_{alias}.npy", arr)

    coordinate_payload = [
        {
            "path": s["path"],
            "count": s["count"],
            "source": s["source"],
            "source_token_start": s["source_token_start"],
            "n_chunks": s["n_chunks"],
        }
        for s in parent_layout["shards"]
    ]
    coordinate_sha256 = hashlib.sha256(
        json.dumps(coordinate_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "coordinate_model": "parent_pool_flat_chunks_v1",
        "chunk_owner": "document_containing_chunk_start",
        "seq_len": int(seq_len),
        "eos_tokens_per_doc": int(eos_tokens_per_doc),
        "n_chunks": len(chunk_rows),
        "n_parent_shards": len(parent_layout["shards"]),
        "coordinate_sha256": coordinate_sha256,
    }


def planned_upload_keys(out_dir: Path, dst_uri: str) -> List[str]:
    """List relative files under out_dir as would-be S3 keys (dry-run)."""
    base = dst_uri.rstrip("/") + "/"
    keys: List[str] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_dir).as_posix()
            keys.append(base + rel)
    return keys


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels-root",
        type=Path,
        required=True,
        help="Heuristic labels root (contains metrics_index.jsonl.gz + docs/)",
    )
    ap.add_argument(
        "--lm-labels-root",
        type=Path,
        required=True,
        help="LM labels root (contains metrics_index.jsonl.gz + docs/)",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Local staging directory")
    ap.add_argument(
        "--parent-layout",
        type=Path,
        required=True,
        help="Captured published-parent layout in dataset_paths() shard order",
    )
    ap.add_argument("--parent-dataset-id", default=DEFAULT_PARENT_DATASET_ID)
    ap.add_argument("--parent-version", required=True)
    ap.add_argument("--parent-manifest-sha256", required=True)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument(
        "--dst-uri",
        default=DEFAULT_DST_URI,
        help="Target S3 URI for documentation / dry-run listing",
    )
    ap.add_argument(
        "--dry-run-upload",
        action="store_true",
        help="Print planned S3 keys after local write; no network",
    )
    ap.add_argument(
        "--upload",
        action="store_true",
        help="Request upload (requires --i-understand-s3-mutation); still no AWS in this script",
    )
    ap.add_argument(
        "--i-understand-s3-mutation",
        action="store_true",
        help="Acknowledge that upload would mutate cloud state",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    heur_index = args.labels_root / "metrics_index.jsonl.gz"
    lm_index = args.lm_labels_root / "metrics_index.jsonl.gz"
    if not heur_index.is_file():
        raise SystemExit(f"missing heuristic metrics index: {heur_index}")
    if not lm_index.is_file():
        raise SystemExit(f"missing LM metrics index: {lm_index}")

    log.info("loading heuristic index %s", heur_index)
    heuristic = load_metrics_index(heur_index)
    log.info("loading LM index %s", lm_index)
    lm = load_metrics_index(lm_index)
    joined, coverage = join_label_indexes(heuristic, lm)
    log.info("joined %d docs (coverage=%s)", len(joined), coverage)
    if not joined:
        raise SystemExit("label join is empty")
    ranks = assign_ranks(joined)

    write_doc_manifest(out_dir / "doc_manifest.jsonl.gz", joined, ranks)
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    parent_layout = load_parent_layout(
        path=args.parent_layout,
        dataset_id=args.parent_dataset_id,
        version=args.parent_version,
        manifest_sha256=args.parent_manifest_sha256,
        seq_len=int(args.seq_len),
    )
    parent_stats = build_parent_pool_orders(
        joined_rows=joined,
        ranks=ranks,
        parent_layout=parent_layout,
        out_dir=out_dir,
        seq_len=int(args.seq_len),
    )

    manifest = {
        "version": 2,
        "dst_uri": args.dst_uri,
        "coverage": coverage,
        "metrics": list(DIFFICULTY_METRICS),
        "metric_sort": {k: {"column": v[0], "reverse": v[1]} for k, v in METRIC_SORT.items()},
        "parent": {
            "dataset_id": args.parent_dataset_id,
            "version": args.parent_version,
            "manifest_sha256": args.parent_manifest_sha256,
            **parent_stats,
        },
        "n_ranked": {alias: len(ranks[alias]) for alias in DIFFICULTY_METRICS},
    }
    (out_dir / "curriculum_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log.info("wrote curriculum index → %s", out_dir)

    if args.dry_run_upload or args.upload:
        keys = planned_upload_keys(out_dir, args.dst_uri)
        print(json.dumps({"dry_run": True, "n_objects": len(keys), "objects": keys[:50], "truncated": len(keys) > 50}, indent=2))
        if args.upload:
            if not args.i_understand_s3_mutation:
                raise SystemExit(
                    "--upload requires --i-understand-s3-mutation; "
                    "this script does not perform AWS uploads — stage locally and use an approved uploader"
                )
            raise SystemExit(
                "Refusing live S3 mutation from build_curriculum_index.py. "
                "Artifacts are ready under --out-dir; upload via an explicitly authorized path."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
