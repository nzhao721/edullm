#!/usr/bin/env python3
"""Build the RegMix-10B curriculum training index.

Merges heuristic ``labels/`` and LM ``lm_labels/`` metrics on doc ``id``, assigns
per-metric easy→hard ranks, optionally tokenizes labeled docs into 2048-token
chunks, and writes **local** staging artifacts. Publish the resulting token-order
curricula into ``s3://edullm-data/curriculum/regmix-*-370m`` via the
``edullm-data`` package (trainer consumes those IDs; this script does not write
legacy raw-datasets buckets).

**S3 mutations are opt-in.** Default is local staging only. ``--dry-run-upload``
prints the planned object keys without contacting AWS. ``--upload`` is refused
unless ``--i-understand-s3-mutation`` is also passed (and still requires an
external upload helper — this script never calls AWS SDKs directly).

Artifacts under ``--out-dir``:

  - ``doc_manifest.jsonl.gz`` — joined metrics per doc
  - ``coverage.json`` — join coverage stats
  - ``doc_index.json`` — ranks + token spans (when tokenized)
  - ``chunk_index.jsonl.gz`` — flat ``(doc_id, chunk_idx, global_chunk_idx, ranks…)``
  - ``ranked_chunks_<metric>.npy`` — easy→hard ``global_chunk_idx`` arrays
  - ``tokenized/<domain>/<domain>.npy`` — uint32 memmaps (optional)
  - ``paths_train.txt`` — memmap path list
  - ``curriculum_manifest.json`` — build summary
"""

from __future__ import annotations

import argparse
import gzip
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


def iter_labeled_docs(docs_roots: Sequence[Path]) -> Iterator[dict]:
    """Yield docs from the first root that contains each id (prefer earlier roots)."""
    seen: set[str] = set()
    for root in docs_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    doc_id = obj.get("id")
                    if not doc_id or doc_id in seen:
                        continue
                    seen.add(str(doc_id))
                    yield obj


class DomainWriter:
    def __init__(self, out_npy: Path, *, initial_capacity: int = 1 << 20) -> None:
        self.out_npy = out_npy
        self.tmp = out_npy.with_suffix(out_npy.suffix + ".tmp")
        if self.tmp.exists():
            self.tmp.unlink()
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = int(initial_capacity)
        self.n = 0
        self.docs = 0
        import numpy as np

        self._np = np
        self.mm = np.memmap(self.tmp, mode="w+", dtype=np.uint32, shape=(self.capacity,))

    def _grow(self, need: int) -> None:
        np = self._np
        if need <= self.capacity:
            return
        while self.capacity < need:
            self.capacity = int(self.capacity * 1.5) + (1 << 20)
        self.mm.flush()
        old = np.memmap(self.tmp, mode="r", dtype=np.uint32, shape=(self.n,))
        data = np.array(old, dtype=np.uint32)
        del old
        self.mm = np.memmap(self.tmp, mode="w+", dtype=np.uint32, shape=(self.capacity,))
        if self.n:
            self.mm[: self.n] = data

    def append_ids(self, ids: List[int]) -> int:
        """Append token ids; return start offset."""
        start = self.n
        if not ids:
            return start
        need = self.n + len(ids)
        self._grow(need)
        self.mm[self.n : need] = self._np.asarray(ids, dtype=self._np.uint32)
        self.n = need
        return start

    def finalize(self) -> dict:
        np = self._np
        self.mm.flush()
        del self.mm
        if self.n == 0:
            if self.tmp.exists():
                self.tmp.unlink()
            return {"docs": 0, "tokens": 0, "tokenized_npy": None}
        final = np.memmap(self.tmp, mode="r", dtype=np.uint32, shape=(self.n,))
        out = np.memmap(self.out_npy, mode="w+", dtype=np.uint32, shape=(self.n,))
        out[:] = final
        out.flush()
        del out, final
        self.tmp.unlink(missing_ok=True)
        meta = {
            "docs": self.docs,
            "tokens": self.n,
            "bytes": int(self.out_npy.stat().st_size),
            "tokenized_npy": str(self.out_npy.resolve()),
            "tokenizer": TOKENIZER_ID,
            "eos_token_id": EOS_TOKEN_ID,
        }
        self.out_npy.with_suffix(".json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return meta


def count_doc_chunks(n_tokens: int, seq_len: int) -> int:
    """Number of non-overlapping ``seq_len`` chunks in ``n_tokens`` tokens.

    Matches ``CurriculumIndexedDataset``, which reads exactly ``chunk_size``
    tokens per chunk (no +1 next-token reserve).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    return max(0, int(n_tokens) // int(seq_len))


def tokenize_and_index(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    ranks: Mapping[str, Mapping[str, int]],
    docs_roots: Sequence[Path],
    out_dir: Path,
    tokenizer_id: str = TOKENIZER_ID,
    eos_token_id: int = EOS_TOKEN_ID,
    seq_len: int = SEQ_LEN,
    skip_tokenize: bool = False,
) -> dict:
    """Write tokenized memmaps + doc/chunk indexes. Returns summary stats."""
    import numpy as np

    wanted = {str(r["id"]) for r in joined_rows}
    doc_meta_by_id = {str(r["id"]): r for r in joined_rows}

    if skip_tokenize:
        # Rank-only index: one synthetic chunk per doc (global_chunk_idx = rank order
        # is not defined yet — emit doc_index without token spans).
        doc_index = []
        for row in joined_rows:
            doc_id = str(row["id"])
            entry = {
                "id": doc_id,
                "domain": row.get("domain"),
                "n_chars": row.get("n_chars"),
            }
            for alias in DIFFICULTY_METRICS:
                if doc_id in ranks.get(alias, {}):
                    entry[f"difficulty_rank_{alias}"] = ranks[alias][doc_id]
            doc_index.append(entry)
        (out_dir / "doc_index.json").write_text(
            json.dumps(doc_index) + "\n", encoding="utf-8"
        )
        return {
            "skip_tokenize": True,
            "n_docs": len(doc_index),
            "n_chunks": 0,
            "n_tokens": 0,
        }

    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "transformers is required for tokenization; pass --skip-tokenize for ranks-only"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    writers: Dict[str, DomainWriter] = {}
    chunk_rows: List[dict] = []
    doc_index: List[dict] = []
    global_chunk_idx = 0
    total_tokens = 0

    for obj in iter_labeled_docs(docs_roots):
        doc_id = str(obj.get("id") or "")
        if doc_id not in wanted:
            continue
        text = obj.get("text") or obj.get("content") or ""
        if not isinstance(text, str) or not text:
            continue
        domain = str(obj.get("domain") or doc_meta_by_id[doc_id].get("domain") or "unknown")
        if domain not in writers:
            npy = out_dir / "tokenized" / domain / f"{domain}.npy"
            writers[domain] = DomainWriter(npy)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ids = list(ids) + [int(eos_token_id)]
        start = writers[domain].append_ids(ids)
        writers[domain].docs += 1
        end = start + len(ids)
        n_chunks = count_doc_chunks(len(ids), seq_len)
        entry = {
            "id": doc_id,
            "domain": domain,
            "token_start": start,
            "token_end": end,
            "n_tokens": len(ids),
            "n_chunks": n_chunks,
            "memmap": f"tokenized/{domain}/{domain}.npy",
        }
        for alias in DIFFICULTY_METRICS:
            if doc_id in ranks.get(alias, {}):
                entry[f"difficulty_rank_{alias}"] = ranks[alias][doc_id]
        doc_index.append(entry)
        for c in range(n_chunks):
            crow = {
                "doc_id": doc_id,
                "domain": domain,
                "chunk_idx": c,
                "global_chunk_idx": global_chunk_idx,
                "memmap": entry["memmap"],
                "token_offset": start + c * seq_len,
            }
            for alias in DIFFICULTY_METRICS:
                if doc_id in ranks.get(alias, {}):
                    crow[f"difficulty_rank_{alias}"] = ranks[alias][doc_id]
            chunk_rows.append(crow)
            global_chunk_idx += 1
        total_tokens += len(ids)

    domain_stats = {d: w.finalize() for d, w in writers.items()}
    (out_dir / "doc_index.json").write_text(json.dumps(doc_index) + "\n", encoding="utf-8")
    chunk_path = out_dir / "chunk_index.jsonl.gz"
    with gzip.open(chunk_path, "wt", encoding="utf-8") as handle:
        for crow in chunk_rows:
            handle.write(json.dumps(crow, ensure_ascii=False) + "\n")

    # Per-metric ranked global_chunk_idx arrays (easy → hard).
    for alias in DIFFICULTY_METRICS:
        scored = [
            (crow.get(f"difficulty_rank_{alias}"), crow["global_chunk_idx"])
            for crow in chunk_rows
            if crow.get(f"difficulty_rank_{alias}") is not None
        ]
        scored.sort(key=lambda t: int(t[0]))
        arr = np.asarray([g for _, g in scored], dtype=np.int64)
        np.save(out_dir / f"ranked_chunks_{alias}.npy", arr)

    paths = []
    for d, st in sorted(domain_stats.items()):
        p = st.get("tokenized_npy")
        if p:
            paths.append(p)
    (out_dir / "paths_train.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")

    return {
        "skip_tokenize": False,
        "n_docs": len(doc_index),
        "n_chunks": len(chunk_rows),
        "n_tokens": total_tokens,
        "domains": domain_stats,
        "tokenizer": tokenizer_id,
        "seq_len": seq_len,
        "eos_token_id": eos_token_id,
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
    ap.add_argument("--skip-tokenize", action="store_true")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    ap.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
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
    ranks = assign_ranks(joined)

    write_doc_manifest(out_dir / "doc_manifest.jsonl.gz", joined, ranks)
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    docs_roots = [
        args.labels_root / "docs",
        args.lm_labels_root / "docs",
    ]
    tok_stats = tokenize_and_index(
        joined_rows=joined,
        ranks=ranks,
        docs_roots=docs_roots,
        out_dir=out_dir,
        tokenizer_id=args.tokenizer,
        eos_token_id=int(args.eos_token_id),
        seq_len=int(args.seq_len),
        skip_tokenize=bool(args.skip_tokenize),
    )

    manifest = {
        "version": 1,
        "dst_uri": args.dst_uri,
        "coverage": coverage,
        "metrics": list(DIFFICULTY_METRICS),
        "metric_sort": {k: {"column": v[0], "reverse": v[1]} for k, v in METRIC_SORT.items()},
        "tokenize": tok_stats,
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
