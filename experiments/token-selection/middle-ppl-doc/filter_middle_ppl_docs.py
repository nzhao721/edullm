#!/usr/bin/env python3
"""Offline Middle-PPL document filter for RegMix LM labels.

Ranks documents by late-RefHQ ``avg_perplexity`` and keeps the middle 60% of
**tokens** (token-weighted): drop the easiest 20% and hardest 20% of token mass.

Requires finalized LM labels (``metrics_index.jsonl.gz`` + ``READY``):

  datasets/regmix/label_regmix_doc_lm.py
  datasets/regmix/finalize_regmix_lm_labels.py

Writes under ``--out-dir``:
  keep_ids.txt
  keep_manifest.jsonl.gz   (metrics rows for kept docs, ranked by PPL)
  drop_manifest.jsonl.gz   (metrics rows for dropped docs)
  filter_summary.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple


METRIC = "avg_perplexity"
DEFAULT_KEEP_FRAC = 0.6


@dataclass(frozen=True)
class DocRow:
    doc_id: str
    score: float
    n_tokens: int
    domain: str
    raw: dict[str, Any]


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_metrics_index(index_path: Path) -> Iterator[dict[str, Any]]:
    with open_maybe_gzip(index_path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _token_weight(obj: dict[str, Any]) -> int:
    # Prefer loss-token mass (matches learnability-doc / LM label semantics).
    for key in ("n_loss_tokens", "n_tokens"):
        val = obj.get(key)
        if val is None:
            continue
        n = int(val)
        if n > 0:
            return n
    return 0


def _score(obj: dict[str, Any], metric: str) -> Optional[float]:
    val = obj.get(metric)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def load_scored_rows(index_path: Path, metric: str = METRIC) -> List[DocRow]:
    rows: List[DocRow] = []
    for obj in iter_metrics_index(index_path):
        doc_id = obj.get("id")
        if not doc_id:
            continue
        score = _score(obj, metric)
        if score is None:
            continue
        n_tokens = _token_weight(obj)
        if n_tokens <= 0:
            continue
        rows.append(
            DocRow(
                doc_id=str(doc_id),
                score=score,
                n_tokens=n_tokens,
                domain=str(obj.get("domain") or "unknown"),
                raw=obj,
            )
        )
    return rows


def select_middle_token_mass(
    rows: Sequence[DocRow],
    keep_frac: float = DEFAULT_KEEP_FRAC,
) -> Tuple[List[DocRow], List[DocRow], dict[str, Any]]:
    """Keep the middle ``keep_frac`` of token mass by ascending score.

    Documents are treated as atomic. A doc is kept iff the midpoint of its
    cumulative token interval falls in ``[drop_tail, 1 - drop_tail)`` of the
    total token mass, where ``drop_tail = (1 - keep_frac) / 2``.

    For ``keep_frac=0.6`` this drops the easiest and hardest 20% of tokens each.
    """
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")
    ordered = sorted(rows, key=lambda r: (r.score, r.doc_id))
    total = sum(r.n_tokens for r in ordered)
    if total <= 0:
        raise SystemExit("no scored tokens in metrics index")

    drop_tail = (1.0 - keep_frac) / 2.0
    lo = drop_tail * total
    hi = (1.0 - drop_tail) * total

    kept: List[DocRow] = []
    dropped: List[DocRow] = []
    cum = 0
    for row in ordered:
        start = cum
        end = cum + row.n_tokens
        cum = end
        mid = 0.5 * (start + end)
        if lo <= mid < hi:
            kept.append(row)
        else:
            dropped.append(row)

    kept_tokens = sum(r.n_tokens for r in kept)
    dropped_tokens = sum(r.n_tokens for r in dropped)
    summary = {
        "metric": None,
        "keep_frac": keep_frac,
        "drop_tail_frac": drop_tail,
        "selection": "middle_token_mass_by_midpoint",
        "n_docs_scored": len(ordered),
        "n_docs_kept": len(kept),
        "n_docs_dropped": len(dropped),
        "tokens_scored": total,
        "tokens_kept": kept_tokens,
        "tokens_dropped": dropped_tokens,
        "tokens_kept_frac": (kept_tokens / total) if total else 0.0,
        "band_lo_tokens": lo,
        "band_hi_tokens": hi,
        "score_min_kept": kept[0].score if kept else None,
        "score_max_kept": kept[-1].score if kept else None,
        "score_min_all": ordered[0].score if ordered else None,
        "score_max_all": ordered[-1].score if ordered else None,
    }
    return kept, dropped, summary


def write_manifest(path: Path, rows: Iterable[DocRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row.raw)
            payload["filter_score"] = row.score
            payload["filter_n_tokens"] = row.n_tokens
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def require_labels_ready(labels_root: Path, *, allow_incomplete: bool) -> Path:
    index_path = labels_root / "metrics_index.jsonl.gz"
    ready_path = labels_root / "READY"
    if not index_path.is_file():
        raise SystemExit(
            f"missing {index_path}; run finalize_regmix_lm_labels.py first"
        )
    if not ready_path.is_file() and not allow_incomplete:
        raise SystemExit(
            f"missing {ready_path}; LM labeling is not finalized. "
            "Pass --allow-incomplete only for debugging partial indexes."
        )
    return index_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels-root",
        type=Path,
        required=True,
        help="Root of finalized RegMix LM labels (contains metrics_index.jsonl.gz)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keep-frac", type=float, default=DEFAULT_KEEP_FRAC)
    ap.add_argument(
        "--metric",
        default=METRIC,
        help="Score column (default: avg_perplexity = late RefHQ)",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow running without READY (partial metrics_index)",
    )
    args = ap.parse_args()

    index_path = require_labels_ready(args.labels_root, allow_incomplete=args.allow_incomplete)
    rows = load_scored_rows(index_path, metric=args.metric)
    if not rows:
        raise SystemExit(f"no usable rows with metric={args.metric} in {index_path}")

    kept, dropped, summary = select_middle_token_mass(rows, keep_frac=args.keep_frac)
    summary["metric"] = args.metric
    summary["labels_root"] = str(args.labels_root.resolve())
    summary["metrics_index"] = str(index_path.resolve())
    summary["ready"] = (args.labels_root / "READY").is_file()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "keep_ids.txt").write_text(
        "\n".join(r.doc_id for r in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    write_manifest(out / "keep_manifest.jsonl.gz", kept)
    write_manifest(out / "drop_manifest.jsonl.gz", dropped)
    (out / "filter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
