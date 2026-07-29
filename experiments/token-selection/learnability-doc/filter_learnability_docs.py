#!/usr/bin/env python3
"""Offline document filter: keep top-γ tokens by early→late learnability.

Polarity
--------
RegMix LM labels store ``learnability_late_minus_early_avg_nll`` = late − early.
Improvement (learnability) is the opposite sign::

    improvement = early − late = −learnability_late_minus_early_avg_nll

Docs are ranked by **largest improvement** (most negative late−early). We keep
the top ``--keep-token-fraction`` (default 0.6) of **tokens** (token-weighted
by ``n_loss_tokens``, falling back to ``n_tokens``).

Inputs (from ``datasets/regmix/finalize_regmix_lm_labels.py``)::

    <labels-root>/metrics_index.jsonl.gz
    <labels-root>/READY   (required unless --allow-incomplete)

Outputs under ``--out-dir``::

    filter_manifest.json
    kept_ids.jsonl.gz          # one metrics row per kept doc (+ score fields)
    kept_ids.txt               # doc ids only
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterator


METRIC_KEY = "learnability_late_minus_early_avg_nll"
DEFAULT_KEEP_FRACTION = 0.6


def open_maybe_gzip(path: Path, mode: str = "rt"):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def doc_token_weight(row: dict[str, Any]) -> int:
    for key in ("n_loss_tokens", "n_tokens"):
        val = row.get(key)
        if val is None:
            continue
        try:
            n = int(val)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return 0


def improvement_score(row: dict[str, Any]) -> float | None:
    """Return early−late improvement; None if metric missing/non-finite."""
    raw = row.get(METRIC_KEY)
    if raw is None:
        return None
    try:
        late_minus_early = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(late_minus_early):
        return None
    return -late_minus_early


def iter_metric_rows(index_path: Path) -> Iterator[dict[str, Any]]:
    with open_maybe_gzip(index_path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def select_top_token_fraction(
    rows: list[dict[str, Any]],
    *,
    keep_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank by improvement desc; keep cumulative tokens until fraction reached."""
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")

    scored: list[tuple[float, int, dict[str, Any]]] = []
    skipped = 0
    for row in rows:
        score = improvement_score(row)
        weight = doc_token_weight(row)
        if score is None or weight <= 0 or not row.get("id"):
            skipped += 1
            continue
        scored.append((score, weight, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    total_tokens = sum(w for _, w, _ in scored)
    target = int(math.ceil(total_tokens * keep_fraction)) if total_tokens else 0

    kept: list[dict[str, Any]] = []
    kept_tokens = 0
    for score, weight, row in scored:
        out = dict(row)
        out["improvement_early_minus_late"] = score
        out["filter_token_weight"] = weight
        kept.append(out)
        kept_tokens += weight
        if kept_tokens >= target and target > 0:
            break

    stats = {
        "metric_stored": METRIC_KEY,
        "rank_by": "improvement_early_minus_late = -learnability_late_minus_early_avg_nll",
        "polarity": "keep largest improvements (top by early-late)",
        "keep_token_fraction": keep_fraction,
        "n_docs_scored": len(scored),
        "n_docs_skipped": skipped,
        "n_docs_kept": len(kept),
        "total_tokens_scored": total_tokens,
        "target_tokens": target,
        "kept_tokens": kept_tokens,
        "kept_token_fraction_realized": (
            (kept_tokens / total_tokens) if total_tokens else 0.0
        ),
    }
    return kept, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--labels-root",
        type=Path,
        required=True,
        help="RegMix LM labels root (metrics_index.jsonl.gz + docs/)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--keep-token-fraction",
        type=float,
        default=DEFAULT_KEEP_FRACTION,
        help="Fraction of scored tokens to keep (default 0.6)",
    )
    ap.add_argument(
        "--metrics-index",
        type=Path,
        default=None,
        help="Override path to metrics_index.jsonl.gz",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow running without READY (partial metrics_index; debug only)",
    )
    ap.add_argument(
        "--require-ready",
        dest="allow_incomplete",
        action="store_false",
        help=argparse.SUPPRESS,  # legacy alias; READY is the default
    )
    args = ap.parse_args()

    labels_root = args.labels_root
    index_path = args.metrics_index or (labels_root / "metrics_index.jsonl.gz")
    ready_path = labels_root / "READY"

    if not index_path.is_file():
        raise SystemExit(
            f"missing metrics index: {index_path}\n"
            "Label pipeline: datasets/regmix/submit_regmix_doc_lm_labeling.sh → "
            "finalize_regmix_lm_labels.py"
        )
    if not ready_path.is_file() and not args.allow_incomplete:
        raise SystemExit(
            f"missing READY marker at {ready_path}; "
            "wait for datasets/regmix/finalize_regmix_lm_labels.py "
            "(pass --allow-incomplete only for debugging partial indexes)"
        )

    rows = list(iter_metric_rows(index_path))
    kept, stats = select_top_token_fraction(rows, keep_fraction=float(args.keep_token_fraction))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_gz = out_dir / "kept_ids.jsonl.gz"
    kept_txt = out_dir / "kept_ids.txt"
    with gzip.open(kept_gz, "wt", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kept_txt.write_text("".join(f"{row['id']}\n" for row in kept), encoding="utf-8")

    manifest = {
        "arm": "learnability-doc",
        "labels_root": str(labels_root.resolve()),
        "metrics_index": str(index_path.resolve()),
        "ready_present": ready_path.is_file(),
        "kept_ids_jsonl_gz": str(kept_gz.resolve()),
        "kept_ids_txt": str(kept_txt.resolve()),
        **stats,
    }
    (out_dir / "filter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
