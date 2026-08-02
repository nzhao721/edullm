#!/usr/bin/env python3
"""Pairwise linear regression between RegMix difficulty metrics."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

METRICS = {
    "compression_ratio": "compression_ratio",
    "flesch_reading_ease": "flesch_reading_ease",
    "mtld": "mtld",
    "learnability": "learnability_late_minus_early_avg_nll",
}


def iter_jsonl_gz(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_index(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for obj in iter_jsonl_gz(path):
        doc_id = obj.get("id")
        if doc_id:
            rows[str(doc_id)] = obj
    return rows


def join_rows(heuristic: Dict[str, dict], lm: Dict[str, dict]) -> List[dict]:
    joined: List[dict] = []
    for doc_id in sorted(set(heuristic) & set(lm)):
        h = heuristic[doc_id]
        l = lm[doc_id]
        row = {"id": doc_id}
        for key in METRICS.values():
            if key in h:
                row[key] = h[key]
            elif key in l:
                row[key] = l[key]
        joined.append(row)
    return joined


def pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Return slope, intercept, r_squared."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        return float("nan"), float("nan"), float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot == 0.0:
        r2 = float("nan")
    else:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - (ss_res / ss_tot)
    return slope, intercept, r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--lm-labels-root", type=Path, required=True)
    args = parser.parse_args()

    heuristic_path = args.labels_root / "metrics_index.jsonl.gz"
    lm_path = args.lm_labels_root / "metrics_index.jsonl.gz"
    heuristic = load_index(heuristic_path)
    lm = load_index(lm_path)
    joined = join_rows(heuristic, lm)

    print(f"heuristic_docs={len(heuristic)}")
    print(f"lm_docs={len(lm)}")
    print(f"joined_docs={len(joined)}")
    print()

    metric_names = list(METRICS.keys())
    for i, name_x in enumerate(metric_names):
        col_x = METRICS[name_x]
        for name_y in metric_names[i + 1 :]:
            col_y = METRICS[name_y]
            pairs = [
                (float(row[col_x]), float(row[col_y]))
                for row in joined
                if row.get(col_x) is not None
                and row.get(col_y) is not None
                and not math.isnan(float(row[col_x]))
                and not math.isnan(float(row[col_y]))
            ]
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            slope, intercept, r2 = linear_regression(xs, ys)
            r = pearson_r(xs, ys)
            print(f"{name_x} ~ {name_y}")
            print(f"  n={len(pairs)}")
            print(f"  pearson_r={r:.6f}")
            print(f"  r_squared={r2:.6f}")
            print(f"  slope={slope:.6g}")
            print(f"  intercept={intercept:.6g}")
            print()


if __name__ == "__main__":
    main()
