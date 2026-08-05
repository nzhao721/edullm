#!/usr/bin/env python3
"""Carve 0.15% document holdout per (source, domain) BEFORE tokenize.

Reads out/<source>/<domain>/documents/documents-*.jsonl.gz (post-English;
also accepts flat out/<source>/<domain>/documents-*.jsonl.gz) and writes:
  holdout/<source>/<domain>/train/documents-*.jsonl.gz
  holdout/<source>/<domain>/val/documents-*.jsonl.gz

Also refreshes manifests/tokenize_tasks.txt with real non-empty (source, domain, split) rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from olmo_shard_utils import iter_docs  # noqa: E402
from refhq_new_sources import DOCS_PER_SHARD, HOLDOUT_FRACTION, holdout_counts  # noqa: E402


def _pair_seed(base_seed: int, source: str, domain: str) -> int:
    """Deterministic per-(source, domain) seed (stable across processes)."""
    digest = hashlib.sha256(f"{base_seed}:{source}:{domain}".encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def _list_shards(domain_dir: Path) -> list[Path]:
    # Prefer Dolma layout (.../documents/*.jsonl.gz); fall back to flat.
    for base in (domain_dir / "documents", domain_dir):
        if not base.is_dir():
            continue
        shards = sorted(base.glob("documents-*.jsonl.gz"))
        if not shards:
            shards = sorted(base.glob("documents-*.json.gz"))
        if shards:
            return shards
    return []


def _iter_shard_docs(shards: list[Path]) -> Iterator[dict[str, Any]]:
    for shard in shards:
        yield from iter_docs(shard)


def _write_split(
    docs: list[dict[str, Any]],
    out_dir: Path,
    *,
    max_per_shard: int = DOCS_PER_SHARD,
) -> list[str]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not docs:
        return []
    paths: list[str] = []
    handle = None
    n = 0
    shard_idx = 0
    try:
        for doc in docs:
            if handle is None or n >= max_per_shard:
                if handle is not None:
                    handle.close()
                path = out_dir / f"documents-{shard_idx:05d}.jsonl.gz"
                handle = gzip.open(path, "wt", encoding="utf-8")
                paths.append(str(path))
                shard_idx += 1
                n = 0
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    finally:
        if handle is not None:
            handle.close()
    return paths


def holdout_domain(
    *,
    shards: list[Path],
    holdout_domain_dir: Path,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    docs = list(_iter_shard_docs(shards))
    n_docs = len(docs)
    n_train, n_val = holdout_counts(n_docs, fraction)
    rng = random.Random(seed)
    order = list(range(n_docs))
    rng.shuffle(order)
    val_idx = set(order[:n_val])
    train_docs = [docs[i] for i in range(n_docs) if i not in val_idx]
    val_docs = [docs[i] for i in range(n_docs) if i in val_idx]

    train_paths = _write_split(train_docs, holdout_domain_dir / "train")
    val_paths = _write_split(val_docs, holdout_domain_dir / "val")
    return {
        "n_docs": n_docs,
        "n_train": len(train_docs),
        "n_val": len(val_docs),
        "fraction": fraction,
        "seed": seed,
        "train_shards": train_paths,
        "val_shards": val_paths,
    }


def run_holdout(plan: dict[str, Any]) -> dict[str, Any]:
    scratch = Path(plan["scratch_root"])
    fraction = float(plan.get("holdout_fraction", HOLDOUT_FRACTION))
    base_seed = int(plan.get("seed", 42))
    summary: dict[str, Any] = {
        "holdout_fraction": fraction,
        "seed": base_seed,
        "pairs": {},
    }
    task_lines: list[str] = []

    for source, src_plan in plan["sources"].items():
        out_root = Path(src_plan["paths"]["out"])
        holdout_root = Path(src_plan["paths"]["holdout"])
        if holdout_root.exists():
            shutil.rmtree(holdout_root)
        holdout_root.mkdir(parents=True, exist_ok=True)

        if not out_root.is_dir():
            print(f"skip source={source}: missing out dir {out_root}", flush=True)
            continue

        domain_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
        for domain_dir in domain_dirs:
            domain = domain_dir.name
            shards = _list_shards(domain_dir)
            if not shards:
                continue
            pair_seed = _pair_seed(base_seed, source, domain)
            print(
                f"holdout source={source} domain={domain} shards={len(shards)} "
                f"fraction={fraction} seed={pair_seed}",
                flush=True,
            )
            stats = holdout_domain(
                shards=shards,
                holdout_domain_dir=holdout_root / domain,
                fraction=fraction,
                seed=pair_seed,
            )
            summary["pairs"][f"{source}/{domain}"] = stats
            # One tokenize Slurm task per holdout document shard (max parallelization).
            for split_name, key in (("train", "train_shards"), ("val", "val_shards")):
                for shard_path in stats.get(key) or []:
                    shard_name = Path(shard_path).name
                    task_lines.append(f"{source} {domain} {split_name} {shard_name}")
            print(
                f"holdout done {source}/{domain} train={stats['n_train']} val={stats['n_val']} "
                f"train_shards={len(stats.get('train_shards') or [])} "
                f"val_shards={len(stats.get('val_shards') or [])}",
                flush=True,
            )

    tasks_path = scratch / "manifests" / "tokenize_tasks.txt"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    summary["tokenize_tasks"] = str(tasks_path)
    summary["tokenize_task_count"] = len(task_lines)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = run_holdout(plan)
    out_path = Path(plan["scratch_root"]) / "manifests" / "holdout_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
