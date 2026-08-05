#!/usr/bin/env python3
"""Run Dolma English tag+mix on one docs shard (Slurm array over english_tasks.txt).

Task line: ``source domain documents-NNNNN.jsonl.gz``
Copies docs → out, then filters in place under
out/<source>/<domain>/documents/<shard>.

The ``documents/`` path segment is required: Dolma derives attribute paths by
substituting ``.../documents/...`` → ``.../attributes/...``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from refhq_new.process import apply_dolma_english_filter  # noqa: E402


def _load_task(tasks_file: Path, task_index: int) -> tuple[str, str, str]:
    lines = [ln.strip() for ln in tasks_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if task_index < 0 or task_index >= len(lines):
        raise SystemExit(f"task_index {task_index} out of range (n={len(lines)})")
    parts = lines[task_index].split()
    if len(parts) != 3:
        raise SystemExit(f"bad english task line: {lines[task_index]!r}")
    return parts[0], parts[1], parts[2]


def filter_one_shard(
    *,
    source: str,
    domain: str,
    shard_name: str,
    docs_root: Path,
    out_root: Path,
    work_root: Path,
    processes: int,
    english_score_threshold: float,
) -> dict:
    src_shard = docs_root / domain / shard_name
    if not src_shard.is_file():
        raise SystemExit(f"missing docs shard: {src_shard}")

    # Dolma requires a ".../documents/..." path segment (not just a filename).
    dest_dir = out_root / domain / "documents"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_shard = dest_dir / shard_name
    shutil.copy2(src_shard, dest_shard)

    shard_work = work_root / domain / Path(shard_name).stem
    if shard_work.exists():
        shutil.rmtree(shard_work)
    shard_work.mkdir(parents=True, exist_ok=True)

    print(
        f"english-filter source={source} domain={domain} shard={shard_name} "
        f"processes={processes}",
        flush=True,
    )
    n_docs = apply_dolma_english_filter(
        shard_path=dest_shard,
        work_root=shard_work,
        processes=processes,
        domain="english",
        english_score_threshold=english_score_threshold,
    )
    stats = {
        "source": source,
        "domain": domain,
        "shard": shard_name,
        "documents": n_docs,
        "out_shard": str(dest_shard),
        "english_score_threshold": english_score_threshold,
        "processes": processes,
    }
    stats_path = (
        work_root.parent / "english-stats" / source / domain / f"{Path(shard_name).stem}.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--shard", default=None, help="documents-NNNNN.jsonl.gz")
    parser.add_argument(
        "--processes",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    parser.add_argument("--english-score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    scratch = Path(plan["scratch_root"])
    tasks_file = args.tasks_file or (scratch / "manifests" / "english_tasks.txt")

    if args.task_index is not None:
        source, domain, shard_name = _load_task(tasks_file, args.task_index)
    else:
        if not (args.source and args.domain and args.shard):
            raise SystemExit("provide --task-index or --source --domain --shard")
        source, domain, shard_name = args.source, args.domain, args.shard

    if source not in plan["sources"]:
        raise SystemExit(f"source {source} missing from plan")
    src_plan = plan["sources"][source]

    filter_one_shard(
        source=source,
        domain=domain,
        shard_name=shard_name,
        docs_root=Path(src_plan["paths"]["docs"]),
        out_root=Path(src_plan["paths"]["out"]),
        work_root=Path(src_plan["paths"]["work"]) / "english",
        processes=max(1, args.processes),
        english_score_threshold=args.english_score_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
