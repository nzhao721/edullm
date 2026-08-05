#!/usr/bin/env python3
"""Build manifests/english_tasks.txt: one Dolma English job per docs shard.

Line format: ``source domain shard_name``
Example: ``tulu-v2 general documents-00000.jsonl.gz``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()


def build_english_tasks(plan: dict) -> list[str]:
    lines: list[str] = []
    for source, src_plan in plan["sources"].items():
        docs_root = Path(src_plan["paths"]["docs"])
        if not docs_root.is_dir():
            print(f"warn: missing docs for {source}: {docs_root}", flush=True)
            continue
        for domain_dir in sorted(p for p in docs_root.iterdir() if p.is_dir()):
            shards = sorted(domain_dir.glob("documents-*.jsonl.gz"))
            if not shards:
                shards = sorted(domain_dir.glob("documents-*.json.gz"))
            for shard in shards:
                lines.append(f"{source} {domain_dir.name} {shard.name}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to <scratch>/manifests/english_tasks.txt",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    lines = build_english_tasks(plan)
    out = args.out or (Path(plan["scratch_root"]) / "manifests" / "english_tasks.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {out} tasks={len(lines)}", flush=True)
    if not lines:
        raise SystemExit("no english tasks found under docs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
