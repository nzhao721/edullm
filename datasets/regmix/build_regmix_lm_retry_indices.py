#!/usr/bin/env python3
"""List work-manifest chunk indices that lack completed LM label shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def shard_stem(index: int, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"lm-{index:05d}-{digest}"


def load_index_set(path: Path | None) -> set[int]:
    if path is None or not path.exists():
        return set()
    return {
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--failed-indices",
        type=Path,
        default=None,
        help="Chunk indices to deprioritize (retried at end of queue, not skipped).",
    )
    args = parser.parse_args()

    deprioritized = load_index_set(args.failed_indices)

    rows = [
        json.loads(line)
        for line in args.work_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing: list[int] = []
    for item in rows:
        index = int(item["index"])
        domain = item["domain"]
        stem = shard_stem(index, item["path"])
        done = args.labels_root / "docs" / domain / f"{stem}.done"
        if not done.exists():
            missing.append(index)

    priority = [index for index in missing if index not in deprioritized]
    retry_tail = [index for index in missing if index in deprioritized]
    ordered = priority + retry_tail

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(i) for i in ordered) + ("\n" if ordered else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "retry_indices_ready",
                "n_missing": len(ordered),
                "n_priority": len(priority),
                "n_failed_deprioritized": len(retry_tail),
                "n_total": len(rows),
                "out": str(args.out),
                "sample": ordered[:20],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
