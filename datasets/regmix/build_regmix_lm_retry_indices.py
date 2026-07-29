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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--failed-indices",
        type=Path,
        default=None,
        help="Optional file listing chunk indices to skip (one per line).",
    )
    args = parser.parse_args()

    failed: set[int] = set()
    if args.failed_indices is not None and args.failed_indices.exists():
        failed = {
            int(line.strip())
            for line in args.failed_indices.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

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
        if not done.exists() and index not in failed:
            missing.append(index)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(i) for i in missing) + ("\n" if missing else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "retry_indices_ready",
                "n_missing": len(missing),
                "n_failed_skipped": len(failed),
                "n_total": len(rows),
                "out": str(args.out),
                "sample": missing[:20],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
