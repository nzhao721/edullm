#!/usr/bin/env python3
"""Download Proof-Pile-2, Lean4-Mathlib, and arXiv metadata to scratch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PP2_REPO = "EleutherAI/proof-pile-2"
LEAN4_REPO = "phanerozoic/Lean4-Mathlib"
ARXIV_META_REPO = "jackkuo/arXiv-metadata-oai-snapshot"


def _download(
    repo_id: str,
    local_dir: Path,
    *,
    repo_type: str = "dataset",
    allow_patterns: list[str] | None = None,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {local_dir}", flush=True)
    kwargs: dict = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "local_dir": str(local_dir),
        "max_workers": int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    path = snapshot_download(**kwargs)
    print(f"[download] done {repo_id}: {path}", flush=True)
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch-root", type=Path, required=True)
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional: pp2 lean4 arxiv-meta",
    )
    p.add_argument(
        "--pp2-splits",
        nargs="*",
        default=["train"],
        help="PP2 splits to download (default: train only)",
    )
    args = p.parse_args(argv)

    root = args.scratch_root.resolve()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {"pp2", "lean4", "arxiv-meta"}
    summary: dict[str, object] = {
        "scratch_root": str(root),
        "pp2_splits": list(args.pp2_splits),
        "downloads": {},
    }

    if "pp2" in wanted:
        patterns: list[str] = ["README.md", "proof-pile-2.py", ".gitattributes"]
        for split in args.pp2_splits:
            for subset in ("arxiv", "open-web-math", "algebraic-stack"):
                patterns.append(f"{subset}/{split}/*")
        path = _download(PP2_REPO, raw / "proof-pile-2", allow_patterns=patterns)
        summary["downloads"]["pp2"] = str(path)

    if "lean4" in wanted:
        path = _download(LEAN4_REPO, raw / "lean4-mathlib")
        summary["downloads"]["lean4"] = str(path)

    if "arxiv-meta" in wanted:
        path = _download(ARXIV_META_REPO, raw / "arxiv-metadata")
        summary["downloads"]["arxiv-meta"] = str(path)

    out = root / "manifests" / "download_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
