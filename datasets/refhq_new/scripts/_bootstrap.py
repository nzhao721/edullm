"""Shared import path setup for refhq-new FarmShare scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> tuple[Path, Path]:
    """Insert datasets/ and this scripts/ dir on sys.path.

    Returns (refhq_new_root, repo_root).
    """
    refhq_new_root = Path(__file__).resolve().parents[1]
    repo_root = refhq_new_root.parents[1]  # datasets/refhq_new -> edullm root
    datasets_dir = repo_root / "datasets"
    scripts_dir = refhq_new_root / "scripts"
    for entry in (datasets_dir, scripts_dir):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    return refhq_new_root, repo_root
