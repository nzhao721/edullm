"""Shared import path setup for Dolma HQ FarmShare scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_paths() -> tuple[Path, Path]:
    """Insert repo, FarmShare shared utils, and HQ script paths on sys.path."""

    refhq_root = Path(__file__).resolve().parents[1]
    repo_root = refhq_root.parents[1]  # datasets/refhq -> edullm root
    datasets_dir = repo_root / "datasets"
    scripts_dir = refhq_root / "scripts"
    for entry in (datasets_dir, scripts_dir):
        path = str(entry)
        if path not in sys.path:
            sys.path.insert(0, path)
    return refhq_root, repo_root
