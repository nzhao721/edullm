"""pytest path setup for refhq tests."""

from __future__ import annotations

import sys
from pathlib import Path

REFHQ_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REFHQ_ROOT.parents[1]

for entry in (
    REPO_ROOT / "datasets",
    REFHQ_ROOT / "scripts",
):
    path = str(entry)
    if path not in sys.path:
        sys.path.insert(0, path)
