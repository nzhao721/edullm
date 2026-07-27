"""pytest path setup for token_selection tests."""

from __future__ import annotations

import sys
from pathlib import Path

TS_ROOT = Path(__file__).resolve().parents[2]  # experiments/token-selection

path = str(TS_ROOT)
if path not in sys.path:
    sys.path.insert(0, path)
