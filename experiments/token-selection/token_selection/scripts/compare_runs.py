#!/usr/bin/env python3
"""Arm comparison is deferred until later test-loss evaluation.

Training now produces checkpoints and train-only metrics. Scientific comparison
of REL vs full (previously validation-loss-vs-compute) will run once a shared
test-loss / benchmark protocol exists. This entry point fails closed so callers
do not treat train metrics as a substitute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_rho_10b.yaml")
    ap.add_argument("--compute-unit", choices=["tokens", "fwd_equiv", "flops"], default="fwd_equiv")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()
    raise SystemExit(
        "compare_runs is deferred: held-out validation comparison was removed. "
        "Train both arms to produce checkpoints, then evaluate test loss later "
        "under a shared eval protocol."
    )


if __name__ == "__main__":
    main()
