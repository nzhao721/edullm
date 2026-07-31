"""Step-law curve points must not include the step-1451 final anchor."""
from __future__ import annotations

import sys
from pathlib import Path

_MIXLAW = Path(__file__).resolve().parents[1]
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

from extrapolate_chinchilla import _curve_points  # noqa: E402


def test_curve_points_exclude_final_anchor() -> None:
    run = {
        "curve": [
            {"step": 120, "task_loss_bpb": {"arc_challenge_val_rc_5shot_bpb": 2.0}},
            {"step": 1440, "task_loss_bpb": {"arc_challenge_val_rc_5shot_bpb": 1.5}},
        ],
        "task_loss_labels": {"arc_challenge_val_rc_5shot_bpb": 3.0},
    }
    per_family = _curve_points(run)
    steps = [s for s, _ in per_family["arc_challenge"]]
    assert steps == [120, 1440]
    assert 1451 not in steps
