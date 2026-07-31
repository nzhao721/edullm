#!/usr/bin/env python3
"""Unit tests for learnability-doc filter polarity (no GPU / labels required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_learnability_docs import (  # noqa: E402
    improvement_score,
    select_top_token_fraction,
)


class TestLearnabilityFilterPolarity(unittest.TestCase):
    def test_improvement_is_negation_of_stored_metric(self) -> None:
        row = {"learnability_late_minus_early_avg_nll": -0.5, "id": "a", "n_loss_tokens": 10}
        self.assertAlmostEqual(improvement_score(row), 0.5)

    def test_keeps_largest_improvements_token_weighted(self) -> None:
        # A improved most, B medium, C worst (late worse than early).
        rows = [
            {
                "id": "C",
                "learnability_late_minus_early_avg_nll": 1.0,  # improvement -1
                "n_loss_tokens": 40,
            },
            {
                "id": "A",
                "learnability_late_minus_early_avg_nll": -2.0,  # improvement +2
                "n_loss_tokens": 30,
            },
            {
                "id": "B",
                "learnability_late_minus_early_avg_nll": -0.5,  # improvement +0.5
                "n_loss_tokens": 30,
            },
        ]
        # Total 100 tokens; keep 60% → 60 tokens. Order A(30)+B(30)=60; C dropped.
        kept, stats = select_top_token_fraction(rows, keep_fraction=0.6)
        ids = [r["id"] for r in kept]
        self.assertEqual(ids, ["A", "B"])
        self.assertEqual(stats["kept_tokens"], 60)
        self.assertNotIn("C", ids)

    def test_2360_ladder_omits_2250(self) -> None:
        ts_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(ts_root))
        from token_selection.olmo_ext.checkpoint_ladder import permanent_checkpoint_steps

        steps = permanent_checkpoint_steps(2360, 125)
        self.assertIn(0, steps)
        self.assertIn(2125, steps)
        self.assertIn(2360, steps)
        self.assertNotIn(2250, steps)

    def test_ready_required_by_default(self) -> None:
        import tempfile
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # metrics_index present but no READY
            (root / "metrics_index.jsonl.gz").write_bytes(b"")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "filter_learnability_docs.py"),
                    "--labels-root",
                    str(root),
                    "--out-dir",
                    str(root / "out"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("READY", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
