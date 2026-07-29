#!/usr/bin/env python3
"""Unit tests for Middle-PPL document token-mass filter (no GPU)."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_middle_ppl_docs import DocRow, select_middle_token_mass


def _rows(pairs: list[tuple[str, float, int]]) -> list[DocRow]:
    out = []
    for doc_id, score, n in pairs:
        out.append(
            DocRow(
                doc_id=doc_id,
                score=score,
                n_tokens=n,
                domain="t",
                raw={"id": doc_id, "avg_perplexity": score, "n_tokens": n},
            )
        )
    return out


class TestMiddleTokenMass(unittest.TestCase):
    def test_drops_easy_and_hard_20pct(self) -> None:
        # 10 docs × 10 tokens = 100; keep middle 60 → tokens with mid in [20, 80)
        rows = _rows([(f"d{i}", float(i), 10) for i in range(10)])
        kept, dropped, summary = select_middle_token_mass(rows, keep_frac=0.6)
        kept_ids = [r.doc_id for r in kept]
        # mids: 5,15,25,...,95 → keep 25..75 → d2..d7
        self.assertEqual(kept_ids, [f"d{i}" for i in range(2, 8)])
        self.assertEqual(summary["tokens_kept"], 60)
        self.assertEqual(len(dropped), 4)

    def test_atomic_docs_near_boundaries(self) -> None:
        rows = _rows(
            [
                ("easy", 1.0, 25),
                ("mid", 2.0, 50),
                ("hard", 3.0, 25),
            ]
        )
        kept, _dropped, summary = select_middle_token_mass(rows, keep_frac=0.6)
        # mids: 12.5 (drop), 50 (keep), 87.5 (drop); band [20, 80)
        self.assertEqual([r.doc_id for r in kept], ["mid"])
        self.assertEqual(summary["tokens_kept"], 50)

    def test_ladder_2384_imported(self) -> None:
        ts = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(ts))
        from token_selection.olmo_ext.checkpoint_ladder import permanent_checkpoint_steps
        from token_selection.olmo_ext.s3_layout import arm_prefix

        steps = permanent_checkpoint_steps(2384, 125)
        self.assertIn(0, steps)
        self.assertIn(2250, steps)
        self.assertIn(2384, steps)
        self.assertNotIn(2375, steps)
        self.assertEqual(arm_prefix("middle-ppl-doc"), "token-sel/middle-ppl-doc")

    def test_trainer_arm_constants(self) -> None:
        # Import constants without pulling heavy CUDA deps at module load —
        # only validate the ARM / run-id / length defaults via source scan if
        # torch/olmo_core are unavailable.
        trainer = Path(__file__).resolve().parent / "train_ce_middle_ppl_doc.py"
        text = trainer.read_text(encoding="utf-8")
        self.assertIn('ARM = "middle-ppl-doc"', text)
        self.assertIn("edullm-370M-middle-ppl-doc-ladder125-v1", text)
        self.assertIn("DEFAULT_LENGTH_TOKENS = 10_000_000_000", text)
        self.assertIn('OLMO_ATTN_BACKEND", "torch"', text)
        self.assertIn('--save-interval"', text)
        self.assertIn('method": "plain_ce"', text)
        self.assertIn("export_arm_checkpoint(ARM, path)", text)


if __name__ == "__main__":
    unittest.main()
