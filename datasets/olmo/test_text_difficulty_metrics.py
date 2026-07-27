#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

from text_difficulty_metrics import (
    compression_ratio,
    compute_difficulty_metrics,
    flesch_reading_ease,
    mtld,
)


class DifficultyMetricsTests(unittest.TestCase):
    def test_compression_ratio_repetitive_vs_diverse(self) -> None:
        repetitive = ("the cat sat on the mat. " * 200).strip()
        diverse = (
            "Quantum entanglement bewilders philosophers contemplating nonlocal correlations "
            "amidst cryptographic protocols and topological invariants unexpectedly."
        )
        r_rep, _, _ = compression_ratio(repetitive)
        r_div, _, _ = compression_ratio(diverse)
        self.assertGreater(r_rep, r_div)
        self.assertGreater(r_rep, 1.0)

    def test_flesch_simple_text_is_easier(self) -> None:
        easy = "I like cats. Cats are nice. The cat sat on the mat."
        hard = (
            "Notwithstanding the aforementioned methodological considerations, "
            "the interdisciplinary synthesis remains epistemologically fraught."
        )
        easy_score, _, _, _ = flesch_reading_ease(easy)
        hard_score, _, _, _ = flesch_reading_ease(hard)
        self.assertGreater(easy_score, hard_score)

    def test_mtld_diverse_higher(self) -> None:
        repetitive = " ".join(["alpha beta"] * 80)
        diverse = " ".join(f"token{i}" for i in range(160))
        self.assertGreater(mtld(diverse), mtld(repetitive))

    def test_compute_bundle(self) -> None:
        metrics = compute_difficulty_metrics(
            "Organic chemistry covers bonds, reactions, and mechanisms. "
            "Students learn carefully with examples and practice problems."
        )
        self.assertFalse(math.isnan(metrics.compression_ratio))
        self.assertFalse(math.isnan(metrics.flesch_reading_ease))
        self.assertFalse(math.isnan(metrics.mtld))
        self.assertGreater(metrics.n_words, 0)


if __name__ == "__main__":
    unittest.main()
