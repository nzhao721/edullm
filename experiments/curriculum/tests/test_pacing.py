"""Unit tests for curriculum pacing schedules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from curriculum_pacing import (
    N_BUCKETS,
    SEGMENT_BOUNDARIES,
    TOTAL_STEPS,
    CurriculumChunkStream,
    expanding_eligible_fraction,
    interleave_subbucket_durations,
    interleave_subbucket_index,
    pool_for_step,
    segment_index,
    segment_range,
    split_equal_mass,
)


def test_segment_boundaries_align_to_checkpoints():
    assert SEGMENT_BOUNDARIES == (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2384)
    assert SEGMENT_BOUNDARIES[-1] == TOTAL_STEPS
    # Every interior boundary is on the 125-step ladder grid.
    for b in SEGMENT_BOUNDARIES[1:-1]:
        assert b % 125 == 0
    # Last segment is 134 steps.
    assert SEGMENT_BOUNDARIES[-1] - SEGMENT_BOUNDARIES[-2] == 134


@pytest.mark.parametrize(
    "step,expected",
    [
        (0, 0),
        (249, 0),
        (250, 1),
        (500, 2),
        (999, 3),
        (1000, 4),
        (2000, 8),
        (2249, 8),
        (2250, 9),
        (2383, 9),
    ],
)
def test_segment_index_at_boundaries(step: int, expected: int):
    assert segment_index(step) == expected


def test_linear_n10_uses_sequential_buckets():
    n = 1000
    buckets = split_equal_mass(n, N_BUCKETS)
    for step, seg in [(0, 0), (250, 1), (1000, 4), (2250, 9)]:
        pool = pool_for_step(step, n, "linear_n10")
        assert (pool.start, pool.end) == buckets[seg]
        assert not pool.ordered


def test_expanding_fraction_at_probes():
    assert expanding_eligible_fraction(0) == pytest.approx(0.25)
    assert expanding_eligible_fraction(500) == pytest.approx(0.25 + 0.75 * 0.5)
    assert expanding_eligible_fraction(1000) == pytest.approx(1.0)
    assert expanding_eligible_fraction(2000) == pytest.approx(1.0)
    n = 1000
    p0 = pool_for_step(0, n, "expanding_25_1000")
    assert p0.start == 0
    assert p0.end == 250  # 25% of 1000
    p500 = pool_for_step(500, n, "expanding_25_1000")
    assert p500.end == 625
    p1000 = pool_for_step(1000, n, "expanding_25_1000")
    assert p1000.end == n


def test_warmup_switches_at_1000():
    n = 500
    early = pool_for_step(0, n, "warmup_1000")
    assert early.ordered is True
    late = pool_for_step(1000, n, "warmup_1000")
    assert late.ordered is False
    assert late.start == 0 and late.end == n


def test_interleave_subbuckets_250_and_134():
    assert interleave_subbucket_durations(250) == [25] * 10
    d134 = interleave_subbucket_durations(134)
    assert sum(d134) == 134
    assert d134[:9] == [13] * 9
    assert d134[-1] == 17  # remainder on last


def test_interleave_subbucket_index_within_segment():
    # First segment: steps 0-24 → sub 0, 25-49 → sub 1, ...
    assert interleave_subbucket_index(0) == 0
    assert interleave_subbucket_index(24) == 0
    assert interleave_subbucket_index(25) == 1
    assert interleave_subbucket_index(249) == 9
    # Segment 2 restarts easy→hard.
    assert interleave_subbucket_index(250) == 0
    assert interleave_subbucket_index(275) == 1
    # Final segment uses 13-step base.
    assert interleave_subbucket_index(2250) == 0
    assert interleave_subbucket_index(2262) == 0  # 13 steps: 2250-2262
    assert interleave_subbucket_index(2263) == 1


def test_stream_next_indices_in_pool_and_rank_split():
    ranked = np.arange(100, dtype=np.int64)
    s0 = CurriculumChunkStream(ranked, pacing="linear_n10", seed=7, rank=0, world_size=2)
    s1 = CurriculumChunkStream(ranked, pacing="linear_n10", seed=7, rank=1, world_size=2)
    a = s0.next_indices(0, batch_size=4)
    b = s1.next_indices(0, batch_size=4)
    assert len(a) == 4 and len(b) == 4
    assert set(a).isdisjoint(set(b))
    buckets = split_equal_mass(100, 10)
    lo, hi = buckets[0]
    for idx in a + b:
        # ranked[i] == i here; pool is easy bucket 0.
        assert lo <= idx < hi


def test_segment_range_last():
    start, end = segment_range(9)
    assert start == 2250 and end == 2384


def test_warmup_fresh_stream_honors_ordered_offset():
    """Fresh CurriculumChunkStream at step N must start at ordered_offset, not 0."""
    n = 20
    ranked = list(range(n))
    step = 999
    pool = pool_for_step(step, n, "warmup_1000")
    assert pool.ordered is True
    assert pool.ordered_offset == step % n  # 19

    stream = CurriculumChunkStream(ranked, pacing="warmup_1000", difficulty_metric="compression_ratio")
    idxs = stream.next_indices(step, batch_size=4)
    # Must not restart from the beginning of the ranked list.
    assert idxs != [0, 1, 2, 3]
    expected_start = int(pool.ordered_offset)
    assert idxs[0] == ranked[expected_start]
    expected = [ranked[(expected_start + i) % n] for i in range(4)]
    assert idxs == expected


def test_warmup_sequential_steps_follow_ordered_offset():
    """Sequential step-addressed draws advance with step % n, not a zeroed cursor."""
    n = 20
    ranked = list(range(n))
    stream = CurriculumChunkStream(ranked, pacing="warmup_1000", difficulty_metric="compression_ratio")
    for step in (0, 1, 5, 19, 20):
        pool = pool_for_step(step, n, "warmup_1000")
        idxs = stream.next_indices(step, batch_size=1)
        assert idxs == [ranked[int(pool.ordered_offset) % n]]
