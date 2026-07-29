"""Curriculum pacing schedules for RegMix-10B OLMo2-370M experiments.

All curriculum modes operate over pre-sorted ``difficulty_rank`` (easy = low
rank / low index). Training runs ``TOTAL_STEPS=2384`` with a **250-step**
segment grid so every bucket/segment ends on a checkpoint (250 = 2× the 125
save interval). The final segment is **134** steps (2250→2384).

Pacing names
------------
``control``
    Uniform over the full corpus (caller uses flat memmap shuffle; this module
    still exposes a full-pool helper for tests).
``linear_n10``
    Sequential difficulty buckets 1→10 on the 250-step table.
``expanding_25_1000``
    Eligible pool grows from easiest 25% at step 0 to 100% at step 1000.
``warmup_1000``
    Strict easy→hard order for steps 0–999; uniform shuffle thereafter.
``interleave_i10_linear``
    Ten segments; within each, replay a full linear N=10 mini-curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

TOTAL_STEPS = 2384
N_BUCKETS = 10
SEGMENT_BOUNDARIES: Tuple[int, ...] = (
    0,
    250,
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    2000,
    2250,
    2384,
)
EXPANDING_T_C = 1000
EXPANDING_LAMBDA0 = 0.25
WARMUP_SWITCH = 1000

PACING_NAMES = (
    "control",
    "linear_n10",
    "expanding_25_1000",
    "warmup_1000",
    "interleave_i10_linear",
)

DIFFICULTY_METRICS = (
    "compression_ratio",
    "flesch",
    "mtld",
    "learnability",
)

# Easy → hard sort: (column_name, reverse)
METRIC_SORT: dict[str, tuple[str, bool]] = {
    "compression_ratio": ("compression_ratio", False),  # asc
    "flesch": ("flesch_reading_ease", True),  # desc (higher = easier)
    "mtld": ("mtld", False),  # asc
    "learnability": ("learnability_late_minus_early_avg_nll", False),  # asc
}


def segment_index(step: int, boundaries: Sequence[int] = SEGMENT_BOUNDARIES) -> int:
    """0-based segment index for ``step`` in ``[start, end)`` ranges."""
    s = int(step)
    if s < 0:
        raise ValueError(f"step must be >= 0, got {s}")
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= s < boundaries[i + 1]:
            return i
    if s == boundaries[-1]:
        # Final step is past the last exclusive end for sampling; clamp to last.
        return len(boundaries) - 2
    raise ValueError(f"step {s} outside [0, {boundaries[-1]}]")


def segment_range(seg: int, boundaries: Sequence[int] = SEGMENT_BOUNDARIES) -> Tuple[int, int]:
    """Inclusive start, exclusive end for segment ``seg``."""
    if not 0 <= seg < len(boundaries) - 1:
        raise ValueError(f"segment {seg} out of range")
    return int(boundaries[seg]), int(boundaries[seg + 1])


def split_equal_mass(n: int, n_buckets: int = N_BUCKETS) -> List[Tuple[int, int]]:
    """Split ``[0, n)`` into ``n_buckets`` nearly equal contiguous ranges.

    Remainder tokens go to the **last** buckets (standard ceil/floor split so
    the first ``n % n_buckets`` buckets get one extra when using floor+spread
    on the front). We put remainder on the **first** buckets so early easy
    buckets never shrink relative to hard ones.
    """
    if n_buckets <= 0:
        raise ValueError("n_buckets must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    base, rem = divmod(int(n), int(n_buckets))
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_buckets):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def expanding_eligible_fraction(
    step: int,
    *,
    t_c: int = EXPANDING_T_C,
    lambda0: float = EXPANDING_LAMBDA0,
) -> float:
    """Fraction of easiest ranks eligible at ``step`` for expanding pacing."""
    s = max(0, int(step))
    if t_c <= 0:
        return 1.0
    if s >= t_c:
        return 1.0
    return float(lambda0 + (1.0 - lambda0) * (s / float(t_c)))


def interleave_subbucket_durations(segment_steps: int, n_buckets: int = N_BUCKETS) -> List[int]:
    """Steps per sub-bucket inside one interleave segment.

    For 250-step segments: all 25. For the final 134-step segment: base 13 with
    remainder on the last bucket (9×13 + 17 = 134).
    """
    if segment_steps <= 0:
        raise ValueError("segment_steps must be > 0")
    base, rem = divmod(int(segment_steps), int(n_buckets))
    # Remainder on the last bucket (plan convention for segment 10).
    durs = [base] * n_buckets
    durs[-1] += rem
    return durs


def interleave_subbucket_index(
    step: int,
    boundaries: Sequence[int] = SEGMENT_BOUNDARIES,
    n_buckets: int = N_BUCKETS,
) -> int:
    """Which difficulty sub-bucket (0..n_buckets-1) is active at ``step``."""
    seg = segment_index(step, boundaries)
    start, end = segment_range(seg, boundaries)
    local = int(step) - start
    durs = interleave_subbucket_durations(end - start, n_buckets)
    cum = 0
    for i, d in enumerate(durs):
        cum += d
        if local < cum:
            return i
    return n_buckets - 1


@dataclass(frozen=True)
class PoolSpec:
    """Active sampling pool for a given step."""

    mode: str
    # Inclusive start / exclusive end into the easy→hard ranked chunk array.
    # For ``warmup`` ordered mode, ``ordered=True`` and indices are consumed
    # sequentially rather than sampled uniformly.
    start: int
    end: int
    ordered: bool = False
    # Absolute position in the ordered stream (warmup only).
    ordered_offset: Optional[int] = None


def pool_for_step(
    step: int,
    n_chunks: int,
    pacing: str,
    *,
    total_steps: int = TOTAL_STEPS,
    n_buckets: int = N_BUCKETS,
    boundaries: Sequence[int] = SEGMENT_BOUNDARIES,
) -> PoolSpec:
    """Return the active index range into a length-``n_chunks`` easy→hard array."""
    name = str(pacing).strip().lower()
    n = int(n_chunks)
    if n <= 0:
        raise ValueError("n_chunks must be > 0")
    if name == "control":
        return PoolSpec(mode=name, start=0, end=n, ordered=False)

    buckets = split_equal_mass(n, n_buckets)

    if name == "linear_n10":
        seg = segment_index(step, boundaries)
        lo, hi = buckets[seg]
        return PoolSpec(mode=name, start=lo, end=max(lo + 1, hi) if hi == lo else hi, ordered=False)

    if name == "expanding_25_1000":
        frac = expanding_eligible_fraction(step)
        end = max(1, int(round(frac * n)))
        end = min(n, end)
        return PoolSpec(mode=name, start=0, end=end, ordered=False)

    if name == "warmup_1000":
        if int(step) < WARMUP_SWITCH:
            # Strict ascending: each global step advances through the ranked list
            # wrapping if needed. Batching is handled by the stream.
            return PoolSpec(
                mode=name,
                start=0,
                end=n,
                ordered=True,
                ordered_offset=int(step) % n,
            )
        return PoolSpec(mode=name, start=0, end=n, ordered=False)

    if name == "interleave_i10_linear":
        sub = interleave_subbucket_index(step, boundaries, n_buckets)
        lo, hi = buckets[sub]
        return PoolSpec(mode=name, start=lo, end=max(lo + 1, hi) if hi == lo else hi, ordered=False)

    raise ValueError(f"Unknown pacing {pacing!r}; expected one of {PACING_NAMES}")


class CurriculumChunkStream:
    """Step-conditioned sampler over easy→hard ranked global chunk indices.

    ``ranked_chunk_indices[i]`` is the ``global_chunk_idx`` of the i-th easiest
    chunk (by the chosen difficulty metric). Distributed ranks share the same
    ``(seed, step)`` RNG stream and take non-overlapping slices of each draw.
    """

    def __init__(
        self,
        ranked_chunk_indices: Sequence[int],
        *,
        pacing: str,
        difficulty_metric: str = "compression_ratio",
        total_steps: int = TOTAL_STEPS,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if pacing not in PACING_NAMES:
            raise ValueError(f"Unknown pacing {pacing!r}")
        if pacing != "control" and difficulty_metric not in DIFFICULTY_METRICS:
            raise ValueError(f"Unknown difficulty_metric {difficulty_metric!r}")
        self.ranked = np.asarray(ranked_chunk_indices, dtype=np.int64)
        if self.ranked.ndim != 1 or len(self.ranked) == 0:
            raise ValueError("ranked_chunk_indices must be a non-empty 1-D sequence")
        self.pacing = pacing
        self.difficulty_metric = difficulty_metric
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self._warmup_cursor = 0

    def pool(self, step: int) -> PoolSpec:
        return pool_for_step(
            step,
            len(self.ranked),
            self.pacing,
            total_steps=self.total_steps,
        )

    def next_indices(self, step: int, batch_size: int) -> List[int]:
        """Return ``batch_size`` global chunk indices for this rank at ``step``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        pool = self.pool(step)
        n_take = int(batch_size) * self.world_size

        if pool.ordered:
            # Deterministic ascending walk shared across ranks, then slice.
            # Prefer step-derived ordered_offset (resume / step-addressed access);
            # fall back to the stream cursor for uninterrupted sequential draws.
            if pool.ordered_offset is not None:
                start = int(pool.ordered_offset) % len(self.ranked)
            else:
                start = self._warmup_cursor % len(self.ranked)
            pos = np.arange(start, start + n_take, dtype=np.int64) % len(self.ranked)
            self._warmup_cursor = int(start + n_take) % len(self.ranked)
            chosen = self.ranked[pos]
        else:
            lo, hi = pool.start, pool.end
            if hi <= lo:
                raise RuntimeError(f"empty pool at step={step}: [{lo}, {hi})")
            rng = np.random.default_rng(self._step_seed(step))
            # Sample with replacement if the pool is smaller than the draw.
            replace = (hi - lo) < n_take
            local = rng.choice(hi - lo, size=n_take, replace=replace)
            chosen = self.ranked[lo + local]

        offset = self.rank * int(batch_size)
        return [int(x) for x in chosen[offset : offset + int(batch_size)]]

    def _step_seed(self, step: int) -> int:
        # Stable across ranks/processes (avoid PYTHONHASHSEED-dependent hash()).
        pacing_tag = sum(ord(c) for c in self.pacing) % 10_007
        return (self.seed * 1_000_003 + int(step) * 97_651 + pacing_tag) & 0x7FFFFFFF
