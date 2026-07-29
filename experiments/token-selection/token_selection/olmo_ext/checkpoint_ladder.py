"""Permanent checkpoint ladder shared by all token-selection arms.

Contract (every arm):
  * step 0 (pre-train / init snapshot)
  * every ``interval`` steps on the grid
  * always the true final step
  * omit the last on-grid step when it falls within one interval of the final
    (avoids a near-duplicate snapshot). Example for 2384 steps / interval 125:
    ``{0, 125, …, 2250, 2384}`` — omit 2375.
  * ``max_checkpoints=None`` — keep every save permanently; no ephemeral prune.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


DEFAULT_CHECKPOINT_INTERVAL = 125


def permanent_checkpoint_steps(
    total_steps: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> List[int]:
    """Return sorted permanent save steps for a run of ``total_steps``.

    Raises ``ValueError`` if ``total_steps < 0`` or ``interval <= 0``.
    For ``total_steps == 0`` returns ``[0]``.
    """
    if total_steps < 0:
        raise ValueError(f"total_steps must be >= 0, got {total_steps}")
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got {interval}")
    if total_steps == 0:
        return [0]

    steps = {0, int(total_steps)}
    last_grid = (int(total_steps) // int(interval)) * int(interval)
    for s in range(int(interval), last_grid + 1, int(interval)):
        steps.add(s)
    # Skip last on-grid when it is a near-duplicate of the true final.
    if (
        last_grid > 0
        and last_grid != int(total_steps)
        and (int(total_steps) - last_grid) < int(interval)
    ):
        steps.discard(last_grid)
    return sorted(steps)


def is_permanent_checkpoint_step(
    step: int,
    total_steps: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> bool:
    """True iff ``step`` belongs on the permanent ladder for this run."""
    return int(step) in set(permanent_checkpoint_steps(total_steps, interval))


def checkpointer_kwargs_for_ladder(
    total_steps: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    *,
    save_async: bool = False,
) -> Dict[str, Any]:
    """OLMo-core ``CheckpointerCallback`` kwargs implementing the permanent ladder.

    Step 0 is ``pre_train_checkpoint``; the true final is the trainer post-train
    save; intermediate ladder steps go in ``fixed_steps``. Interval cadence is
    disabled so nothing outside the ladder is written. No ephemeral rotation.
    """
    ladder = permanent_checkpoint_steps(total_steps, interval)
    fixed = [s for s in ladder if s not in (0, int(total_steps))]
    return {
        "save_interval": None,
        "fixed_steps": fixed,
        "ephemeral_save_interval": None,
        "pre_train_checkpoint": True,
        "save_async": bool(save_async),
        "max_checkpoints": None,
    }


def assert_ladder_example_2384(interval: int = DEFAULT_CHECKPOINT_INTERVAL) -> Sequence[int]:
    """Sanity helper used by tests / docs: 2384-step ladder omits 2375."""
    steps = permanent_checkpoint_steps(2384, interval)
    assert 0 in steps and 2384 in steps
    assert 2375 not in steps
    assert 2250 in steps
    return steps
