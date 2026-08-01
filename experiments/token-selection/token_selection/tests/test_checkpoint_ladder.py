"""Tests for the shared permanent checkpoint ladder."""

from __future__ import annotations

from token_selection.olmo_ext.checkpoint_ladder import (
    assert_ladder_example_2360,
    checkpointer_kwargs_for_ladder,
    is_permanent_checkpoint_step,
    permanent_checkpoint_steps,
)


def test_ladder_2360_omits_near_final_2250():
    steps = assert_ladder_example_2360()
    assert steps[0] == 0
    assert steps[-1] == 2360
    assert 125 in steps and 2125 in steps
    assert 2250 not in steps


def test_ladder_exact_multiple_keeps_final_grid():
    # final == last grid → keep it (it *is* the final).
    steps = permanent_checkpoint_steps(2500, 125)
    assert 2500 in steps
    assert 2375 in steps
    assert steps[-1] == 2500


def test_checkpointer_kwargs_no_ephemeral():
    kwargs = checkpointer_kwargs_for_ladder(2360, 125)
    assert kwargs["save_interval"] is None
    assert kwargs["ephemeral_save_interval"] is None
    assert kwargs["pre_train_checkpoint"] is True
    assert kwargs["max_checkpoints"] is None
    assert 0 not in kwargs["fixed_steps"]
    assert 2360 not in kwargs["fixed_steps"]
    assert 2250 not in kwargs["fixed_steps"]
    assert 2125 in kwargs["fixed_steps"]


def test_is_permanent_helper():
    assert is_permanent_checkpoint_step(0, 2360)
    assert is_permanent_checkpoint_step(2360, 2360)
    assert not is_permanent_checkpoint_step(2250, 2360)
    assert not is_permanent_checkpoint_step(100, 2360)
