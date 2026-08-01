from __future__ import annotations

import pytest

from token_selection.olmo_ext import durability


def test_local_durable_step_is_atomic_and_monotonic(tmp_path):
    path = durability.write_last_durable_step(
        tmp_path,
        125,
        checkpoint_artifact="team/token-selection/run-checkpoint:step-0000125",
        extra={"run_id": "unit"},
    )
    assert path.name == durability.LAST_DURABLE_STEP_FILENAME
    assert durability.read_last_durable_step(tmp_path) == {
        "schema_version": 2,
        "last_durable_step": 125,
        "durability": "local_scratch+wandb",
        "checkpoint_artifact": "team/token-selection/run-checkpoint:step-0000125",
        "run_id": "unit",
    }
    assert list(tmp_path.glob("*.tmp")) == []
    with pytest.raises(durability.DurabilityError, match="backward"):
        durability.write_last_durable_step(tmp_path, 124)
