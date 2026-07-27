"""compare_runs is deferred until later test-loss evaluation."""

from __future__ import annotations

import pytest

from token_selection.scripts import compare_runs


def test_compare_runs_fails_closed_as_deferred(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["compare_runs", "--config", "token_selection/configs/run_rho_10b.yaml"],
    )
    with pytest.raises(SystemExit, match="deferred"):
        compare_runs.main()
