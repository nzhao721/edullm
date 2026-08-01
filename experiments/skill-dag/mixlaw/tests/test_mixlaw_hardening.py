"""Focused local tests for MixLaw recovery and production contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_MIXLAW = Path(__file__).resolve().parents[1]
_TS_ROOT = _MIXLAW.parents[1] / "token-selection"
for _path in (_MIXLAW, _TS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import mixlaw_runtime as runtime  # noqa: E402
import mixlaw_wandb as mix_wandb  # noqa: E402
from token_selection.olmo_ext.wandb_logging import TASK_LOSS_RAW_LABELS  # noqa: E402


def _durable_metadata(path: Path, step: int = 125, *, local: bool = True) -> Path:
    uri = (
        f"/scratch/mixlaw/save/checkpoints/step{step}"
        if local
        else (
            "s3://edullm-checkpoints/mixlaw/370m-validation/"
            f"mix01/checkpoints/step{step}"
        )
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "last_durable_step": step,
                "checkpoint_uri": uri,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_mixlaw_direct_default_is_10b_and_2384_steps() -> None:
    assert runtime.MIXLAW_DEFAULT_LENGTH_TOKENS == 10_000_000_000
    assert runtime.MIXLAW_DEFAULT_STEPS == 2384


def test_recovery_modes_are_explicit(tmp_path: Path) -> None:
    assert runtime.resolve_recovery_args(
        "fresh", ["--no-auto-stage"], mix_name="mix01"
    ) == ["--no-auto-stage", "--fresh"]
    assert runtime.resolve_recovery_args(
        "fail", ["--no-auto-stage"], mix_name="mix01"
    ) == ["--no-auto-stage"]
    marker = _durable_metadata(tmp_path / "last_durable_step.json")
    resolved = runtime.resolve_recovery_args(
        "resume",
        ["--no-auto-stage"],
        mix_name="mix01",
        durable_metadata_path=marker,
    )
    assert resolved[-2:] == [
        "--load-path",
        "/scratch/mixlaw/save/checkpoints/step125",
    ]


@pytest.mark.parametrize(
    ("mode", "extra"),
    [
        ("fresh", ["--load-path", "s3://bucket/step1"]),
        ("resume", ["--fresh"]),
        ("fail", ["--fresh"]),
        ("fail", ["--load-path=s3://bucket/step1"]),
    ],
)
def test_recovery_rejects_conflicting_intent(mode: str, extra: list[str]) -> None:
    with pytest.raises(runtime.MixLawContractError):
        runtime.resolve_recovery_args(mode, extra, mix_name="mix01")


def test_resume_marker_must_match_mix_and_step(tmp_path: Path) -> None:
    marker = _durable_metadata(tmp_path / "last_durable_step.json")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["last_durable_step"] = 250
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runtime.MixLawContractError, match="does not match"):
        runtime.checkpoint_uri_from_durable_metadata(marker, mix_name="mix01")


def test_production_requires_strict_complete_eval() -> None:
    assert runtime.production_contract_errors(
        durable_export=True,
        task_loss_on_save=False,
        task_loss_strict=False,
    ) == [
        "production durability requires --task-loss-on-save",
        "production durability requires --task-loss-strict",
    ]
    assert runtime.production_contract_errors(
        durable_export=False,
        task_loss_on_save=False,
        task_loss_strict=False,
    ) == []


def test_dependency_preflight_requires_all_bpb_labels_and_current_edullm(
    tmp_path: Path,
) -> None:
    versions = {name: "1.0.0" for name in runtime.DEPENDENCY_DISTRIBUTIONS}
    versions["edullm-data"] = "0.2.9"
    ladder = tmp_path / "config.yaml"
    evaluator = tmp_path / "eval.py"
    ladder.write_text("model: {}\n", encoding="utf-8")
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    errors = runtime.dependency_contract_errors(
        versions,
        runtime.OLMES_BPB_LABELS[:-1],
        ladder_base_config=ladder,
        eval_script=evaluator,
    )
    assert any("obsolete" in error for error in errors)
    assert any("missing 1/20" in error for error in errors)

    versions["edullm-data"] = "0.6.3"
    assert runtime.dependency_contract_errors(
        versions,
        runtime.OLMES_BPB_LABELS,
        ladder_base_config=ladder,
        eval_script=evaluator,
    ) == []


def test_mixlaw_wandb_macro_uses_only_complete_raw_suite(monkeypatch) -> None:
    class Run:
        def __init__(self) -> None:
            self.logged: list[tuple[int, dict[str, float]]] = []

        def log(self, metrics, step=None):
            self.logged.append((step, dict(metrics)))

    run = Run()
    partial = {"macro_mean": 0.1, "labels": {TASK_LOSS_RAW_LABELS[0]: 2.0}}
    mix_wandb.wandb_log_eval(run, partial, step=1)
    assert "eval/macro_bpb" not in run.logged[-1][1]

    labels = {label: float(index + 1) for index, label in enumerate(TASK_LOSS_RAW_LABELS)}
    mix_wandb.wandb_log_eval(
        run,
        {"macro_mean": -999.0, "labels": labels},
        step=2,
    )
    assert run.logged[-1][1]["eval/macro_bpb"] == sum(labels.values()) / 20


def test_trainer_records_local_durable_marker_after_wandb_upload() -> None:
    source = (_MIXLAW / "train_mixlaw_validation_370m.py").read_text(encoding="utf-8")
    function = source[source.index("def _commit_durable_checkpoint") : source.index("def resolve_load_path")]
    assert "write_last_durable_step(" in function
    assert "export_mixlaw_checkpoint(" not in function
    assert "publish_last_durable_step(" not in function
    assert "wandb_log_checkpoint(" in source
