"""Focused source-contract tests for curriculum execution hardening."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train_curriculum_regmix_370m.py"
LAUNCHER = ROOT / "launch" / "launch_arm.sh"


def _function_source(name: str) -> str:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    return ast.unparse(node)


def test_checkpoint_eval_wandb_publish_order_is_fail_closed() -> None:
    run = _function_source("_run")
    first_save = run.index("save_checkpoint(ckpt0")
    first_eval = run.index("pause_eval_reload_curriculum(", first_save)
    first_publish = run.index("publish_completed_checkpoint(args, wb_run, ckpt0", first_eval)
    assert first_save < first_eval < first_publish

    publish = _function_source("publish_completed_checkpoint")
    assert publish.index("wandb_log_eval(") < publish.index(
        "wandb_log_checkpoint("
    ) < publish.index("wandb_log_runtime_artifacts(")
    assert "last_wandb_step.json" in publish
    assert "_abort_all_ranks" in publish

    upload = _function_source("_wandb_log_artifact_and_wait")
    assert "wait()" in upload
    assert "required" in upload


def test_trainer_consumes_shared_pause_eval_reload_helper() -> None:
    pause = _function_source("pause_eval_reload_curriculum")
    assert "pause_eval_reload_distributed" in pause
    assert "strict=True" in pause
    assert "suite_complete" in pause
    assert "task_loss.jsonl" in pause


def test_trainer_rejects_legacy_coordinates_and_implicit_recovery() -> None:
    parse = _function_source("parse_args")
    assert "legacy document-local coordinate path" in parse
    assert "choose recovery mode explicitly" in parse
    assert "--fresh and --load-path are mutually exclusive" in parse
    assert "--ladder-base-config" in parse

    stage = _function_source("stage_load_path")
    assert "S3 checkpoint resume is prohibited" in stage
    assert "wandb-artifact://" in stage
    assert "use_artifact" in stage


def test_launcher_preflights_and_explicit_recovery() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "LADDER_BASE_CONFIG is required" in text
    assert "HF token present" in text
    assert "FRESH=1 and LOAD_PATH are mutually exclusive" in text
    assert "choose recovery mode explicitly" in text
    assert "CURRICULUM_INDEX is rejected" in text
    assert "--no-task-loss-on-save" in text
    assert "S3_EXPORT" not in text
    assert "--no-s3-export" not in text
    assert "WANDB_MODE=online is required" in text
