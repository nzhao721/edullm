from __future__ import annotations

import json

import pytest

from token_selection.olmo_ext import permanent_checkpoint as contract


def _identity(**updates):
    identity = {
        "arm": "control",
        "run_id": "control-regmix10b-v2",
        "method": "random",
        "seed": 6198,
        "max_tokens": 9_900_000_000,
        "total_steps": 2360,
        "keep_fraction": 0.6,
        "task_loss_definition": "olmo-ladder-20-label-macro-bpb",
        "fused_ce": False,
    }
    identity.update(updates)
    return identity


def test_fingerprint_round_trip_and_resume_guard(tmp_path):
    root = tmp_path / "checkpoints"
    step = root / "step125"
    step.mkdir(parents=True)
    fp = contract.write_run_fingerprint(root, _identity())
    contract.copy_fingerprint_into_checkpoint(fp, step)

    contract.assert_resume_fingerprint(step, _identity())
    with pytest.raises(contract.CheckpointContractError, match="keep_fraction"):
        contract.assert_resume_fingerprint(step, _identity(keep_fraction=0.5))


def test_legacy_checkpoint_is_explicitly_out_of_contract(tmp_path):
    step = tmp_path / "step2384"
    step.mkdir()
    (step / "state.pt").write_bytes(b"legacy")
    with pytest.raises(contract.CheckpointContractError, match="out-of-contract legacy"):
        contract.assert_resume_fingerprint(step, _identity())


def test_partial_legacy_task_loss_is_out_of_contract(tmp_path):
    result = tmp_path / "step125_task_loss.json"
    result.write_text(
        json.dumps(
            {
                "task_loss_bpb": {"arc_challenge_bpb": 1.0},
                "macro_mean_task_loss_bpb": 1.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(contract.CheckpointContractError, match="partial/legacy"):
        contract.validate_task_loss_result(result)


def test_production_export_cannot_disable_task_loss(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step125"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"state")
    with pytest.raises(contract.CheckpointContractError, match="without the complete"):
        contract.finalize_permanent_checkpoint(
            arm="control",
            checkpoint_dir=checkpoint,
            step=125,
            run_name="control",
            task_loss_dir=tmp_path / "eval",
            task_loss_enabled=False,
            wandb_run=object(),
            wandb_mode="online",
            production=True,
        )


def test_finalize_orders_eval_exports_then_marker(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints" / "step125"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"state")
    task_loss_dir = tmp_path / "task_loss"
    progress = tmp_path / "progress"
    progress.mkdir()
    fp = contract.write_run_fingerprint(checkpoint.parent, _identity())
    events: list[str] = []

    labels = {
        f"label_{i:02d}_bpb": float(i) for i in range(20)
    }

    def fake_eval(_checkpoint, *, out_path, **_kwargs):
        events.append("eval")
        out = type(checkpoint)(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "labels": labels,
                    "raw_label_count": 20,
                    "suite_complete": True,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "token_selection.olmo_ext.task_loss_hook.trigger_task_loss_eval", fake_eval
    )
    monkeypatch.setattr(
        "token_selection.olmo_ext.wandb_logging.task_loss_payload_complete",
        lambda payload: payload.get("suite_complete") is True
        and payload.get("raw_label_count") == 20,
    )
    monkeypatch.setattr(
        "token_selection.olmo_ext.wandb_logging.wandb_log_checkpoint",
        lambda *_a, **_k: events.append("checkpoint"),
    )
    monkeypatch.setattr(
        "token_selection.olmo_ext.wandb_logging.wandb_log_eval",
        lambda *_a, **_k: events.append("eval_artifact"),
    )
    monkeypatch.setattr(
        "token_selection.olmo_ext.wandb_logging.wandb_log_directory_artifact",
        lambda _run, _path, *, artifact_type, **_k: events.append(artifact_type),
    )
    monkeypatch.setattr(
        "token_selection.olmo_ext.durability.write_last_durable_step",
        lambda *_a, **_k: events.append("marker"),
    )

    contract.finalize_permanent_checkpoint(
        arm="control",
        checkpoint_dir=checkpoint,
        step=125,
        run_name="control-regmix10b-v2",
        task_loss_dir=task_loss_dir,
        task_loss_enabled=True,
        progress_dir=progress,
        fingerprint_path=fp,
        wandb_run=object(),
        wandb_mode="online",
        production=True,
    )

    assert events == [
        "eval",
        "checkpoint",
        "eval_artifact",
        "metrics",
        "eval",
        "marker",
    ]
