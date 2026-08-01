from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_standalone_trainers_share_permanent_checkpoint_contract():
    trainers = [
        "control/train_ce_regmix_olmo_370m.py",
        "blade/train_blade_olmo_370m.py",
        "middle-ppl-doc/train_ce_middle_ppl_doc.py",
        "learnability-doc/train_ce_learnability_doc_olmo_370m.py",
    ]
    for trainer in trainers:
        text = _text(trainer)
        assert (
            "DEFAULT_LENGTH_TOKENS = 9_900_000_000" in text
            or "9900000000 -> 2360 steps" in text
        ), trainer
        assert "finalize_permanent_checkpoint" in text, trainer
        assert "write_run_fingerprint" in text, trainer
        assert "assert_resume_fingerprint" in text, trainer
        assert "fused = False" in text, trainer


def test_launchers_declare_strict_synchronous_eval_resources():
    launchers = [
        "control/launch_control.sh",
        "blade/launch_train.sh",
        "middle-ppl-doc/launch_train.sh",
        "learnability-doc/launch_train.sh",
        "attention/launch.sh",
        "learnability-token/launch.sh",
        "middle-ppl-token/launch_train.sh",
        "rel-ema-exp/launch_train.sh",
        "rel-ema-refhq/launch_train.sh",
        "rho-1/launch.sh",
    ]
    for launcher in launchers:
        text = _text(launcher)
        assert "TASK_LOSS_STRICT=1" in text, launcher
        assert "TASK_LOSS_NPROC" in text, launcher


def test_template_launchers_forward_wandb_resume_artifacts():
    assert "--wandb-resume-artifact" in _text(
        "token_selection/scripts/train_olmo_template.py"
    )
    launchers = [
        "attention/launch.sh",
        "learnability-token/launch.sh",
        "middle-ppl-token/launch_train.sh",
        "rel-ema-exp/launch_train.sh",
        "rel-ema-refhq/launch_train.sh",
        "rho-1/launch.sh",
    ]
    for launcher in launchers:
        text = _text(launcher)
        assert "WANDB_RESUME_ARTIFACT" in text, launcher
        assert "--wandb-resume-artifact" in text, launcher


def test_blade_schedule_and_seed_offsets_remain_locked():
    text = _text("blade/train_blade_olmo_370m.py")
    assert "BLADE_START = 500" in text
    assert "BLADE_SYNC_STEPS: Tuple[int, ...] = (500, 875, 1250, 1625, 2000)" in text
    assert '"reference_stream_seed_offset": 101' in text
    assert '"reference_train_stream_seed_offset": 17' in text
