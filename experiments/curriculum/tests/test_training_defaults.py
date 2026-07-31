"""Training defaults for curriculum RegMix-370M (ladder, GBS, constant LR)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_TS_ROOT = Path(__file__).resolve().parents[2] / "token-selection"
if str(_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TS_ROOT))

from token_selection.olmo_ext.checkpoint_ladder import (  # noqa: E402
    assert_ladder_example_2384,
    permanent_checkpoint_steps,
)

_TRAIN = Path(__file__).resolve().parents[1] / "train_curriculum_regmix_370m.py"


def _module_constants() -> dict:
    """Parse trainer module-level numeric/string constants without importing torch/olmo."""
    tree = ast.parse(_TRAIN.read_text(encoding="utf-8"))
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and isinstance(node.value, (ast.Constant,)):
                out[tgt.id] = node.value.value
    return out


def test_ladder_2384_omits_2375():
    steps = assert_ladder_example_2384()
    assert 2375 not in steps
    assert steps == permanent_checkpoint_steps(2384, 125)
    assert 0 in steps and 2250 in steps and 2384 in steps


def test_ema_steps_subset_of_ladder():
    ladder = set(permanent_checkpoint_steps(2384, 125))
    for s in (2000, 2125, 2250, 2384):
        assert s in ladder


def test_trainer_defaults():
    c = _module_constants()
    assert c["SEQ_LEN"] == 2048
    assert c["EMBEDDING_SIZE"] == 100_352
    assert c["GLOBAL_BATCH_TOKENS"] == 4_194_304
    assert c["MICROBATCH_TOKENS"] == 65_536
    assert c["PEAK_LR"] == pytest.approx(4.0e-4)
    assert c["DEFAULT_SEED"] == 42
    assert c["DEFAULT_LENGTH_TOKENS"] == 10_000_058_051
    # 10_000_058_051 // 4_194_304 == 2384
    assert c["DEFAULT_LENGTH_TOKENS"] // c["GLOBAL_BATCH_TOKENS"] == 2384


def test_default_lr_alpha_f_is_constant():
    tree = ast.parse(_TRAIN.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            args = [a for a in node.args if isinstance(a, ast.Constant)]
            if args and args[0].value == "--lr-alpha-f":
                for kw in node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        assert kw.value.value == 1.0
                        found = True
    assert found, "--lr-alpha-f default=1.0 not found"


def test_arm_id_helper():
    # Import only the pure helper by exec'ing a slice is heavy; replicate mapping
    # contract from source text.
    text = _TRAIN.read_text(encoding="utf-8")
    assert 'return "control"' in text
    assert "linear10" in text and "expand" in text and "warmup" in text and "interleave" in text
    assert '"cr"' in text and '"learn"' in text


def test_trainer_binds_edullm_data_not_legacy_bucket():
    text = _TRAIN.read_text(encoding="utf-8")
    assert 'DEFAULT_TRAIN_DATASET_ID = "pretrain/regmix-10b"' in text
    assert "dataset_paths" in text and "resolve_latest" in text
    assert "s3://edullm-datasets/regmix/regmix-10b/" not in text
    assert "curriculum/regmix-compression-370m" in text
    assert "curriculum/regmix-flesch-370m" in text
    assert "curriculum/regmix-mtld-370m" in text
    assert "curriculum/regmix-learnability-370m" in text


def test_trainer_ephemeral_durable_s3_contract():
    text = _TRAIN.read_text(encoding="utf-8")
    assert "export_curriculum_checkpoint" in text
    assert "export_curriculum_artifacts" in text
    assert "curriculum_s3_uri" in text
    assert "_require_sync_to_s3" in text
    assert "_abort_all_ranks" in text
    assert "stage_load_path" in text
    assert "sync_from_s3" in text
    assert 'CHECKPOINT_BUCKET = "edullm-checkpoints"' in text
    assert 'CURRICULUM_S3_ROOT = "curriculum"' in text
    assert "ephemeral_runtime" in text
    # Fail-closed durable export (not warn-only).
    assert "durable S3 export failed" in text
    assert "fail closed via broadcast" in text or "_abort_all_ranks" in text
    # No auto-resume from leftover job scratch.
    assert "do not auto-resume from scratch leftovers" in text
    assert 'Path(args.progress_dir) / "task_loss_results"' in text
    # Plan S3 layout includes metrics/.
    assert 'curriculum_s3_uri(arm_id, "metrics")' in text
    assert 'parent / "metrics"' in text
    # Fail-closed on unpublished curricula.
    assert "no published version of" in text
    assert "Publish a token-order/v1 curriculum under" in text


def test_trainer_wandb_smollm_protocol():
    """W&B mirrors SmolLM: project curriculum, durable sink, train/eval/ckpt logging."""
    text = _TRAIN.read_text(encoding="utf-8")
    assert 'DEFAULT_WANDB_PROJECT = "curriculum"' in text
    assert "WANDB_DISABLED" not in text  # must not force-disable like older CE forks
    assert "def durable_backend_ok" in text
    assert "def init_wandb" in text
    assert "def wandb_log_eval" in text
    assert "def wandb_log_checkpoint" in text
    assert "def wandb_drain_task_loss_evals" in text
    assert '"train/loss"' in text
    assert '"train/lr"' in text
    assert "--wandb-project" in text
    assert "--wandb-mode" in text
    assert "--allow-local-only" in text
    assert "allow_local_only" in text


def test_readme_and_launch_scrub_legacy_bucket():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    launch = (root / "launch" / "launch_arm.sh").read_text(encoding="utf-8")
    matrix = (root / "launch" / "submit_matrix.sh").read_text(encoding="utf-8")
    for text in (readme, launch, matrix):
        assert "edullm-datasets" not in text
        assert "edullm-data" in text
    assert "Ephemeral" in readme
    assert "edullm-checkpoints" in readme
    assert "job-scoped" in launch
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-curriculum}"' in launch
    assert "wandb-session.env" in launch
    assert "push_wandb_session_to_farmshare.sh" in launch
    assert "curriculum" in readme
    assert "W&B" in readme or "wandb" in readme.lower() or "Weights & Biases" in readme
