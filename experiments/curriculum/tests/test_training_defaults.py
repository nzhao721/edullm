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
