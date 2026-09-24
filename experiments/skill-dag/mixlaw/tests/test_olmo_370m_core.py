"""Source-contract tests for the shared OLMo2-370M trainer core."""
from __future__ import annotations

import ast
from pathlib import Path

_MIXLAW = Path(__file__).resolve().parents[1]
_CORE = _MIXLAW / "olmo_370m_core.py"
_TRAINERS = (
    _MIXLAW / "train_mixlaw_validation_370m.py",
    _MIXLAW.parent / "skillit" / "train_skillit_370m.py",
)


def _tree() -> ast.Module:
    return ast.parse(_CORE.read_text(encoding="utf-8"))


def _function_source(name: str) -> str:
    node = next(
        item
        for item in _tree().body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return ast.unparse(node)


def test_core_defines_every_name_the_trainers_use() -> None:
    defined = set()
    for node in _tree().body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined |= {x.id for t in node.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    used = set()
    for trainer in _TRAINERS:
        tree = ast.parse(trainer.read_text(encoding="utf-8"))
        used |= {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "core"
        }
    assert used, "trainers no longer reference the core module"
    assert used <= defined, sorted(used - defined)


def test_core_has_no_curriculum_imports() -> None:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom):
            assert "curriculum" not in (node.module or "")
        elif isinstance(node, ast.Import):
            assert all("curriculum" not in alias.name for alias in node.names)


def test_resume_staging_refuses_s3_checkpoints() -> None:
    stage = _function_source("stage_load_path")
    assert "S3 checkpoint resume is prohibited" in stage
    assert "wandb-artifact://" in stage
    assert "use_artifact" in stage
