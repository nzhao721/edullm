"""Focused tests for SkillIt data-source, eval, and launcher hardening."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SKILLIT = Path(__file__).resolve().parents[1]
_PREPARE = _SKILLIT / "prepare_skillit_370m_data.py"
_PROBE_LAUNCHER = _SKILLIT / "launch_probe.sh"
_PROBE_SUBMITTER = _SKILLIT / "submit_skillit_probes.sh"
_WANDB = _SKILLIT / "wandb_logging.py"
_BUILD_ADJACENCY = _SKILLIT / "build_adjacency.py"
_RECIPE = _SKILLIT / "skillit_train_recipe.json"
_PROBES = _SKILLIT / "probes.json"
_PROBE_EVAL = _SKILLIT.parent / "mixlaw" / "eval_task_loss.py"


def _load_prepare():
    spec = importlib.util.spec_from_file_location("skillit_prepare_hardening", _PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_probe_pool(root: Path, *, dataset_id: str, version_key: str, version: str) -> None:
    (root / "tokenized").mkdir(parents=True)
    (root / "edullm_data_source.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                version_key: version,
                "data_bucket": "edullm-data",
            }
        ),
        encoding="utf-8",
    )


def test_skillit_recipes_pin_olmo_127b_v1() -> None:
    for path in (_RECIPE, _PROBES):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["data_source"]["dataset_id"] == "pretrain/olmo-127b"
        assert payload["data_source"]["version"] == "v1"
        assert "olmo-original-30b" not in path.read_text(encoding="utf-8")


def test_pool_provenance_accepts_mixlaw_version_key(tmp_path: Path) -> None:
    prepare = _load_prepare()
    _write_probe_pool(
        tmp_path,
        dataset_id="pretrain/olmo-127b",
        version_key="dataset_version",
        version="v1",
    )
    source = prepare.validate_pool_source(
        tmp_path,
        dataset_id="pretrain/olmo-127b",
        version="v1",
        require_370m_layout=False,
    )
    assert source["version"] == "v1"


def test_pool_provenance_rejects_wrong_or_conflicting_identity(tmp_path: Path) -> None:
    prepare = _load_prepare()
    _write_probe_pool(
        tmp_path,
        dataset_id="pretrain/olmo-original-30b",
        version_key="dataset_version",
        version="v1",
    )
    with pytest.raises(SystemExit, match="does not match pinned"):
        prepare.validate_pool_source(tmp_path)

    (tmp_path / "_EDULLM_DATA_SOURCE.json").write_text(
        json.dumps({"dataset_id": "pretrain/olmo-127b", "version": "v1"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="conflicting pool provenance"):
        prepare.load_pool_source(tmp_path)


def test_launchers_never_write_artifacts_to_s3() -> None:
    for path in (_PROBE_LAUNCHER, _PROBE_SUBMITTER):
        text = path.read_text(encoding="utf-8")
        assert "aws s3 sync" not in text
        assert "RESULTS_S3" not in text
        assert "S3_EXPORT" not in text
        assert "ALLOW_LOCAL_ONLY" in text
        assert "WANDB_MODE" in text
    build_text = _BUILD_ADJACENCY.read_text(encoding="utf-8")
    assert "sync_to_s3" not in build_text
    assert "edullm-checkpoints" not in build_text
    assert "log_artifact" in build_text


def test_checkpoint_artifact_upload_waits_for_wandb(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("skillit_wandb_hardening", _WANDB)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    events: list[str] = []

    class Logged:
        def wait(self) -> None:
            events.append("wait")

    class Artifact:
        def __init__(self, *_: object, **kwargs: object) -> None:
            self.name = str(kwargs.get("name", "artifact"))

        def add_dir(self, _: str) -> None:
            events.append("add_dir")

    class Run:
        def log(self, *_: object, **__: object) -> None:
            events.append("log")

        def log_artifact(self, _: object) -> Logged:
            events.append("log_artifact")
            return Logged()

    checkpoint = tmp_path / "step125"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"checkpoint")
    module.wandb = type("FakeWandb", (), {"Artifact": Artifact})()
    uploaded = module.wandb_log_checkpoint(
        Run(),
        checkpoint,
        step=125,
        tokens_seen=1000,
        arm_id="skillit-probe",
    )
    assert uploaded is True
    assert events[-2:] == ["log_artifact", "wait"]


def test_skillit_recipe_update_schedule_unchanged() -> None:
    recipe = json.loads(_RECIPE.read_text(encoding="utf-8"))
    assert recipe["skillit"] == {
        "update_steps": [500, 875, 1250, 1625, 2000],
        "eta": 0.2,
        "w": 1.0,
    }


def test_60m_eval_reuses_shared_compatible_config_loader() -> None:
    text = _PROBE_EVAL.read_text(encoding="utf-8")
    assert "def load_compatible_train_config(" in text
    assert "module.build_train_config(candidate)" in text
    assert "TrainConfig.load(" not in text
    assert "--base-config" in text
