"""Focused local contracts for the seven-arm platform runtime."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_MIXLAW = Path(__file__).resolve().parents[1]
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

import platform_array_entrypoint as entrypoint  # noqa: E402
import platform_artifacts as artifacts  # noqa: E402
import stage_validation_pool_from_edullm_data as staging  # noqa: E402
from mixlaw_common import DOMAINS  # noqa: E402


def _platform_env(index: str = "0") -> dict[str, str]:
    return {
        "AWS_BATCH_JOB_ARRAY_INDEX": index,
        "AWS_BATCH_JOB_ID": f"batch-child-{index}",
        "EDULLM_RUN_ID": "run-123",
        "EDULLM_DATASET_ID": "pretrain/olmo-127b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_CHECKPOINT_DIR": "s3://outputs/teams/pre-training/runs/run-123/checkpoints/",
        "EDULLM_OUTPUT_PREFIX": "s3://outputs/teams/pre-training/runs/run-123/",
        "EDULLM_WANDB_PROJECT": "mixlaw",
    }


def test_array_manifest_is_explicit_complete_and_excludes_mix01() -> None:
    resolved = [entrypoint.array_arm(str(index)) for index in range(7)]
    assert [name for _, _, name in resolved] == [
        "olmo-mix-1124",
        "mix07",
        "mix18",
        "ML-pilot_caps",
        "ML-near-opt-4",
        "LGB-min1pct",
        "LGB-near-opt-8",
    ]
    assert len({mix_id for _, mix_id, _ in resolved}) == 7
    assert "mix01" not in {name for _, _, name in resolved}


@pytest.mark.parametrize("value", [None, "", "-1", "7", "01", "x", "1.0"])
def test_array_index_is_required_and_bounded(value: str | None) -> None:
    with pytest.raises(entrypoint.PlatformLaunchError):
        entrypoint.array_arm(value)


def test_platform_launch_isolated_paths_and_bounded_threads(tmp_path: Path) -> None:
    launch = entrypoint.prepare_launch(_platform_env("3"), scratch_root=tmp_path)
    env = launch.environment
    assert launch.mix_name == "ML-pilot_caps"
    assert launch.scratch_dir == tmp_path / "batch-child-3"
    assert env["SAVE_FOLDER"].startswith(str(launch.scratch_dir))
    assert env["POOL_DIR"].startswith(str(launch.scratch_dir))
    assert env["NPROC"] == "8"
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
    assert int(env["NPROC"]) * int(env["OMP_NUM_THREADS"]) == 16
    assert env["DATASET_VERSION"] == "v1"
    assert env["WANDB_PROJECT"] == "mixlaw"
    assert env["WANDB_GROUP"] == "370m-validation"
    assert env["WANDB_RUN_ID"].endswith("-ML-pilot_caps")
    assert env["CHECKPOINT_PREFIX"].endswith("/array/03-ML-pilot_caps/")
    assert env["OUTPUT_PREFIX"].endswith("/array/03-ML-pilot_caps/")


def test_platform_refuses_unpinned_dataset_or_wrong_project(tmp_path: Path) -> None:
    for key, value in (
        ("EDULLM_DATASET_VERSION", "latest"),
        ("EDULLM_DATASET_ID", "pretrain/other"),
        ("EDULLM_WANDB_PROJECT", "other"),
    ):
        env = _platform_env()
        env[key] = value
        with pytest.raises(entrypoint.PlatformLaunchError):
            entrypoint.prepare_launch(env, scratch_root=tmp_path)


def test_selected_sidecar_is_deterministic_and_arm_specific(tmp_path: Path) -> None:
    recipe = _MIXLAW / "validation_mixtures_10b.json"
    first = entrypoint.write_selected_sidecar(
        recipe,
        tmp_path / "one.json",
        mix_id=7,
        mix_name="mix07",
        dataset_id="pretrain/olmo-127b",
        dataset_version="v1",
    )
    second = entrypoint.write_selected_sidecar(
        recipe,
        tmp_path / "two.json",
        mix_id=7,
        mix_name="mix07",
        dataset_id="pretrain/olmo-127b",
        dataset_version="v1",
    )
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["run_name"] == "mix07"
    assert payload["stream_seed"] == 6198 + 7
    assert payload["dataset_version"] == "v1"


def _inventory(tokens: int = 100) -> dict[str, dict]:
    return {
        domain: {
            "shards": [
                {
                    "uri": f"s3://bucket/{domain}/train-{index:02d}.bin",
                    "tokens": tokens,
                    "bytes": tokens * 4,
                }
                for index in range(8)
            ]
        }
        for domain in DOMAINS
    }


def test_selected_arm_demand_and_shard_prefixes_are_deterministic() -> None:
    recipe = _MIXLAW / "validation_mixtures_10b.json"
    selected_demand = staging.arm_tokens_from_mixtures(recipe, 10_000_000_000, "mix07")
    shared_demand = staging.peak_tokens_from_mixtures(recipe, 10_000_000_000)
    assert all(selected_demand[domain] <= shared_demand[domain] for domain in DOMAINS)

    selected_arm = staging.select_shards(
        _inventory(),
        {domain: 200 for domain in DOMAINS},
        6198,
        deterministic_prefix=True,
    )
    legacy_shared_pool = staging.select_shards(
        _inventory(),
        {domain: 500 for domain in DOMAINS},
        6198,
        deterministic_prefix=False,
    )
    repeated = staging.select_shards(
        _inventory(),
        {domain: 200 for domain in DOMAINS},
        6198,
        deterministic_prefix=True,
    )
    for domain in DOMAINS:
        selected_uris = [row["uri"] for row in selected_arm[domain]]
        assert selected_uris == [row["uri"] for row in repeated[domain]]
        assert selected_uris == [
            row["uri"] for row in legacy_shared_pool[domain]
        ][: len(selected_uris)]


def test_selected_stage_deletes_temporary_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inventory(
        dataset_id: str,
        version: str,
        domain: str,
        *,
        s3: object,
    ) -> dict:
        del dataset_id, version, s3
        tokens = staging.SEQ_LEN * 2
        return {
            "domain": domain,
            "dtype": "uint32",
            "byte_order": "little",
            "header_bytes": 0,
            "rows": tokens,
            "shards": [
                {
                    "uri": f"s3://bucket/{domain}/train-000.bin",
                    "bucket": "bucket",
                    "key": f"{domain}/train-000.bin",
                    "bytes": tokens * 4,
                    "tokens": tokens,
                    "name": "train-000.bin",
                }
            ],
        }

    def download(_: object, __: dict, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.arange(staging.SEQ_LEN * 2, dtype=np.uint32).tofile(destination)

    monkeypatch.setattr(staging, "resolve_dataset", lambda dataset_id, version, **_: (dataset_id, version))
    monkeypatch.setattr(staging, "domain_train_shards", inventory)
    monkeypatch.setattr(staging, "download_shard", download)
    out = tmp_path / "pool"
    summary = staging.stage_pool(
        out_dir=out,
        mixtures_json=_MIXLAW / "validation_mixtures_10b.json",
        budget_tokens=staging.SEQ_LEN,
        dataset_id="pretrain/olmo-127b",
        dataset_version="v1",
        mix_name="mix07",
        deterministic_prefix=True,
        delete_shards=True,
        s3=object(),
    )
    assert summary["mix_name"] == "mix07"
    assert summary["selection"] == "deterministic_domain_prefix"
    assert not (out / "shards").exists()
    assert staging.pool_is_ready(out)


class _FakeS3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def upload_file(self, local: str, bucket: str, key: str) -> None:
        assert Path(local).is_file()
        self.calls.append((bucket, key))

    def put_object(self, *, Bucket: str, Key: str, **_: object) -> None:
        self.calls.append((Bucket, Key))


def test_platform_artifacts_use_isolated_progress_taskloss_and_sentinel(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    task_loss = tmp_path / "task-loss"
    checkpoint = tmp_path / "step10"
    for path, text in (
        (progress / "run_meta.json", "{}"),
        (progress / "wandb" / "private.tmp", "skip"),
        (task_loss / "step10_task_loss.json", "{}"),
        (checkpoint / "state.pt", "state"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    fake = _FakeS3()
    artifacts.upload_run_outputs(
        progress,
        task_loss,
        "s3://outputs/run/array/00-mix07/",
        client=fake,
    )
    uri = artifacts.upload_checkpoint(
        checkpoint,
        "s3://outputs/run/checkpoints/array/00-mix07/step10/",
        step=10,
        mix_name="mix07",
        client=fake,
    )
    keys = [key for _, key in fake.calls]
    assert "run/array/00-mix07/progress/run_meta.json" in keys
    assert "run/array/00-mix07/task-loss/step10_task_loss.json" in keys
    assert not any("wandb" in key for key in keys)
    assert keys[-1].endswith("/_COMPLETE.json")
    assert uri.endswith("/step10/")


def test_container_and_legacy_launch_contracts_are_checked_in() -> None:
    repo = _MIXLAW.parents[2]
    dockerfile = (repo / ".edullm" / "Dockerfile").read_text(encoding="utf-8")
    lock = (repo / ".edullm" / "requirements-linux-cu128.lock").read_text(
        encoding="utf-8"
    )
    workflow = (
        repo / ".github" / "workflows" / "publish-research-image.yml"
    ).read_text(encoding="utf-8")
    launcher = (_MIXLAW / "launch_validation_370m.sh").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}" in dockerfile
    assert "platform_array_entrypoint.py" in dockerfile
    assert "torch==2.8.0+cu128" in lock
    assert "ai2-olmo-core==2.4.0" in lock
    assert "workflow_dispatch:" in workflow
    assert "repository: edullm-p1" in workflow
    assert 'NPROC="${NPROC:-1}"' in launcher
    assert "SLURM_JOB_ID" not in entrypoint.__file__
    assert 'if [[ "${MIXLAW_PLATFORM:-0}" == "1" ]]' in launcher
