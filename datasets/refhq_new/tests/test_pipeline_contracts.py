"""Contract tests across plan / holdout / tokenize / finalize / sbatch scripts."""

from __future__ import annotations

import ast
import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from finalize_upload import (
    collect_pair_report,
    resolve_tok_root,
    split_npy_to_shards,
    stage_tokens_from_tok,
)
from holdout_docs import run_holdout
from refhq_new.domain_map import DOMAINS, SOURCES
from refhq_new_sources import HOLDOUT_FRACTION, scratch_layout
from tokenize_source import _load_task

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REPO_ROOT = ROOT.parents[1]


def test_plan_cli_writes_tokenize_tasks(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "plan_refhq_new.py"),
            "--scratch-root",
            str(scratch),
            "--sources",
            "tulu-v2",
            "openhermes-25",
            "--seed",
            "7",
            "--s3-bucket",
            "edullm-datasets",
            "--s3-prefix",
            "refhq/refhq-new",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in proc.stdout
    plan = json.loads((scratch / "manifests" / "plan.json").read_text(encoding="utf-8"))
    assert plan["holdout_fraction"] == HOLDOUT_FRACTION
    assert plan["seed"] == 7
    assert set(plan["sources"]) == {"tulu-v2", "openhermes-25"}
    assert plan["layout"]["tokenized_npy"].startswith("tokenized/")
    assert "tokens/<source>/<domain>/" in plan["layout"]["tokens_publish"]

    tasks = (scratch / "manifests" / "tokenize_tasks.txt").read_text(encoding="utf-8").splitlines()
    assert tasks
    for line in tasks:
        parts = line.split()
        assert len(parts) == 3
        source, domain, split = parts
        assert source in {"tulu-v2", "openhermes-25"}
        assert domain in DOMAINS
        assert split in {"train", "val"}
    # 2 sources × 5 domains × 2 splits
    assert len(tasks) == 2 * len(DOMAINS) * 2


def test_plan_source_list_flag(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "plan_refhq_new.py"),
            "--scratch-root",
            str(scratch),
            "--source-list",
            "tulu-3 hermes-3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads((scratch / "manifests" / "plan.json").read_text(encoding="utf-8"))
    assert list(plan["sources"]) == ["tulu-3", "hermes-3"]


def test_holdout_refreshes_tokenize_tasks(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    layout = scratch_layout(scratch)
    for key in layout:
        layout[key].mkdir(parents=True, exist_ok=True)

    # Minimal plan with one source; write 100 synthetic English docs under out/.
    out_dir = layout["out"] / "tulu-v2" / "general"
    out_dir.mkdir(parents=True)
    with gzip.open(out_dir / "documents-00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i in range(100):
            handle.write(
                json.dumps(
                    {
                        "id": f"tulu-v2:{i}",
                        "text": f"User: q{i}\n\nAssistant: a{i}",
                        "source": "tulu-v2",
                        "metadata": {"domain": "general"},
                    }
                )
                + "\n"
            )

    plan = {
        "scratch_root": str(scratch),
        "holdout_fraction": 0.0015,
        "seed": 42,
        "sources": {
            "tulu-v2": {
                "paths": {
                    "out": str(layout["out"] / "tulu-v2"),
                    "holdout": str(layout["holdout"] / "tulu-v2"),
                }
            }
        },
    }
    (layout["manifests"] / "plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
    # Placeholder tasks (should be replaced).
    (layout["manifests"] / "tokenize_tasks.txt").write_text(
        "tulu-v2 general train\ntulu-v2 math val\n",
        encoding="utf-8",
    )

    summary = run_holdout(plan)
    assert summary["pairs"]["tulu-v2/general"]["n_docs"] == 100
    # 100 * 0.0015 = 0.15 → round to 0 val; train-only tasks
    assert summary["pairs"]["tulu-v2/general"]["n_val"] == 0
    assert summary["pairs"]["tulu-v2/general"]["n_train"] == 100
    tasks = (layout["manifests"] / "tokenize_tasks.txt").read_text(encoding="utf-8").strip().splitlines()
    assert tasks == ["tulu-v2 general train documents-00000.jsonl.gz"]

    # Larger pool so val appears.
    with gzip.open(out_dir / "documents-00000.jsonl.gz", "wt", encoding="utf-8") as handle:
        for i in range(10_000):
            handle.write(json.dumps({"id": f"id{i}", "text": f"doc {i}"}) + "\n")
    summary2 = run_holdout(plan)
    assert summary2["pairs"]["tulu-v2/general"]["n_val"] == 15
    tasks2 = (layout["manifests"] / "tokenize_tasks.txt").read_text(encoding="utf-8").strip().splitlines()
    assert any(t.startswith("tulu-v2 general train ") for t in tasks2)
    assert any(t.startswith("tulu-v2 general val ") for t in tasks2)


def test_load_task_index_contract(tmp_path: Path) -> None:
    tasks = tmp_path / "tokenize_tasks.txt"
    tasks.write_text(
        "tulu-v2 math train documents-00000.jsonl.gz\n"
        "smoltalk code val documents-00001.jsonl.gz\n"
        "\n"
        "dolci science train documents-00000.jsonl.gz\n",
        encoding="utf-8",
    )
    assert _load_task(tasks, 0) == ("tulu-v2", "math", "train", "documents-00000.jsonl.gz")
    assert _load_task(tasks, 1) == ("smoltalk", "code", "val", "documents-00001.jsonl.gz")
    assert _load_task(tasks, 2) == ("dolci", "science", "train", "documents-00000.jsonl.gz")
    with pytest.raises(SystemExit):
        _load_task(tasks, 3)


def test_finalize_prefers_tokenized_over_tok(tmp_path: Path) -> None:
    scratch = tmp_path / "run"
    (scratch / "tokenized" / "tulu-v2" / "general").mkdir(parents=True)
    (scratch / "tok" / "tulu-v2" / "general").mkdir(parents=True)
    assert resolve_tok_root(scratch) == scratch / "tokenized"
    # tok-only still works
    scratch2 = tmp_path / "run2"
    (scratch2 / "tok" / "hermes-3" / "chat").mkdir(parents=True)
    assert resolve_tok_root(scratch2) == scratch2 / "tok"


def test_finalize_stages_npy_to_u32le(tmp_path: Path) -> None:
    import numpy as np

    tok_root = tmp_path / "tokenized"
    domain_dir = tok_root / "tulu-v2" / "general"
    domain_dir.mkdir(parents=True)
    ids = np.array([11, 22, 33, 100257], dtype=np.uint32)
    npy = domain_dir / "train.npy"
    npy.write_bytes(ids.tobytes())
    (domain_dir / "train.json").write_text(
        json.dumps({"stream_tokens_with_eos": 4, "docs": 1}) + "\n",
        encoding="utf-8",
    )
    report = collect_pair_report(domain_dir)
    assert report["train"]["stream_tokens_with_eos"] == 4
    assert report["train"]["kind"] == "npy"

    stage = tmp_path / "stage"
    staged = stage_tokens_from_tok(
        tok_root=tok_root,
        out_root=stage,
        shard_bytes=1_073_741_824,
        force=True,
    )
    assert "tulu-v2/general" in staged
    shard = stage / "tokens" / "tulu-v2" / "general" / "train-00000.u32le.bin"
    assert shard.is_file()
    assert shard.stat().st_size == 16
    # Direct split helper
    out = tmp_path / "shards"
    paths = split_npy_to_shards(npy, out, shard_bytes=8, split="train")
    assert len(paths) == 2
    assert paths[0].name == "train-00000.u32le.bin"


def test_tokenize_sbatch_uses_task_index() -> None:
    text = (SCRIPTS / "tokenize_source.sbatch").read_text(encoding="utf-8")
    assert "--task-index" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "--all-for-source" not in text
    assert "SOURCES[${SLURM_ARRAY_TASK_ID}]" not in text


def test_submit_exports_match_sbatch_env() -> None:
    submit = (SCRIPTS / "submit_refhq_new.sh").read_text(encoding="utf-8")
    assert "TOKENIZE_TASKS=" in submit
    assert "ENGLISH_TASKS=" in submit
    assert "build_english_tasks.py" in submit
    assert "merge_tokenized.sbatch" in submit
    assert "--array=0-$((N_TASKS - 1))" in submit
    assert "--array=0-$((N_ENG - 1))" in submit
    assert "tokenize_source.sbatch" in submit
    for name in (
        "download_hf_source.sbatch",
        "normalize_filter_source.sbatch",
        "dolma_english_filter.sbatch",
        "holdout_docs.sbatch",
        "finalize_upload.sbatch",
        "publish_refhq_new.sbatch",
    ):
        assert name in submit
    # Download + normalize remain source-indexed arrays
    assert submit.count("--array=0-$((N - 1))") >= 2


def test_english_sbatch_uses_task_index() -> None:
    text = (SCRIPTS / "dolma_english_filter.sbatch").read_text(encoding="utf-8")
    assert "--task-index" in text
    assert "ENGLISH_TASKS" in text
    assert "SOURCES[${SLURM_ARRAY_TASK_ID}]" not in text
    assert "--cpus-per-task=4" in text


def test_script_modules_import_cleanly() -> None:
    """Import smoke for FarmShare entrypoints (no network)."""
    modules = [
        "plan_refhq_new",
        "download_hf_source",
        "normalize_filter_source",
        "dolma_english_filter",
        "holdout_docs",
        "tokenize_source",
        "finalize_upload",
        "publish_refhq_new",
        "refhq_new_sources",
    ]
    for name in modules:
        path = SCRIPTS / f"{name}.py"
        # Compile-check avoids executing network-y main()
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        # Also ensure top-level import works for modules we use in tests
        if name in {
            "refhq_new_sources",
            "holdout_docs",
            "tokenize_source",
            "finalize_upload",
            "normalize_filter_source",
            "plan_refhq_new",
        }:
            __import__(name)


def test_lib_sh_helpers_present() -> None:
    text = (SCRIPTS / "lib.sh").read_text(encoding="utf-8")
    for fn in (
        "refhq_new_stage_shared_utils",
        "refhq_new_sync_to_run",
        "refhq_new_export_pythonpath",
        "refhq_new_load_hf_token",
    ):
        assert fn in text
    # Shared utils include TokenWriter dependency
    assert "trim_and_tokenize_regmix.py" in text
    assert "olmo_shard_utils.py" in text
    assert "edullm_text_companion.py" in text


def test_normalize_fixture_applies_exclusions(tmp_path: Path) -> None:
    from normalize_filter_source import _iter_fixture_rows, normalize_rows

    fixture = tmp_path / "rows.jsonl"
    fixture.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "keep",
                        "source": "ai2-adapt-dev/flan_v2_converted",
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "id": "drop",
                        "source": "allenai/wildguardmix",
                        "messages": [{"role": "user", "content": "x"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    stats = normalize_rows(
        source="tulu-3",
        repo_id="allenai/tulu-3-sft-mixture",
        docs_root=docs,
        rows=_iter_fixture_rows(fixture),
    )
    assert stats["kept"] == 1
    assert stats["dropped_metadata"] == 1
    assert stats["domain_counts"]["general"] == 1
    assert (docs / "general" / "documents-00000.jsonl.gz").is_file()


def test_sources_constant_matches_submit_default() -> None:
    submit = (SCRIPTS / "submit_refhq_new.sh").read_text(encoding="utf-8")
    match = re.search(r'SOURCE_LIST="\$\{SOURCE_LIST:-([^}]+)\}"', submit)
    assert match
    default_sources = match.group(1).split()
    assert tuple(default_sources) == SOURCES
