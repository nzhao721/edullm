"""Tests for label join + rank assignment in build_curriculum_index."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_curriculum_index import (  # noqa: E402
    assign_ranks,
    build_parent_pool_orders,
    count_doc_chunks,
    join_label_indexes,
    load_parent_layout,
    planned_upload_keys,
)


def test_count_doc_chunks_exact_multiple_and_remainder():
    # Match MemmapTokenDataset: each input needs one next-token reserve.
    assert count_doc_chunks(2048, 2048) == 0
    assert count_doc_chunks(2049, 2048) == 1
    assert count_doc_chunks(4096, 2048) == 1
    assert count_doc_chunks(4097, 2048) == 2
    assert count_doc_chunks(0, 2048) == 0
    assert count_doc_chunks(2047, 2048) == 0
    assert count_doc_chunks(5000, 2048) == 2


def test_join_on_id_and_coverage():
    heur = {
        "a": {"id": "a", "domain": "wiki", "compression_ratio": 2.0, "flesch_reading_ease": 80.0, "mtld": 40.0},
        "b": {"id": "b", "domain": "code", "compression_ratio": 1.2, "flesch_reading_ease": 30.0, "mtld": 90.0},
        "c": {"id": "c", "domain": "wiki", "compression_ratio": 1.5, "flesch_reading_ease": 50.0, "mtld": 60.0},
    }
    lm = {
        "a": {"id": "a", "domain": "wiki", "learnability_late_minus_early_avg_nll": -0.5},
        "b": {"id": "b", "domain": "code", "learnability_late_minus_early_avg_nll": 0.1},
        "d": {"id": "d", "domain": "wiki", "learnability_late_minus_early_avg_nll": -1.0},
    }
    joined, cov = join_label_indexes(heur, lm)
    ids = {r["id"] for r in joined}
    assert ids == {"a", "b"}
    assert cov["n_joined"] == 2
    assert cov["n_heuristic_only"] == 1  # c
    assert cov["n_lm_only"] == 1  # d
    row_a = next(r for r in joined if r["id"] == "a")
    assert row_a["compression_ratio"] == 2.0
    assert row_a["learnability_late_minus_early_avg_nll"] == -0.5


def test_rank_monotonicity_per_metric():
    rows = [
        {"id": "easy_cr", "compression_ratio": 1.0, "flesch_reading_ease": 90.0, "mtld": 10.0,
         "learnability_late_minus_early_avg_nll": -2.0},
        {"id": "mid", "compression_ratio": 2.0, "flesch_reading_ease": 50.0, "mtld": 50.0,
         "learnability_late_minus_early_avg_nll": -0.5},
        {"id": "hard_cr", "compression_ratio": 3.0, "flesch_reading_ease": 10.0, "mtld": 100.0,
         "learnability_late_minus_early_avg_nll": 1.0},
    ]
    ranks = assign_ranks(rows)
    # compression_ratio asc: easy_cr < mid < hard_cr
    assert ranks["compression_ratio"]["easy_cr"] < ranks["compression_ratio"]["hard_cr"]
    # flesch desc (higher = easier): easy flesch rank 0
    assert ranks["flesch"]["easy_cr"] == 0
    assert ranks["flesch"]["hard_cr"] == 2
    # mtld asc
    assert ranks["mtld"]["easy_cr"] < ranks["mtld"]["hard_cr"]
    # learnability asc (more negative = easier)
    assert ranks["learnability"]["easy_cr"] == 0
    assert ranks["learnability"]["hard_cr"] == 2


def test_planned_upload_keys_dry_run(tmp_path: Path):
    (tmp_path / "coverage.json").write_text("{}\n", encoding="utf-8")
    sub = tmp_path / "tokenized" / "wiki"
    sub.mkdir(parents=True)
    (sub / "wiki.npy").write_bytes(b"\x00\x00\x00\x00")
    keys = planned_upload_keys(tmp_path, "s3://edullm-data/curriculum/")
    assert any(k.endswith("coverage.json") for k in keys)
    assert any("tokenized/wiki/wiki.npy" in k for k in keys)
    assert all(k.startswith("s3://edullm-data/curriculum/") for k in keys)
    assert not any("edullm-datasets" in k for k in keys)


def test_build_skip_tokenize_end_to_end(tmp_path: Path):
    """Smoke exact parent-coordinate output without tokenization or network."""
    import subprocess

    labels = tmp_path / "labels"
    lm = tmp_path / "lm_labels"
    for root, rows in (
        (
            labels,
            [
                {"id": "a", "domain": "wiki", "compression_ratio": 1.0, "flesch_reading_ease": 80.0, "mtld": 20.0},
                {"id": "b", "domain": "wiki", "compression_ratio": 2.0, "flesch_reading_ease": 40.0, "mtld": 80.0},
            ],
        ),
        (
            lm,
            [
                {"id": "a", "domain": "wiki", "source_path": "trim/wiki/wiki-trimmed.json.gz",
                 "source_doc": 0, "n_tokens": 3,
                 "learnability_late_minus_early_avg_nll": -1.0},
                {"id": "b", "domain": "wiki", "source_path": "trim/wiki/wiki-trimmed.json.gz",
                 "source_doc": 1, "n_tokens": 4,
                 "learnability_late_minus_early_avg_nll": 0.5},
            ],
        ),
    ):
        root.mkdir(parents=True)
        with gzip.open(root / "metrics_index.jsonl.gz", "wt", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")

    out = tmp_path / "curriculum"
    parent_layout = tmp_path / "parent_layout.json"
    parent_layout.write_text(
        json.dumps(
            {
                "dataset_id": "pretrain/regmix-10b",
                "version": "v7",
                "manifest_sha256": "abc123",
                "seq_len": 4,
                "tokenizer_id": "allenai/dolma2-tokenizer",
                "eos_token_id": 100257,
                "source_total_tokens": {"wiki": 9},
                "shards": [
                    {
                        "path": "tokens/wiki/train-00000.u32le.bin",
                        "source": "wiki",
                        "source_token_start": 0,
                        "count": 9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    script = _SCRIPTS / "build_curriculum_index.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--labels-root",
            str(labels),
            "--lm-labels-root",
            str(lm),
            "--out-dir",
            str(out),
            "--parent-layout",
            str(parent_layout),
            "--parent-version",
            "v7",
            "--parent-manifest-sha256",
            "abc123",
            "--seq-len",
            "4",
            "--dry-run-upload",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out / "curriculum_manifest.json").is_file()
    assert (out / "doc_manifest.jsonl.gz").is_file()
    assert (out / "coverage.json").is_file()
    assert (out / "parent_chunk_index.jsonl.gz").is_file()
    for metric in ("compression_ratio", "flesch", "mtld", "learnability"):
        import numpy as np

        order = np.load(out / f"ranked_chunks_{metric}.npy")
        assert sorted(order.tolist()) == [0, 1]
    assert "dry_run" in proc.stdout


def test_parent_layout_rejects_missing_source_offsets(tmp_path: Path):
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "dataset_id": "pretrain/regmix-10b",
                "version": "v1",
                "manifest_sha256": "hash",
                "seq_len": 2048,
                "tokenizer_id": "allenai/dolma2-tokenizer",
                "eos_token_id": 100257,
                "source_total_tokens": {"wiki": 10},
                "shards": [{"path": "tokens/wiki/train-00000.u32le.bin", "count": 10}],
            }
        ),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(SystemExit, match="source_token_start"):
        load_parent_layout(
            path=layout,
            dataset_id="pretrain/regmix-10b",
            version="v1",
            manifest_sha256="hash",
            seq_len=2048,
        )


def test_parent_orders_fail_closed_on_incomplete_metric(tmp_path: Path):
    rows = [
        {
            "id": "a",
            "domain": "wiki",
            "source_path": "trim/wiki/wiki-trimmed.json.gz",
            "source_doc": 0,
            "n_tokens": 8,
        },
    ]
    ranks = {metric: {} for metric in ("compression_ratio", "flesch", "mtld", "learnability")}
    layout = {
        "source_total_tokens": {"wiki": 9},
        "shards": [
            {
                "path": "tokens/wiki/train-00000.u32le.bin",
                "source": "wiki",
                "source_token_start": 0,
                "count": 9,
                "n_chunks": 2,
            }
        ],
    }
    import pytest

    with pytest.raises(SystemExit, match="lacks a finite"):
        build_parent_pool_orders(
            joined_rows=rows,
            ranks=ranks,
            parent_layout=layout,
            out_dir=tmp_path,
            seq_len=4,
        )
