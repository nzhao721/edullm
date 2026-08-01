"""Tests for staging curriculum token-order groups before edullm-data publish."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REGMIX = Path(__file__).resolve().parents[3] / "datasets" / "regmix"
if str(_REGMIX) not in sys.path:
    sys.path.insert(0, str(_REGMIX))

from publish_regmix_curriculum_edullm_data import (  # noqa: E402
    _load_index_manifest,
    stage_curriculum_orders,
)


def test_stage_curriculum_orders_writes_four_groups(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    n = 8
    for metric in ("compression_ratio", "flesch", "mtld", "learnability"):
        order = np.arange(n - 1, -1, -1, dtype=np.uint32)
        np.save(index_dir / f"ranked_chunks_{metric}.npy", order)

    out = tmp_path / "stage"
    lengths = stage_curriculum_orders(
        index_dir=index_dir, out_root=out, expected_block_count=n
    )
    assert set(lengths) == {"compression", "flesch", "mtld", "learnability"}
    assert all(v == n for v in lengths.values())

    for group in lengths:
        train = out / group / "train-00000.u32le.bin"
        val = out / group / "val-00000.u32le.bin"
        assert train.is_file() and val.is_file()
        ranked = np.frombuffer(train.read_bytes(), dtype="<u4")
        identity = np.frombuffer(val.read_bytes(), dtype="<u4")
        assert ranked.shape == (n,)
        assert identity.shape == (n,)
        assert np.array_equal(identity, np.arange(n, dtype=np.uint32))
        assert np.array_equal(ranked, np.arange(n - 1, -1, -1, dtype=np.uint32))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0, 1, 1, 3], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2, 4], dtype=np.int64),
        np.array([-1, 0, 1, 2], dtype=np.int64),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    ],
)
def test_stage_rejects_non_parent_permutations(tmp_path: Path, bad: np.ndarray) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    for metric in ("compression_ratio", "flesch", "mtld", "learnability"):
        np.save(index_dir / f"ranked_chunks_{metric}.npy", bad)
    with pytest.raises(SystemExit):
        stage_curriculum_orders(
            index_dir=index_dir,
            out_root=tmp_path / "stage",
            expected_block_count=4,
        )


def test_publish_rejects_legacy_document_local_manifest(tmp_path: Path) -> None:
    (tmp_path / "curriculum_manifest.json").write_text(
        '{"version": 1, "tokenize": {"n_chunks": 12}}\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="legacy/document-local"):
        _load_index_manifest(tmp_path)
