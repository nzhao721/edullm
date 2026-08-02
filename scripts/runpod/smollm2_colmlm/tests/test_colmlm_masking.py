from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from prepare_annotated_corpus import ShardWriter, annotation_files, token_fact_mask
from train_smollm2_135m_colmlm_ddp import (
    FactMaskedDataset,
    resolve_resume_geometry,
    unwrap_model,
)


def test_half_open_boundaries_and_entire_span_are_masked() -> None:
    offsets = [(0, 3), (3, 5), (5, 8), (8, 10), (0, 0)]
    mask = token_fact_mask(offsets, [(3, 8)])
    assert mask.tolist() == [0, 1, 1, 0, 0]


def test_tokens_crossing_start_and_end_boundaries_are_masked() -> None:
    offsets = [(0, 2), (2, 5), (5, 7), (7, 10), (10, 12)]
    mask = token_fact_mask(offsets, [(3, 8)])
    assert mask.tolist() == [0, 1, 1, 1, 0]


def test_adjacent_and_overlapping_spans_mask_union() -> None:
    offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    mask = token_fact_mask(offsets, [(1, 5), (4, 7), (7, 9)])
    assert mask.tolist() == [1, 1, 1, 1, 1]


def test_writer_keeps_eos_unmasked_and_packs_fixed_sequences(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, seq_len=4, sequences_per_shard=2)
    writer.add_document([10, 11, 12], np.asarray([0, 1, 1], dtype=np.uint8), eos_id=2)
    writer.flush()

    tokens = np.fromfile(tmp_path / "train-00000.tokens.u32le.bin", dtype="<u4")
    mask = np.fromfile(tmp_path / "train-00000.mask.u8.bin", dtype=np.uint8)
    assert tokens.tolist() == [10, 11, 12, 2]
    assert mask.tolist() == [0, 1, 1, 0]
    assert writer.pending_tokens == []


def test_dataset_returns_aligned_ids_and_loss_mask(tmp_path: Path) -> None:
    np.asarray([[10, 11, 12, 13]], dtype="<u4").tofile(
        tmp_path / "train-00000.tokens.u32le.bin"
    )
    np.asarray([[0, 1, 1, 0]], dtype=np.uint8).tofile(
        tmp_path / "train-00000.mask.u8.bin"
    )
    ready = {
        "seq_len": 4,
        "tokens": 4,
        "masked_targets": 2,
        "masked_fraction": 0.5,
        "shards": [
            {
                "tokens": "train-00000.tokens.u32le.bin",
                "mask": "train-00000.mask.u8.bin",
                "sequences": 1,
            }
        ],
    }
    (tmp_path / "_READY.json").write_text(json.dumps(ready), encoding="utf-8")

    batch = FactMaskedDataset(tmp_path)[0]
    labels = batch["input_ids"].clone()
    labels.masked_fill_(batch["loss_mask"], -100)
    assert labels.tolist() == [10, -100, -100, 13]
    assert batch["input_ids"].dtype == torch.int64


def test_annotation_inventory_requires_all_shards(tmp_path: Path) -> None:
    for index in range(2):
        path = tmp_path / f"worker-{index}" / f"train-{index:05d}.annotations.jsonl.zst"
        path.parent.mkdir()
        path.write_bytes(b"fixture")
    assert len(annotation_files(tmp_path, expected_shards=2)) == 2
    with pytest.raises(ValueError, match="expected 3"):
        annotation_files(tmp_path, expected_shards=3)


def test_compile_wrapper_is_unwrapped_for_checkpoints() -> None:
    original = torch.nn.Linear(2, 2)

    class CompiledLike(torch.nn.Module):
        def __init__(self, module: torch.nn.Module):
            super().__init__()
            self._orig_mod = module

    assert unwrap_model(CompiledLike(original)) is original


def test_resume_geometry_exact_match() -> None:
    state = {
        "world_size": 4,
        "per_device_batch_size": 40,
        "seq_len": 2048,
        "global_batch_tokens": 4 * 40 * 2048,
    }
    assert (
        resolve_resume_geometry(
            state, world_size=4, per_device_batch_size=40, seq_len=2048
        )
        == "exact"
    )


def test_resume_geometry_allows_4x40_to_8x20() -> None:
    state = {
        "world_size": 4,
        "per_device_batch_size": 40,
        "seq_len": 2048,
    }
    assert (
        resolve_resume_geometry(
            state, world_size=8, per_device_batch_size=20, seq_len=2048
        )
        == "same_global_batch"
    )


def test_resume_geometry_rejects_changed_global_batch() -> None:
    state = {
        "world_size": 4,
        "per_device_batch_size": 40,
        "seq_len": 2048,
    }
    with pytest.raises(ValueError, match="global batch tokens"):
        resolve_resume_geometry(
            state, world_size=8, per_device_batch_size=40, seq_len=2048
        )


def test_resume_geometry_rejects_changed_seq_len() -> None:
    state = {
        "world_size": 4,
        "per_device_batch_size": 40,
        "seq_len": 2048,
    }
    with pytest.raises(ValueError, match="seq_len"):
        resolve_resume_geometry(
            state, world_size=4, per_device_batch_size=40, seq_len=1024
        )
