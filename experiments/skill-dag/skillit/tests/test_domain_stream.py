#!/usr/bin/env python3
"""Smoke tests for mixlaw DomainMixtureStream (shared with skillit)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SKILLIT = Path(__file__).resolve().parents[1]
_MIXLAW = _SKILLIT.parent / "mixlaw"
for _p in (_MIXLAW, _SKILLIT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from domain_stream import DomainMixtureStream  # noqa: E402

DOMAINS = ("dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki")
SEQ = 2048


@pytest.fixture()
def tiny_pool(tmp_path: Path) -> Path:
    """Legacy single-file ``tokenized/<d>/<d>.npy`` layout (mixlaw peak-pool style)."""
    root = tmp_path / "tokenized"
    for i, d in enumerate(DOMAINS):
        ddir = root / d
        ddir.mkdir(parents=True)
        arr = np.full(SEQ * 2, i, dtype=np.uint32)
        path = ddir / f"{d}.npy"
        mm = np.memmap(path, mode="w+", dtype=np.uint32, shape=(arr.size,))
        mm[:] = arr
        mm.flush()
        del mm
    return tmp_path


@pytest.fixture()
def multi_shard_pool(tmp_path: Path) -> Path:
    """Canonical skillit layout: ``tokens/<d>/train-*.u32le.bin``."""
    for i, d in enumerate(DOMAINS):
        ddir = tmp_path / "tokens" / d
        ddir.mkdir(parents=True)
        # Two shards × one chunk each; fill with domain index.
        for shard_i in range(2):
            arr = np.full(SEQ, i, dtype=np.uint32)
            path = ddir / f"train-{shard_i:05d}.u32le.bin"
            mm = np.memmap(path, mode="w+", dtype="<u4", shape=(arr.size,))
            mm[:] = arr
            mm.flush()
            del mm
    return tmp_path


def test_set_weights_onehot_samples_only_that_domain(tiny_pool: Path):
    p0 = [1.0 / 7.0] * 7
    stream = DomainMixtureStream(tiny_pool, p0, domains=DOMAINS, seed=0)
    onehot = [0.0] * 7
    onehot[2] = 1.0  # starcoder
    stream.set_weights(onehot)
    batch = stream.next_input_ids(32)
    assert batch.shape == (32, SEQ)
    assert int(batch.min()) == 2 and int(batch.max()) == 2


def test_weights_dict_roundtrip(tiny_pool: Path):
    stream = DomainMixtureStream(
        tiny_pool,
        {"dclm": 0.5, "arxiv": 0.5, "starcoder": 0, "pes2o": 0,
         "open-web-math": 0, "algebraic-stack": 0, "wiki": 0},
        domains=DOMAINS,
        seed=1,
    )
    d = stream.weights_dict()
    assert pytest.approx(d["dclm"] + d["arxiv"], abs=1e-12) == 1.0
    assert d["wiki"] == 0.0


def test_multi_shard_tokens_layout_onehot(multi_shard_pool: Path):
    stream = DomainMixtureStream(
        multi_shard_pool,
        {d: (1.0 if d == "wiki" else 0.0) for d in DOMAINS},
        domains=DOMAINS,
        seed=3,
    )
    batch = stream.next_input_ids(16)
    assert batch.shape == (16, SEQ)
    wiki_idx = DOMAINS.index("wiki")
    assert int(batch.min()) == wiki_idx and int(batch.max()) == wiki_idx
    # Two shards ⇒ two chunks available under one-hot wiki.
    assert stream._n_chunks["wiki"] == 2


def test_import_is_mixlaw_module():
    import domain_stream as ds

    assert Path(ds.__file__).resolve().parent == _MIXLAW.resolve()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i, d in enumerate(DOMAINS):
            ddir = root / "tokenized" / d
            ddir.mkdir(parents=True)
            arr = np.full(SEQ * 2, i, dtype=np.uint32)
            mm = np.memmap(ddir / f"{d}.npy", mode="w+", dtype=np.uint32, shape=(arr.size,))
            mm[:] = arr
            mm.flush()
            del mm
        test_set_weights_onehot_samples_only_that_domain(root)
        test_weights_dict_roundtrip(root)
    print("ok")
