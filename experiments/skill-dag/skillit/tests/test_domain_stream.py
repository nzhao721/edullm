#!/usr/bin/env python3
"""Smoke tests for DomainMixtureStream.set_weights mid-run reweighting."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SKILLIT = Path(__file__).resolve().parents[1]
if str(_SKILLIT) not in sys.path:
    sys.path.insert(0, str(_SKILLIT))

from domain_stream import DomainMixtureStream  # noqa: E402


@pytest.fixture()
def tiny_pool(tmp_path: Path) -> Path:
    domains = ("dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki")
    seq = 2048
    root = tmp_path / "tokenized"
    for i, d in enumerate(domains):
        ddir = root / d
        ddir.mkdir(parents=True)
        # Two chunks per domain; fill with domain index for identity checks.
        arr = np.full(seq * 2, i, dtype=np.uint32)
        path = ddir / f"{d}.npy"
        mm = np.memmap(path, mode="w+", dtype=np.uint32, shape=(arr.size,))
        mm[:] = arr
        mm.flush()
        del mm
    return tmp_path


def test_set_weights_onehot_samples_only_that_domain(tiny_pool: Path):
    domains = ("dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki")
    p0 = [1.0 / 7.0] * 7
    stream = DomainMixtureStream(tiny_pool, p0, domains=domains, seed=0)
    onehot = [0.0] * 7
    onehot[2] = 1.0  # starcoder
    stream.set_weights(onehot)
    batch = stream.next_input_ids(32)
    assert batch.shape == (32, 2048)
    assert int(batch.min()) == 2 and int(batch.max()) == 2


def test_weights_dict_roundtrip(tiny_pool: Path):
    stream = DomainMixtureStream(
        tiny_pool,
        {"dclm": 0.5, "arxiv": 0.5, "starcoder": 0, "pes2o": 0,
         "open-web-math": 0, "algebraic-stack": 0, "wiki": 0},
        seed=1,
    )
    d = stream.weights_dict()
    assert pytest.approx(d["dclm"] + d["arxiv"], abs=1e-12) == 1.0
    assert d["wiki"] == 0.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # Minimal inline fixture
        root = Path(td)
        domains = ("dclm", "arxiv", "starcoder", "pes2o", "open-web-math", "algebraic-stack", "wiki")
        for i, d in enumerate(domains):
            ddir = root / "tokenized" / d
            ddir.mkdir(parents=True)
            arr = np.full(2048 * 2, i, dtype=np.uint32)
            mm = np.memmap(ddir / f"{d}.npy", mode="w+", dtype=np.uint32, shape=(arr.size,))
            mm[:] = arr
            mm.flush()
            del mm
        test_set_weights_onehot_samples_only_that_domain(root)
        test_weights_dict_roundtrip(root)
    print("ok")
