#!/usr/bin/env python3
"""Patch OLMo classic ``build_memmap_dataset`` to stream from ``DomainMixtureStream``."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from domain_stream import DomainMixtureStream
from mixlaw_common import DOMAINS, SEQ_LEN


class DomainStreamOlmoDataset:
    """Drop-in replacement for ``MemMapDataset`` during DataDecide-60M streaming runs."""

    def __init__(
        self,
        pool_dir: str | Path,
        weights: Mapping[str, float] | Sequence[float],
        *,
        length_tokens: int,
        seed: int,
        domains: Sequence[str] = DOMAINS,
    ) -> None:
        self._stream = DomainMixtureStream(
            pool_dir,
            weights,
            domains=domains,
            seq_len=SEQ_LEN,
            seed=int(seed),
        )
        self._len = max(1, int(length_tokens) // SEQ_LEN)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index  # stochastic stream; OLMo index order is shuffled externally
        row = self._stream.next_input_ids(1)[0].to(dtype=torch.long)
        return {"input_ids": row}

    def set_weights(self, weights: Mapping[str, float] | Sequence[float]) -> None:
        self._stream.set_weights(weights)


_PATCHED = False


def apply_olmo_domain_stream_patch(
    pool_dir: str | Path,
    weights: Mapping[str, float] | Sequence[float],
    *,
    length_tokens: int,
    seed: int,
    domains: Sequence[str] = DOMAINS,
) -> DomainStreamOlmoDataset:
    """Monkey-patch OLMo dataset construction; return the live dataset handle."""
    global _PATCHED
    dataset = DomainStreamOlmoDataset(
        pool_dir,
        weights,
        length_tokens=length_tokens,
        seed=seed,
        domains=domains,
    )

    def _build(_cfg: Any, _data_cfg: Any) -> DomainStreamOlmoDataset:
        return dataset

    import olmo.data as od

    od.build_memmap_dataset = _build  # type: ignore[assignment]
    try:
        import olmo.data.memmap_dataset as mmd

        mmd.build_memmap_dataset = _build  # type: ignore[assignment]
    except Exception:
        pass
    _PATCHED = True
    return dataset
