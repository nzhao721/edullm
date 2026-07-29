#!/usr/bin/env python3
"""Olmohq domain-stratified chunk sampler with mid-run ``set_weights(p)``.

At each draw: sample domain i ~ Categorical(p), then a random contiguous
``seq_len``-token chunk from that domain's uint32 memmap. Weights can change
between draws without rebuilding the pool.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

DEFAULT_DOMAINS: tuple[str, ...] = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)
SEQ_LEN = 2048
MEMMAP_DTYPE = np.uint32


def _resolve_domain_npy(pool_dir: Path, domain: str) -> Path:
    """Accept ``tokenized/<d>/<d>.npy`` or flat ``<d>.npy`` under pool_dir."""
    candidates = [
        pool_dir / "tokenized" / domain / f"{domain}.npy",
        pool_dir / domain / f"{domain}.npy",
        pool_dir / f"{domain}.npy",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"no memmap for domain {domain!r} under {pool_dir} "
        f"(tried tokenized/{domain}/{domain}.npy and flat layouts)"
    )


class DomainMixtureStream:
    """Domain-stratified infinite stream over an olmohq working pool."""

    def __init__(
        self,
        pool_dir: Union[str, Path],
        weights: Union[Sequence[float], Mapping[str, float]],
        *,
        domains: Sequence[str] = DEFAULT_DOMAINS,
        seq_len: int = SEQ_LEN,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.pool_dir = Path(pool_dir)
        self.domains = tuple(domains)
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._rng = np.random.default_rng(int(seed) + 1_000_003 * self.rank)

        self._mmaps: Dict[str, np.memmap] = {}
        self._n_chunks: Dict[str, int] = {}
        for d in self.domains:
            path = _resolve_domain_npy(self.pool_dir, d)
            mm = np.memmap(path, mode="r", dtype=MEMMAP_DTYPE)
            n = len(mm) // self.seq_len
            if n <= 0:
                raise SystemExit(f"domain {d}: memmap too short for seq_len={self.seq_len}")
            self._mmaps[d] = mm
            self._n_chunks[d] = int(n)

        self._p = np.zeros(len(self.domains), dtype=np.float64)
        self.set_weights(weights)

    @property
    def weights(self) -> np.ndarray:
        return self._p.copy()

    def weights_dict(self) -> Dict[str, float]:
        return {d: float(self._p[i]) for i, d in enumerate(self.domains)}

    def set_weights(self, weights: Union[Sequence[float], Mapping[str, float]]) -> None:
        """Update domain sampling distribution (renormalized to a simplex)."""
        if isinstance(weights, Mapping):
            vec = np.array([float(weights.get(d, 0.0)) for d in self.domains], dtype=np.float64)
        else:
            vec = np.asarray(weights, dtype=np.float64).reshape(-1)
            if vec.shape[0] != len(self.domains):
                raise ValueError(
                    f"weights length {vec.shape[0]} != n_domains {len(self.domains)}"
                )
        if np.any(vec < 0):
            raise ValueError("domain weights must be non-negative")
        # Domains with no chunks cannot receive mass.
        for i, d in enumerate(self.domains):
            if self._n_chunks[d] <= 0:
                vec[i] = 0.0
        s = float(vec.sum())
        if s <= 0.0:
            raise ValueError("domain weights sum to 0")
        self._p = vec / s

    def _sample_chunk(self, domain: str) -> np.ndarray:
        n = self._n_chunks[domain]
        # Rank striping: each rank draws from a disjoint residue class when possible.
        if self.world_size > 1 and n >= self.world_size:
            # Sample uniformly among chunks with index ≡ rank (mod world_size).
            n_local = (n - self.rank + self.world_size - 1) // self.world_size
            local = int(self._rng.integers(0, n_local))
            idx = local * self.world_size + self.rank
        else:
            idx = int(self._rng.integers(0, n))
        start = idx * self.seq_len
        mm = self._mmaps[domain]
        return np.asarray(mm[start : start + self.seq_len], dtype=np.int64)

    def next_input_ids(self, n_seqs: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Draw ``n_seqs`` sequences under current ``p``; return ``[n_seqs, seq_len]``."""
        n = int(n_seqs)
        if n <= 0:
            raise ValueError("n_seqs must be > 0")
        domain_ids = self._rng.choice(len(self.domains), size=n, p=self._p)
        rows: List[np.ndarray] = []
        for di in domain_ids:
            rows.append(self._sample_chunk(self.domains[int(di)]))
        arr = np.stack(rows, axis=0)
        t = torch.from_numpy(arr.copy())
        if device is not None:
            t = t.to(device, non_blocking=True)
        return t
