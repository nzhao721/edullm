#!/usr/bin/env python3
"""Domain-stratified chunk sampler with optional mid-run ``set_weights(p)``.

Supports staged working pools from published ``edullm-data`` pretrain corpora:

  * ``tokens/<domain>/train-*.u32le.bin`` (skillit / multi-shard layout)
  * ``tokenized/<domain>/train-*.u32le.bin``
  * single-file ``tokenized/<domain>/<domain>.{u32le.bin,npy}`` (mixlaw peak-pool concat)
  * legacy flat ``<domain>.{u32le.bin,npy}`` (tests)

At each draw: sample domain i ~ Categorical(p), then a random contiguous
``seq_len``-token chunk from that domain's uint32 memmap(s). Weights can change
between draws without rebuilding the pool.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

_MIXLAW = Path(__file__).resolve().parent
if str(_MIXLAW) not in sys.path:
    sys.path.insert(0, str(_MIXLAW))

from mixlaw_common import DOMAINS, MEMMAP_DTYPE as _MEMMAP_DTYPE_RAW, SEQ_LEN  # noqa: E402

DEFAULT_DOMAINS: tuple[str, ...] = DOMAINS
# Published pretrain shards are little-endian uint32; keep that explicit.
MEMMAP_DTYPE = np.dtype("<u4")


def _coerce_dtype(dtype: Union[str, np.dtype]) -> np.dtype:
    d = np.dtype(dtype)
    if d == np.dtype("uint32") or d == np.dtype("<u4") or str(d) in ("uint32", "<u4", "u4"):
        return MEMMAP_DTYPE
    return d


def _shard_paths(pool_dir: Path, domain: str) -> List[Path]:
    """Resolve local shard file(s) for one domain under ``pool_dir``.

    Prefers multi-shard ``train-*.u32le.bin`` layouts, then single-file
    ``*.u32le.bin`` / ``<domain>.npy`` under the usual staging roots.
    """
    candidates_dirs = [
        pool_dir / "tokens" / domain,
        pool_dir / "tokenized" / domain,
        pool_dir / domain,
    ]
    for ddir in candidates_dirs:
        if not ddir.is_dir():
            continue
        found = sorted(ddir.glob("train-*.u32le.bin"))
        if not found:
            # Single-file concat pools: ``<domain>.u32le.bin`` (and any other *.u32le.bin).
            found = sorted(ddir.glob("*.u32le.bin"))
        if found:
            shards = [p for p in found if not p.name.startswith("val-")]
            if shards:
                return shards
        # Named single-file preference matches legacy mixlaw resolve order.
        for name in (f"{domain}.u32le.bin", f"{domain}.npy"):
            named = ddir / name
            if named.is_file():
                return [named]

    for name in (f"{domain}.u32le.bin", f"{domain}.npy"):
        flat = pool_dir / name
        if flat.is_file():
            return [flat]

    raise FileNotFoundError(
        f"no memmap shards for domain {domain!r} under {pool_dir} "
        f"(tried tokens/{domain}/train-*.u32le.bin, tokenized/{domain}/, flat layouts)"
    )


class DomainMixtureStream:
    """Domain-stratified infinite stream over a staged domain working pool."""

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
        dtype: Union[str, np.dtype] = _MEMMAP_DTYPE_RAW,
    ) -> None:
        self.pool_dir = Path(pool_dir)
        self.domains = tuple(domains)
        self.seq_len = int(seq_len)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.dtype = _coerce_dtype(dtype)
        self._rng = np.random.default_rng(int(seed) + 1_000_003 * self.rank)

        # Per domain: list of memmaps + token lengths; chunk index spans shards.
        self._mmaps: Dict[str, List[np.memmap]] = {}
        self._shard_tokens: Dict[str, List[int]] = {}
        self._n_chunks: Dict[str, int] = {}
        for d in self.domains:
            paths = _shard_paths(self.pool_dir, d)
            mmaps: List[np.memmap] = []
            lengths: List[int] = []
            total_chunks = 0
            for path in paths:
                mm = np.memmap(path, mode="r", dtype=self.dtype)
                n_tok = int(mm.shape[0])
                n_chunks = n_tok // self.seq_len
                if n_chunks <= 0:
                    continue
                mmaps.append(mm)
                lengths.append(n_tok)
                total_chunks += n_chunks
            if total_chunks <= 0:
                raise SystemExit(
                    f"domain {d}: memmap(s) too short for seq_len={self.seq_len} under {self.pool_dir}"
                )
            self._mmaps[d] = mmaps
            self._shard_tokens[d] = lengths
            self._n_chunks[d] = int(total_chunks)

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
        for i, d in enumerate(self.domains):
            if self._n_chunks[d] <= 0:
                vec[i] = 0.0
        s = float(vec.sum())
        if s <= 0.0:
            raise ValueError("domain weights sum to 0")
        self._p = vec / s

    def _sample_chunk(self, domain: str) -> np.ndarray:
        n = self._n_chunks[domain]
        if self.world_size > 1 and n >= self.world_size:
            n_local = (n - self.rank + self.world_size - 1) // self.world_size
            local = int(self._rng.integers(0, n_local))
            global_chunk = local * self.world_size + self.rank
        else:
            global_chunk = int(self._rng.integers(0, n))

        # Map global chunk index → (shard, offset). One shard ⇒ same as legacy mixlaw.
        remaining = global_chunk
        for mm, n_tok in zip(self._mmaps[domain], self._shard_tokens[domain]):
            shard_chunks = n_tok // self.seq_len
            if remaining < shard_chunks:
                start = remaining * self.seq_len
                return np.asarray(mm[start : start + self.seq_len], dtype=np.int64)
            remaining -= shard_chunks
        # Fallback (should be unreachable if _n_chunks is consistent).
        mm = self._mmaps[domain][0]
        return np.asarray(mm[0 : self.seq_len], dtype=np.int64)

    def next_input_ids(self, n_seqs: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Draw ``n_seqs`` sequences under current ``p``; return ``[n_seqs, seq_len]``."""
        n = int(n_seqs)
        if n <= 0:
            raise ValueError("n_seqs must be > 0")
        domain_ids = self._rng.choice(len(self.domains), size=n, p=self._p)
        rows = [self._sample_chunk(self.domains[int(i)]) for i in domain_ids]
        arr = np.stack(rows, axis=0)
        # Copy so the torch tensor owns memory (memmap views are not writable/owning).
        t = torch.from_numpy(arr.copy())
        if device is not None:
            t = t.to(device, non_blocking=True)
        return t


__all__ = ["DEFAULT_DOMAINS", "DomainMixtureStream", "MEMMAP_DTYPE", "SEQ_LEN"]
