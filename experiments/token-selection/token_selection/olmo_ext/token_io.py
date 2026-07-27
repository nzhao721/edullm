"""Raw, headerless token I/O compatible with OLMo-core's ``NumpyFSLDataset``.

At the pinned revision OLMo-core reads token shards as *raw* arrays: it takes a
byte range ``start_idx * itemsize`` and interprets it with ``np.frombuffer``
(``olmo_core.data.utils.load_array_slice``), and it derives the number of
training sequences from the file's *byte size* (``_get_file_size_and_length``),
never from a manifest and never by parsing a NumPy ``.npy`` header.

``np.save`` writes a 128-byte header before the data, so a shard written with
``np.save`` would make the trainer (a) read the header bytes as tokens and
(b) miscount sequences by ``header_bytes // itemsize``. To keep the whole
pipeline (tokenize -> freeze order -> train) on a single on-disk format,
these helpers write and read the same headerless layout that OLMo-core's own
``write_array_to_disk`` / ``np.memmap`` produce.
"""

from __future__ import annotations

import hashlib
import os
from typing import Type, Union

import numpy as np

# OLMo-2's padded vocabulary exceeds 2**16, so tokens require uint32.
TOKEN_DTYPE: Type[np.generic] = np.uint32
MASK_DTYPE: Type[np.generic] = np.bool_

PathLike = Union[str, "os.PathLike[str]"]

_DTYPE_BY_NAME = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}


def dtype_from_name(name: str) -> Type[np.generic]:
    key = str(name).lower()
    if key not in _DTYPE_BY_NAME:
        raise ValueError(f"Unsupported token dtype {name!r}; expected one of {sorted(_DTYPE_BY_NAME)}")
    return _DTYPE_BY_NAME[key]


def dtype_name(dtype: Union[str, Type[np.generic], np.dtype]) -> str:
    return np.dtype(dtype).name


def write_token_array(path: PathLike, arr: np.ndarray, *, dtype: Union[str, Type[np.generic]] = TOKEN_DTYPE) -> int:
    """Write ``arr`` as a raw headerless little-endian-native array; return token count.

    Uses ``ndarray.tofile`` which emits exactly the bytes ``np.memmap`` expects,
    i.e. no ``.npy`` header. This is the format OLMo-core's reader assumes.
    """
    array = np.ascontiguousarray(arr, dtype=np.dtype(dtype))
    array.tofile(os.fspath(path))
    return int(array.size)


def read_token_array(path: PathLike, *, dtype: Union[str, Type[np.generic]] = TOKEN_DTYPE) -> np.ndarray:
    """Read a raw headerless token array written by :func:`write_token_array`."""
    return np.fromfile(os.fspath(path), dtype=np.dtype(dtype))


def count_tokens(path: PathLike, *, dtype: Union[str, Type[np.generic]] = TOKEN_DTYPE) -> int:
    """Number of tokens in a raw shard, exactly as OLMo-core computes it (bytes // itemsize)."""
    item_size = np.dtype(dtype).itemsize
    size = os.path.getsize(os.fspath(path))
    if item_size <= 0 or size % item_size:
        raise ValueError(
            f"{path}: byte size {size} is not a whole multiple of itemsize {item_size} for dtype {dtype}"
        )
    return size // item_size


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
