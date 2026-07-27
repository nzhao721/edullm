"""Token shards must be raw/headerless, exactly as OLMo-core's reader expects.

OLMo-core (pinned rev) reads a shard via ``load_array_slice``:
``np.frombuffer(get_bytes_range(path, start*itemsize, n*itemsize), dtype)`` and it
counts sequences from the file's byte size. It never parses a ``.npy`` header. These
tests pin our on-disk format to that contract and guard against a regression back to
``np.save`` (which would prepend a 128-byte header and corrupt both the tokens and the
sequence count).
"""

from __future__ import annotations

import numpy as np

from token_selection.olmo_ext.token_io import (
    TOKEN_DTYPE,
    count_tokens,
    read_token_array,
    write_token_array,
)


def _olmo_style_read(path, dtype, start_idx, end_idx):
    """Reproduce OLMo-core's raw byte-range read (no header handling)."""
    item = np.dtype(dtype).itemsize
    with open(path, "rb") as handle:
        handle.seek(start_idx * item)
        buffer = handle.read((end_idx - start_idx) * item)
    return np.frombuffer(buffer, dtype=dtype)


def test_round_trip_and_count(tmp_path):
    path = tmp_path / "tokens_0000.npy"
    tokens = np.arange(4096, dtype=TOKEN_DTYPE)
    n = write_token_array(path, tokens)
    assert n == tokens.size
    assert count_tokens(path) == tokens.size
    np.testing.assert_array_equal(read_token_array(path), tokens)


def test_raw_read_matches_olmo_byte_range(tmp_path):
    path = tmp_path / "tokens_0000.npy"
    tokens = (np.arange(100, 356) % 50000).astype(TOKEN_DTYPE)
    write_token_array(path, tokens)
    # Whole-array raw read equals the tokens (no header offset).
    np.testing.assert_array_equal(_olmo_style_read(path, TOKEN_DTYPE, 0, tokens.size), tokens)
    # An interior slice lines up exactly, i.e. offset math is byte-for-byte.
    np.testing.assert_array_equal(_olmo_style_read(path, TOKEN_DTYPE, 10, 20), tokens[10:20])


def test_np_save_would_corrupt_the_olmo_read(tmp_path):
    """Regression guard: the buggy np.save format is detectably wrong for OLMo's reader."""
    tokens = np.arange(64, dtype=TOKEN_DTYPE)
    good = tmp_path / "good.npy"
    bad = tmp_path / "bad.npy"
    write_token_array(good, tokens)
    np.save(bad, tokens)  # headered; NOT what OLMo-core reads

    # Raw read of the headerless file recovers the tokens; the np.save file does not,
    # and its byte size implies extra phantom tokens (the 128-byte header).
    np.testing.assert_array_equal(_olmo_style_read(good, TOKEN_DTYPE, 0, tokens.size), tokens)
    bad_first = _olmo_style_read(bad, TOKEN_DTYPE, 0, 1)
    assert int(bad_first[0]) != int(tokens[0])
    assert count_tokens(bad) != tokens.size
