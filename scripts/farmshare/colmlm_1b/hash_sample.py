"""Shared MD5-mod sampling predicate for Co-LMLM 1B corpus.

Cross-tool-consistent with DuckDB:
  int(md5(doc_id)[:8], 16) % modulus == residue
"""

from __future__ import annotations

import hashlib


def doc_in_sample(doc_id: str, *, modulus: int = 100, residue: int = 0) -> bool:
    digest = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus == residue


def duckdb_predicate(column: str = "id", *, modulus: int = 100, residue: int = 0) -> str:
    # DuckDB md5() returns hex string; substr first 8 chars → bigint → mod.
    return (
        f"(CAST(('0x' || substr(md5({column}), 1, 8)) AS UBIGINT) % {int(modulus)}) "
        f"= {int(residue)}"
    )
