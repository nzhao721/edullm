"""The derived token manifest must match the corpus that actually sits in S3.

The corpus under ``data.tokens_s3`` ships one raw shard per domain plus a JSON sidecar,
and no manifest. These tests build a miniature of that exact layout -- including the
``paths.txt`` index written relative to the corpus root -- and pin both the translation
and the ways it must refuse to proceed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from token_selection.olmo_ext.token_io import write_token_array
from token_selection.scripts.build_token_manifest import build_manifest, write_manifest
from token_selection.scripts.experiment_contract import (
    manifest_train_paths,
    validate_token_budget,
    validate_token_manifest,
)

DOMAINS = ("algebraic-stack", "arxiv", "dclm", "open-web-math", "pes2o", "starcoder", "wiki")
TOKENIZER = "allenai/dolma2-tokenizer"
EOS_TOKEN_ID = 100257


def _write_domain(
    tokens_dir: Path,
    domain: str,
    *,
    n_docs: int,
    n_content: int,
    tokenizer: str = TOKENIZER,
    eos_token_id: int = EOS_TOKEN_ID,
) -> str:
    """Write one <domain>/<domain>.npy shard plus its sidecar; return the relative path."""
    domain_dir = tokens_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    n_tokens = n_content + n_docs  # every document is terminated by the EOS id
    tokens = np.arange(n_tokens, dtype=np.uint32)
    write_token_array(domain_dir / f"{domain}.npy", tokens)
    (domain_dir / f"{domain}.json").write_text(
        json.dumps(
            {
                "domain": domain,
                "output": f"/scratch/tokenized/{domain}/{domain}.npy",
                "tokenizer": tokenizer,
                "eos_token_id": eos_token_id,
                "docs": n_docs,
                "tokens_content": n_content,
                "tokens_with_eos": n_tokens,
                "bytes": n_tokens * 4,
                "dtype": "uint32",
            }
        ),
        encoding="utf-8",
    )
    return f"{domain}/{domain}.npy"


def _write_corpus(tokens_dir: Path, *, with_index: bool = True) -> None:
    tokens_dir.mkdir(parents=True, exist_ok=True)
    relatives = [
        _write_domain(tokens_dir, domain, n_docs=4 + i, n_content=500 + 64 * i)
        for i, domain in enumerate(DOMAINS)
    ]
    if with_index:
        # The real index is rooted at the corpus, i.e. every line carries a `tokenized/`
        # head, while the sync pulls the `tokenized/` prefix itself into tokens/.
        (tokens_dir / "paths.txt").write_text(
            "".join(f"tokenized/{rel}\n" for rel in relatives), encoding="utf-8"
        )


def test_manifest_is_derived_from_the_per_domain_sidecars(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)

    manifest = build_manifest(tokens, source_uri="s3://bucket/regmix-10b/tokenized")

    assert manifest["tokenizer"] == TOKENIZER
    assert manifest["eos_token_id"] == EOS_TOKEN_ID
    assert manifest["dtype"] == "uint32"
    assert manifest["source"] == "s3://bucket/regmix-10b/tokenized"
    assert [shard["path"] for shard in manifest["shards"]] == [
        f"{domain}/{domain}.npy" for domain in sorted(DOMAINS)
    ]
    # n_tokens is what the file holds (content + one EOS per document), because that is
    # what the trainer reads back out of the byte range.
    assert manifest["n_tokens"] == sum(shard["n_tokens"] for shard in manifest["shards"])
    assert manifest["n_tokens"] == manifest["n_content_tokens"] + manifest["n_docs"]


def test_derivation_is_byte_deterministic(tmp_path):
    """The order contract fingerprints this file, so rebuilding it must not perturb it."""
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)

    first = write_manifest(tokens, build_manifest(tokens)).read_bytes()
    second = write_manifest(tokens, build_manifest(tokens)).read_bytes()
    assert first == second


def test_derived_manifest_satisfies_the_validator_and_yields_every_shard(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    write_manifest(tokens, build_manifest(tokens))

    validated = validate_token_manifest(tokens, expected_tokenizer=TOKENIZER)
    assert len(validated["shards"]) == len(DOMAINS)

    paths = manifest_train_paths(tokens, expected_tokenizer=TOKENIZER)
    assert [Path(p).relative_to(tokens).as_posix() for p in paths] == [
        f"{domain}/{domain}.npy" for domain in sorted(DOMAINS)
    ]
    assert all(Path(p).is_file() for p in paths)


def test_paths_index_is_optional_but_authoritative_when_present(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens, with_index=False)
    assert len(build_manifest(tokens)["shards"]) == len(DOMAINS)

    # An index that omits a shard present on disk is a disagreement, not a filter: the
    # training set has to be unambiguous.
    (tokens / "paths.txt").write_text(
        "".join(f"tokenized/{d}/{d}.npy\n" for d in DOMAINS[:-1]), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="present but unlisted"):
        build_manifest(tokens)


def test_index_entry_with_no_matching_shard_is_refused(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    with (tokens / "paths.txt").open("a", encoding="utf-8") as handle:
        handle.write("tokenized/books/books.npy\n")
    with pytest.raises(SystemExit, match="not present under"):
        build_manifest(tokens)


def test_truncated_download_is_refused(tmp_path):
    """The sidecar's byte count is the only way to notice a partial sync."""
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    shard = tokens / "wiki" / "wiki.npy"
    shard.write_bytes(shard.read_bytes()[:-4])
    with pytest.raises(SystemExit, match="truncated"):
        build_manifest(tokens)


def test_np_save_shard_is_refused(tmp_path):
    """A 128-byte .npy header would be read as tokens and shift every boundary."""
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    np.save(tokens / "wiki" / "wiki.npy", np.arange(16, dtype=np.uint32))
    with pytest.raises(SystemExit, match="bytes"):
        build_manifest(tokens)


def test_missing_sidecar_is_refused(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    (tokens / "wiki" / "wiki.json").unlink()
    with pytest.raises(SystemExit, match="missing the sidecar"):
        build_manifest(tokens)


def test_mixed_tokenizers_across_domains_are_refused(tmp_path):
    """One vocabulary per dataset; mixed ids cannot share an embedding table."""
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    _write_domain(
        tokens, "wiki", n_docs=4, n_content=500, tokenizer="allenai/OLMo-2-0425-1B"
    )
    with pytest.raises(SystemExit, match="disagree on tokenizer"):
        build_manifest(tokens)


def test_tokenizer_mismatch_against_the_config_is_refused(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    write_manifest(tokens, build_manifest(tokens))
    with pytest.raises(ValueError, match="Tokenizer mismatch"):
        validate_token_manifest(tokens, expected_tokenizer="allenai/OLMo-2-0425-1B")


def test_stray_shard_in_a_domain_subdirectory_is_refused(tmp_path):
    """Shards are nested now, so the stray check cannot be a flat top-level glob."""
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    write_manifest(tokens, build_manifest(tokens))
    write_token_array(tokens / "wiki" / "leftover.npy", np.arange(8, dtype=np.uint32))
    with pytest.raises(ValueError, match="wiki/leftover.npy"):
        validate_token_manifest(tokens)


def test_shard_path_escaping_the_corpus_is_refused(tmp_path):
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    manifest = build_manifest(tokens)
    manifest["shards"][0]["path"] = "../outside.npy"
    write_manifest(tokens, manifest)
    with pytest.raises(ValueError, match=r"relative path inside"):
        validate_token_manifest(tokens)


def test_budget_refuses_wrapping_into_a_replaying_second_epoch(tmp_path):
    """Per-shard truncation makes the trainable corpus smaller than the token total."""
    tokens = tmp_path / "tokens"
    _write_corpus(tokens)
    manifest = build_manifest(tokens)
    seq_len = 64
    cfg = {"data": {"sequence_length": seq_len}, "train": {"max_tokens": 1}}

    budget = validate_token_budget(cfg, manifest)
    expected_sequences = sum(shard["n_tokens"] // seq_len for shard in manifest["shards"])
    assert budget["usable_sequences_per_epoch"] == expected_sequences
    assert budget["usable_tokens_per_epoch"] == expected_sequences * seq_len
    assert budget["usable_tokens_per_epoch"] < manifest["n_tokens"]
    assert budget["cycles_corpus"] is False

    cfg["train"]["max_tokens"] = budget["usable_tokens_per_epoch"] + 1
    with pytest.raises(ValueError, match="second epoch"):
        validate_token_budget(cfg, manifest)

