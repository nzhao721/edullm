"""Fail-closed contracts shared by the token-selection experiment arms.

The public OLMo-core ``NumpyFSLDataLoader`` does not accept an externally supplied
instance permutation. Its deterministic order is instead defined by the dataset
fingerprint and loader settings. This module records and verifies that contract
rather than claiming that an unused permutation file controls production order.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from token_selection.olmo_ext.token_io import TOKEN_DTYPE, count_tokens, dtype_from_name, dtype_name

TOKENS_PLACEHOLDER = "REPLACE_ME"

TOKEN_MANIFEST_SCHEMA = """\
tokens/manifest.json is DERIVED from the corpus in S3 by
`python -m token_selection.scripts.build_token_manifest`. It is not hand-written and
it does not live in the bucket, because the corpus ships per-domain sidecars instead
of a single manifest. It must be a JSON object shaped like:

  {
    "n_tokens": 10004807041,                   # total tokens across every shard
    "dtype": "uint32",                         # optional; defaults to uint32
    "tokenizer": "allenai/dolma2-tokenizer",   # optional; cross-checked vs data.tokenizer
    "eos_token_id": 100257,                    # optional; informational
    "shards": [
      {"path": "algebraic-stack/algebraic-stack.npy", "n_tokens": 615239017},
      {"path": "arxiv/arxiv.npy",                     "n_tokens": 2500162905}
    ]
  }

Shard paths are relative to tokens/ and MAY contain sub-directories: the corpus
stores one shard per domain at tokenized/<domain>/<domain>.npy, so there is no
flat tokens_NNNN.npy naming to rely on.

Shard files are RAW HEADERLESS arrays (numpy ``ndarray.tofile`` / ``np.memmap``),
not ``np.save`` output, despite the .npy suffix. OLMo-core's NumpyFSLDataset reads
them with np.frombuffer over a byte range and derives the sequence count from the
file's byte size, so a 128-byte .npy header would both corrupt the tokens and
miscount sequences. The .npy suffix is OLMo-core's naming convention for these raw
files, and the corpus follows it.\
"""


def _relative_shard_path(name: str, *, manifest_path: Path) -> str:
    """Normalize a manifest shard path, refusing anything that escapes tokens/."""
    raw = str(name).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not raw or raw.endswith("/"):
        raise ValueError(
            f"Shard path {name!r} in {manifest_path} must be a relative path inside "
            f"tokens/ with no '..' components.\n\n{TOKEN_MANIFEST_SCHEMA}"
        )
    return path.as_posix()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_path(cfg: Mapping[str, Any], path_value: str | Path, *, root: Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def validate_scratch_config(
    cfg: Mapping[str, Any],
    *,
    method: Optional[str] = None,
) -> None:
    """Reject any configuration that would accidentally become continued pretraining.

    When ``method='rho_excess'`` (or it is the only configured method), also require a
    non-null ``reference.load_path``. Smoke configs may omit the path and supply an
    in-memory frozen twin instead.
    """
    model = cfg.get("model") or {}
    if model.get("init_mode") != "scratch":
        raise ValueError("model.init_mode must be 'scratch' for this token-selection experiment")
    if model.get("load_path"):
        raise ValueError("model.load_path must be null in scratch mode; checkpoint loading is CPT")

    run_seed = int(cfg.get("seed", 0))
    if int(model.get("init_seed", run_seed)) != run_seed:
        raise ValueError("model.init_seed must match the shared experiment seed")
    train = cfg.get("train") or {}
    if int(train.get("data_loader_seed", run_seed)) != run_seed:
        raise ValueError("train.data_loader_seed must match the shared experiment seed")

    resolved = method
    if resolved is None:
        methods = cfg.get("methods") or []
        if len(methods) == 1:
            resolved = str(methods[0])
    if resolved == "rho_excess":
        ref = cfg.get("reference") or {}
        load_path = ref.get("load_path")
        if load_path is None or str(load_path).strip() in ("", "null", "None", "REPLACE_ME"):
            raise ValueError(
                "rho_excess requires reference.load_path (local path to a frozen reference "
                "checkpoint). Sync the checkpoint locally first if it lives on S3."
            )


def validate_token_manifest(
    tokens_dir: Path,
    *,
    expected_tokenizer: Optional[str] = None,
) -> Dict[str, Any]:
    """Check the pre-tokenized input against ``TOKEN_MANIFEST_SCHEMA``; return the manifest.

    The manifest is the training set: OLMo-core's ``NumpyFSLDataset`` packs whatever
    ``paths`` it is handed and derives each shard's sequence count from the file's byte
    size, so a directory glob would silently absorb a stray shard and a mis-formatted
    shard would be read as tokens without complaint. This verifies, in order, that the
    manifest declares ``n_tokens`` and a non-empty ``shards`` list, that the listed files
    and the ``.npy`` files on disk are the same set, that every shard's byte size is a
    whole number of tokens, and that the observed token totals match what the manifest
    claims. The byte-size check is what catches a shard written with ``np.save``: its
    128-byte header shifts the token count by ``header_bytes // itemsize``.

    ``expected_tokenizer`` is compared against the tokenizer the manifest inherited from
    the corpus sidecars. That pairing is load-bearing: the tokenizer decides
    ``padded_vocab_size()`` and hence the embedding matrix, so training a vocab built for
    one tokenizer on ids emitted by another is an out-of-range index, not a bad score.
    """
    tokens_dir = Path(tokens_dir)
    manifest_path = tokens_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing token manifest: {manifest_path}\n\n{TOKEN_MANIFEST_SCHEMA}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Token manifest must be a JSON object: {manifest_path}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(
            f"Token manifest lists no shards: {manifest_path}\n\n{TOKEN_MANIFEST_SCHEMA}"
        )
    try:
        listed = sorted(
            _relative_shard_path(shard["path"], manifest_path=manifest_path) for shard in shards
        )
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Every entry of shards[] needs a 'path': {manifest_path}\n\n{TOKEN_MANIFEST_SCHEMA}"
        ) from exc
    if len(set(listed)) != len(listed):
        duplicates = sorted({name for name in listed if listed.count(name) > 1})
        raise ValueError(f"Token manifest lists the same shard twice: {duplicates}")

    missing = [name for name in listed if not (tokens_dir / name).exists()]
    if missing:
        raise ValueError(f"Manifest lists shards absent from {tokens_dir}: {missing}")
    # rglob, not a flat tokens_*.npy glob: the corpus nests one shard per domain under
    # tokens/<domain>/<domain>.npy, and a stray shard in any sub-directory is still a
    # stray shard the dataset must not silently absorb.
    on_disk = {p.relative_to(tokens_dir).as_posix() for p in tokens_dir.rglob("*.npy")}
    extras = sorted(on_disk - set(listed))
    if extras:
        raise ValueError(
            f"Unlisted token shard(s) present in {tokens_dir}: {extras}. The training set "
            "must equal the manifest; remove stray shards or re-run build_token_manifest."
        )

    if expected_tokenizer is not None:
        manifest_tokenizer = manifest.get("tokenizer")
        if manifest_tokenizer and str(manifest_tokenizer) != str(expected_tokenizer):
            raise ValueError(
                f"Tokenizer mismatch: the corpus in {tokens_dir} was tokenized with "
                f"{manifest_tokenizer!r} but data.tokenizer is {expected_tokenizer!r}. The "
                "tokenizer sets the model's vocabulary size, so this would train the wrong "
                "embedding table on out-of-range ids. Fix data.tokenizer to match the corpus."
            )

    dtype = dtype_from_name(manifest["dtype"]) if manifest.get("dtype") else TOKEN_DTYPE
    declared_total = manifest.get("n_tokens")
    if not isinstance(declared_total, int) or isinstance(declared_total, bool) or declared_total <= 0:
        raise ValueError(
            f"Token manifest needs a positive integer 'n_tokens': {manifest_path}\n\n"
            f"{TOKEN_MANIFEST_SCHEMA}"
        )

    observed_total = 0
    for shard in shards:
        name = _relative_shard_path(shard["path"], manifest_path=manifest_path)
        try:
            observed = count_tokens(tokens_dir / name, dtype=dtype)
        except ValueError as exc:
            raise ValueError(
                f"{name}: not a raw headerless {dtype_name(dtype)} array. {exc}\n\n"
                f"{TOKEN_MANIFEST_SCHEMA}"
            ) from exc
        declared = shard.get("n_tokens")
        if declared is not None and int(declared) != observed:
            raise ValueError(
                f"{name}: manifest claims {int(declared)} tokens but the file holds "
                f"{observed} as {dtype_name(dtype)}. Either the shard is truncated or it "
                f"was written in the wrong format.\n\n{TOKEN_MANIFEST_SCHEMA}"
            )
        observed_total += observed

    if observed_total != declared_total:
        raise ValueError(
            f"Token manifest claims n_tokens={declared_total} but {tokens_dir} holds "
            f"{observed_total} tokens as {dtype_name(dtype)}. A shard written with np.save "
            f"instead of a raw array is the usual cause.\n\n{TOKEN_MANIFEST_SCHEMA}"
        )
    return manifest


def manifest_train_paths(
    tokens_dir: Path,
    *,
    expected_tokenizer: Optional[str] = None,
) -> List[str]:
    """Ordered training shard paths taken *from the manifest*, never a bare glob.

    Sorted by relative path rather than kept in manifest order, so the sequence the
    dataset packs depends only on *which* shards were selected and not on how the
    manifest happened to enumerate them.
    """
    tokens_dir = Path(tokens_dir)
    manifest_path = tokens_dir / "manifest.json"
    manifest = validate_token_manifest(tokens_dir, expected_tokenizer=expected_tokenizer)
    listed = sorted(
        _relative_shard_path(shard["path"], manifest_path=manifest_path)
        for shard in manifest["shards"]
    )
    return [str(tokens_dir / name) for name in listed]


def contract_sequence_length(cfg: Mapping[str, Any]) -> int:
    """Sequence length the order contract and dataset agree on."""
    data_len = (cfg.get("data") or {}).get("sequence_length")
    seq_len = int(data_len or 0)
    if seq_len <= 0:
        raise ValueError("data.sequence_length must be a positive integer")
    return seq_len


def validate_token_budget(
    cfg: Mapping[str, Any],
    token_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Refuse a token budget that would wrap past one epoch; return the budget report.

    ``NumpyFSLDataset`` truncates *each* shard to ``n_tokens // sequence_length``
    sequences, so the trainable corpus is strictly smaller than the manifest's token
    total and the shortfall grows with the shard count. If ``max_tokens`` exceeds it, the
    loader silently rolls into a second epoch and starts replaying data, which is a
    different experiment from the single-pass run this config describes. Checking it here
    turns that into a launch-time error instead of an artifact nobody notices in the loss
    curve.

    """
    seq_len = contract_sequence_length(cfg)
    shards = token_manifest.get("shards") or []
    usable_sequences = sum(int(shard["n_tokens"]) // seq_len for shard in shards)
    usable_tokens = usable_sequences * seq_len
    if usable_tokens <= 0:
        raise ValueError(
            f"No shard holds a full sequence of {seq_len} tokens; nothing is trainable."
        )
    max_tokens = int((cfg.get("train") or {}).get("max_tokens", 0))
    if max_tokens <= 0:
        raise ValueError("train.max_tokens must be a positive integer")

    cycles_corpus = False
    if max_tokens > usable_tokens and not cycles_corpus:
        raise ValueError(
            f"train.max_tokens={max_tokens} exceeds the {usable_tokens} tokens this corpus "
            f"can serve in one epoch ({usable_sequences} sequences of {seq_len} across "
            f"{len(shards)} shards; each shard is truncated to a whole number of "
            "sequences). The loader would wrap into a second epoch and replay data. "
            "Lower train.max_tokens or add shards."
        )
    return {
        "sequence_length": seq_len,
        "manifest_n_tokens": int(token_manifest["n_tokens"]),
        "usable_sequences_per_epoch": usable_sequences,
        "usable_tokens_per_epoch": usable_tokens,
        "max_tokens": max_tokens,
        "epochs_consumed": round(max_tokens / usable_tokens, 6),
        "cycles_corpus": cycles_corpus,
    }


def build_order_contract(cfg: Mapping[str, Any], *, output_dir: Path, token_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe the public OLMo-core deterministic loader order."""
    train = cfg.get("train") or {}
    token_manifest_path = Path(output_dir) / "tokens" / "manifest.json"
    if not token_manifest_path.exists():
        raise ValueError(f"Missing token manifest: {token_manifest_path}")

    global_batch_size = int(train.get("global_batch_size", 0))
    chunk_size = int(train.get("data_loader_chunk_size", 1))
    initial_epoch = int(train.get("data_loader_initial_epoch", 0))
    if global_batch_size <= 0:
        raise ValueError("train.global_batch_size must be positive in the order contract")
    if chunk_size <= 0 or initial_epoch < 0:
        raise ValueError("data-loader chunk size must be positive and initial epoch non-negative")
    contract: Dict[str, Any] = {
        "schema_version": 1,
        "order_kind": "olmo_core_seeded_global_indices",
        "olmo_data_loader": "NumpyFSLDataLoader",
        "token_manifest_sha256": sha256_file(token_manifest_path),
        "token_manifest_n_tokens": int(token_manifest["n_tokens"]),
        "sequence_length": contract_sequence_length(cfg),
        "global_batch_size": global_batch_size,
        "data_loader_seed": int(train.get("data_loader_seed", cfg.get("seed", 0))),
        "source_permutation_seed": int(cfg.get("seed", 0)),
        "shuffle": True,
        "initial_epoch": initial_epoch,
        "chunk_size": chunk_size,
    }
    contract["contract_sha256"] = hashlib.sha256(_canonical_json(contract)).hexdigest()
    return contract


def validate_order_contract(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    contract: Mapping[str, Any],
) -> None:
    """Ensure the persisted order contract matches the currently requested run."""
    token_manifest_path = Path(output_dir) / "tokens" / "manifest.json"
    if not token_manifest_path.exists():
        raise ValueError(f"Missing token manifest: {token_manifest_path}")
    token_manifest = json.loads(token_manifest_path.read_text(encoding="utf-8"))
    expected = build_order_contract(cfg, output_dir=output_dir, token_manifest=token_manifest)
    actual = dict(contract)
    actual.pop("created_at", None)
    if actual != expected:
        raise ValueError(
            "Order contract mismatch. Both arms must use identical token manifests, "
            "loader seed, sequence length, and global batch size."
        )


def verify_olmo_revision(olmo_root: Path, required_revision: str) -> None:
    """Verify a local editable OLMo-core checkout is on the pinned revision."""
    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(olmo_root),
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to inspect OLMo-core revision at {olmo_root}") from exc
    if actual != required_revision:
        raise RuntimeError(
            f"OLMo-core revision mismatch: expected {required_revision}, found {actual}"
        )
