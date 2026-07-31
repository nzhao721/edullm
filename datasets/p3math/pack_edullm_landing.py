#!/usr/bin/env python3
"""Build a complete edullm-landing tree locally (payload + dataset.json + manifests).

Uses edullm-data 0.5.0 APIs (labels_from_path, split). Upload the resulting tree
via sb-aws; seal by uploading tokens/manifest.json last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edullm_data.contracts import SCHEMA_VERSION, canonical_json, validate_dataset_id, validate_purpose
from edullm_data.manifest import (
    Format,
    ManifestEntry,
    SPLITS,
    build_manifest,
    labels_from_path,
    manifest_sha256,
    parse_shard_name,
)
from edullm_data.publish import _format_for, _load_family, _prev_version, _resolve_path_partitions


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _link_or_copy(src: Path, dest: Path) -> None:
    """Prefer hardlink for large shards; fall back to copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def write_tree(
    *,
    out_root: Path,
    dataset_id: str,
    version: str,
    purpose: str,
    about: str,
    notes: str,
    license: dict[str, Any],
    sources: list[dict[str, Any]],
    family: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    groups_meta: list[dict[str, Any]],
    payload_copies: list[tuple[Path, str]],
    created_at: str,
    in_place: bool = False,
) -> Path:
    validate_dataset_id(dataset_id)
    validate_purpose(purpose)
    ds_root = out_root / dataset_id / version
    if in_place:
        ds_root.mkdir(parents=True, exist_ok=True)
    else:
        if ds_root.exists():
            shutil.rmtree(ds_root)
        ds_root.mkdir(parents=True)

    total_objects = sum(m["objects"] for m in manifests.values())
    total_bytes = sum(m["bytes"] for m in manifests.values())

    dataset_json: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "version": {"id": version, "relation": "supersedes", "of": _prev_version(version)},
        "created_at": created_at,
        "owner": family.get("owner", "edullm-data@alphaaiengineering.com"),
        "purpose": purpose,
        "mutability": family.get("defaults", {}).get("mutability", "frozen"),
        "inventory": {"objects": total_objects, "bytes": total_bytes},
        "groups": groups_meta,
        "sources": sources,
        "build": {
            "executor": {
                "kind": "external",
                "host_class": "workstation",
                "code_sha256": None,
                "packages_lock_sha256": None,
            },
            "reproducibility": "logical",
        },
        "license": license,
        "about": about,
        "notes": notes,
    }

    if not in_place:
        for src, rel in payload_copies:
            dest = ds_root / rel
            _link_or_copy(src, dest)

    for g, man in manifests.items():
        man_path = ds_root / g / "manifest.json"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_bytes(canonical_json(man))

    (ds_root / "dataset.json").write_bytes(canonical_json(dataset_json))
    print(f"wrote {ds_root}")
    print(
        json.dumps(
            {"dataset_id": dataset_id, "version": version, "inventory": dataset_json["inventory"]},
            indent=2,
        )
    )
    return ds_root

def pack_tokenizer(args: argparse.Namespace) -> Path:
    family = _load_family("tokenizer")
    tok_dir = args.tokenizer_dir
    files = sorted(p for p in tok_dir.iterdir() if p.is_file())
    defaults = family.get("defaults", {})
    entries: list[ManifestEntry] = []
    copies: list[tuple[Path, str]] = []
    for path in files:
        rel = f"tokenizer/{path.name}"
        sha, size = _sha256_file(path)
        fmt = _format_for(rel, defaults)
        entries.append(
            ManifestEntry(path=rel, sha256=sha, bytes=size, count=None, format=fmt)
        )
        copies.append((path, rel))
    man = build_manifest(entries, group_name="tokenizer")
    gm = {
        "name": "tokenizer",
        "profile": "tokenizer/v1",
        "prefix": "tokenizer/",
        "manifest": "tokenizer/manifest.json",
        "manifest_sha256": manifest_sha256(man),
    }
    created = args.created_at or datetime.now(timezone.utc).isoformat()
    return write_tree(
        out_root=args.out_root,
        dataset_id="tokenizer/bytes-utf8",
        version=args.version,
        purpose=(
            "Published raw UTF-8 byte tokenizer (vocab 0-255) so byte-token corpora own "
            "the vocabulary their uint32 shards decode against"
        ),
        about=(
            "Identity UTF-8 byte tokenizer: each token id equals the corresponding byte "
            "value (0..255). No BPE merges."
        ),
        notes="vocab_size derived from tokenizer.json (expect 256). No EOS special.",
        license={"id": "Apache-2.0", "basis": "declared"},
        sources=[],
        family=family,
        manifests={"tokenizer": man},
        groups_meta=[gm],
        payload_copies=copies,
        created_at=created,
    )


def pack_pretrain(args: argparse.Namespace) -> Path:
    family = _load_family("pretrain")
    tokens_src = args.tokens_dir  # .../tokens or .../tokens/<source>
    group_payload = tokens_src.parent if tokens_src.name != "tokens" else tokens_src
    defaults = family.get("defaults", {})
    files = sorted(p for p in group_payload.rglob("*.u32le.bin") if p.is_file())
    if not files:
        raise SystemExit(f"no .u32le.bin under {group_payload}")

    entries: list[ManifestEntry] = []
    copies: list[tuple[Path, str]] = []
    for path in files:
        rel = f"tokens/{path.relative_to(group_payload).as_posix()}"
        sha, size = _sha256_file(path)
        fmt = _format_for(rel, defaults)
        parsed = parse_shard_name(rel)
        labels = labels_from_path(rel) or None
        entries.append(
            ManifestEntry(
                path=rel,
                sha256=sha,
                bytes=size,
                count={"unit": "tokens", "value": size // 4},
                format=fmt,
                split=parsed[0] if parsed and parsed[0] in SPLITS else None,
                labels=labels,
            )
        )
        copies.append((path, rel))

    man = build_manifest(entries, group_name="tokens")
    tok_dep = {
        "role": "tokenizer",
        "dataset_id": args.tokenizer_id,
        "version": args.tokenizer_version,
        "manifest_sha256": args.tokenizer_manifest_sha256,
    }
    partitions = _resolve_path_partitions(defaults.get("partitions", []), entries)
    gm = {
        "name": "tokens",
        "profile": "pretrain-tokens/v1",
        "prefix": "tokens/",
        "manifest": "tokens/manifest.json",
        "manifest_sha256": manifest_sha256(man),
        "depends_on": [tok_dep],
        "partitions": partitions,
        "coverage": "partition",
    }

    build_meta: dict[str, Any] = {}
    if args.build_meta and Path(args.build_meta).exists():
        build_meta = json.loads(Path(args.build_meta).read_text(encoding="utf-8"))
    train_tokens = sum(
        int(e.count["value"])
        for e in entries
        if e.count and e.path.rsplit("/", 1)[-1].startswith("train-")
    )
    val_tokens = sum(
        int(e.count["value"])
        for e in entries
        if e.count and e.path.rsplit("/", 1)[-1].startswith("val-")
    )

    dataset_id = args.dataset_id
    purpose = args.purpose
    about = args.about
    notes = args.notes
    sources = args.sources_json
    if sources:
        sources_list = json.loads(Path(sources).read_text(encoding="utf-8"))
    else:
        # Default: Lean-only package (legacy path).
        sources_list = [
            {
                "name": "phanerozoic/Lean4-Mathlib",
                "tokens": train_tokens,
                "documents": build_meta.get("docs"),
                "license": "Apache-2.0",
                "uri": "https://huggingface.co/datasets/phanerozoic/Lean4-Mathlib",
                "scope": "measured-in-this-dataset",
            }
        ]
        if dataset_id == "pretrain/lean4-mathlib-bytes" and not purpose:
            purpose = (
                "Lean 4 Mathlib declarations as raw UTF-8 byte tokens for small byte-LM / "
                "MambaByte-style continual pretraining and formal-math probes"
            )
            about = (
                "phanerozoic/Lean4-Mathlib rendered as Lean-like declaration documents, packed "
                "as uint32 little-endian shards under tokens/mathlib/ where each token id "
                "equals the corresponding UTF-8 byte. Tokenizer: tokenizer/bytes-utf8. "
                "Content only — no Gate A alphabet markers."
            )
            notes = (
                f"Measured train_tokens={train_tokens:,} val_tokens={val_tokens:,}. "
                "Docs joined with blank lines. Raw bytes only (no BPE)."
            )

    if not purpose or not about:
        raise SystemExit("--purpose and --about are required for non-legacy packs")
    if not notes:
        notes = (
            f"Measured train_tokens={train_tokens:,} val_tokens={val_tokens:,}. "
            "Val is Lean-only (0.15% of Lean bytes). Raw UTF-8 bytes as uint32; no alphabet markers."
        )

    created = args.created_at or datetime.now(timezone.utc).isoformat()
    return write_tree(
        out_root=args.out_root,
        dataset_id=dataset_id,
        version=args.version,
        purpose=purpose,
        about=about,
        notes=notes,
        license={"id": "ODC-By-1.0", "basis": "declared"},
        sources=sources_list,
        family=family,
        manifests={"tokens": man},
        groups_meta=[gm],
        payload_copies=copies,
        created_at=created,
        in_place=bool(args.in_place),
    )

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokenizer")
    t.add_argument("--tokenizer-dir", type=Path, required=True)
    t.add_argument("--out-root", type=Path, required=True)
    t.add_argument("--version", default="v1")
    t.add_argument("--created-at", default=None)

    c = sub.add_parser("pretrain")
    c.add_argument("--tokens-dir", type=Path, required=True, help=".../tokens or .../tokens/<source>")
    c.add_argument("--out-root", type=Path, required=True)
    c.add_argument("--version", default="v1")
    c.add_argument("--dataset-id", default="pretrain/lean4-mathlib-bytes")
    c.add_argument("--purpose", default=None)
    c.add_argument("--about", default=None)
    c.add_argument("--notes", default=None)
    c.add_argument("--sources-json", type=Path, default=None, help="JSON list for dataset.json sources[]")
    c.add_argument(
        "--in-place",
        action="store_true",
        help="Tokens already live under out-root/<id>/<ver>/tokens; only write manifests+dataset.json",
    )
    c.add_argument("--tokenizer-id", default="tokenizer/bytes-utf8")
    c.add_argument("--tokenizer-version", default="v1")
    c.add_argument("--tokenizer-manifest-sha256", required=True)
    c.add_argument("--build-meta", type=Path, default=None)
    c.add_argument("--created-at", default=None)

    args = p.parse_args()
    if args.cmd == "tokenizer":
        pack_tokenizer(args)
    else:
        pack_pretrain(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
