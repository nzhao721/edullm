#!/usr/bin/env python3
"""Deterministically sync or verify the local MixLaw runtime bundle.

This tool never creates or uploads a tarball. Packaging and cloud mutation are
separate, explicitly authorized operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

GENERATED_MANIFEST = "BUNDLE_MANIFEST.sha256"


def load_manifest(path: Path) -> list[str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError(f"invalid bundle manifest: {path}")
    files = [str(item).replace("\\", "/").strip("/") for item in payload["files"]]
    if not files or files != sorted(set(files)):
        raise ValueError("bundle file list must be non-empty, unique, and sorted")
    if any(part == ".." for rel in files for part in Path(rel).parts):
        raise ValueError("bundle paths may not escape the repository")
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rendered_hash_manifest(repo_root: Path, files: Iterable[str]) -> str:
    return "".join(f"{sha256(repo_root / rel)}  {rel}\n" for rel in files)


def check_bundle(
    repo_root: Path,
    bundle_root: Path,
    files: list[str],
) -> list[str]:
    errors: list[str] = []
    expected = set(files)
    for rel in files:
        source = repo_root / rel
        bundled = bundle_root / rel
        if not source.is_file():
            errors.append(f"missing source: {rel}")
        elif not bundled.is_file():
            errors.append(f"missing bundle file: {rel}")
        elif sha256(source) != sha256(bundled):
            errors.append(f"stale bundle file: {rel}")

    actual = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name != GENERATED_MANIFEST
    }
    for rel in sorted(actual.difference(expected)):
        errors.append(f"unexpected bundle file: {rel}")

    generated = bundle_root / GENERATED_MANIFEST
    expected_hashes = (
        rendered_hash_manifest(repo_root, files)
        if all((repo_root / rel).is_file() for rel in files)
        else ""
    )
    if not generated.is_file():
        errors.append(f"missing generated manifest: {GENERATED_MANIFEST}")
    elif generated.read_text(encoding="utf-8") != expected_hashes:
        errors.append(f"stale generated manifest: {GENERATED_MANIFEST}")
    return errors


def sync_bundle(repo_root: Path, bundle_root: Path, files: list[str]) -> None:
    for rel in files:
        source = repo_root / rel
        if not source.is_file():
            raise FileNotFoundError(source)
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    for rel in files:
        target = bundle_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / rel, target)
    (bundle_root / GENERATED_MANIFEST).write_text(
        rendered_hash_manifest(repo_root, files),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "mixlaw_bundle_manifest.json",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=script_dir / "_mixlaw_bundle",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Replace the local bundle tree from the manifest before checking",
    )
    args = parser.parse_args()

    files = load_manifest(args.manifest)
    if args.sync:
        sync_bundle(repo_root, args.bundle_root, files)
    errors = check_bundle(repo_root, args.bundle_root, files)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"MixLaw bundle parity ok: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
