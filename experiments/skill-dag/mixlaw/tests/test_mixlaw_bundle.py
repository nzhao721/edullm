"""Deterministic local bundle parity tests (no tarball or cloud access)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_BUILDER_PATH = _REPO / "scripts" / "runpod" / "build_mixlaw_bundle.py"
_spec = importlib.util.spec_from_file_location("build_mixlaw_bundle", _BUILDER_PATH)
assert _spec is not None and _spec.loader is not None
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def test_bundle_sync_and_parity_detect_stale_and_extra_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    files = ["a/one.py", "b/two.sh"]
    for index, rel in enumerate(files):
        source = repo / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"source-{index}\n", encoding="utf-8")

    builder.sync_bundle(repo, bundle, files)
    assert builder.check_bundle(repo, bundle, files) == []

    (bundle / files[0]).write_text("stale\n", encoding="utf-8")
    (bundle / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    errors = builder.check_bundle(repo, bundle, files)
    assert f"stale bundle file: {files[0]}" in errors
    assert "unexpected bundle file: unexpected.txt" in errors


def test_checked_in_bundle_manifest_is_sorted_and_source_complete() -> None:
    manifest = _REPO / "scripts" / "runpod" / "mixlaw_bundle_manifest.json"
    files = builder.load_manifest(manifest)
    assert files == sorted(files)
    assert all((_REPO / rel).is_file() for rel in files)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
