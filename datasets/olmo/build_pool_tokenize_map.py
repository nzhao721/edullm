#!/usr/bin/env python3
"""Build input|output lines for per-shard dolma2 tokenization of an OLMo-mix pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_shard_name(index: int, rel: str) -> str:
    """Flat shard filename — avoids deep DCLM paths and file/dir collisions on scratch."""
    domain = rel.split("/")[1] if rel.startswith("data/") and "/" in rel[5:] else "misc"
    stem = Path(rel).name
    for suf in (".jsonl.zstd", ".jsonl.zst", ".json.gz", ".jsonl.gz", ".zstd", ".jsonl", ".json"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    h = hashlib.sha1(rel.encode()).hexdigest()[:10]
    safe_stem = stem.replace(" ", "_")[:60]
    return f"shards/{index:05d}__{domain}__{safe_stem}__{h}.npy"


def resolve_source(rel: str, data_root: Path, trim_root: Path | None) -> Path | None:
    candidates = [
        data_root / rel,
        data_root / "data" / rel,
    ]
    if rel.startswith("data/"):
        tail = rel[len("data/") :]
        candidates.append(data_root / tail)
        candidates.append(data_root / "data" / tail)
    if trim_root is not None:
        domain = rel.split("/")[1] if rel.startswith("data/") and "/" in rel[5:] else None
        if domain:
            trimmed = trim_root / domain / f"{domain}-trimmed.json.gz"
            if trimmed.is_file():
                candidates.append(trimmed)
            upsampled = trim_root / domain / f"{domain}-upsampled.json.gz"
            if upsampled.is_file():
                candidates.append(upsampled)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--trim-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--map-file", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "shards").mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Path, Path]] = []
    index_rows: list[dict] = []
    missing: list[str] = []

    for index, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        rel = rec.get("path")
        if not rel:
            continue
        src = resolve_source(rel, args.data_root, args.trim_root)
        if src is None:
            missing.append(rel)
            continue
        out_rel = stable_shard_name(index, rel)
        out = args.out_dir / out_rel
        rows.append((src, out))
        domain = rel.split("/")[1] if rel.startswith("data/") and "/" in rel[5:] else None
        index_rows.append(
            {
                "index": index,
                "manifest_path": rel,
                "domain": domain,
                "input": str(src),
                "npy": out_rel,
            }
        )

    seen: set[Path] = set()
    uniq: list[tuple[Path, Path]] = []
    for inp, out in rows:
        if out in seen:
            continue
        seen.add(out)
        uniq.append((inp, out))

    args.map_file.parent.mkdir(parents=True, exist_ok=True)
    with args.map_file.open("w", encoding="utf-8") as fh:
        for inp, out in uniq:
            fh.write(f"{inp}|{out}\n")

    index_path = args.manifest.parent / "tokenize_index.jsonl"
    with index_path.open("w", encoding="utf-8") as fh:
        for row in index_rows:
            fh.write(json.dumps(row) + "\n")

    meta = {
        "n_shards": len(uniq),
        "missing": missing,
        "missing_count": len(missing),
        "layout": "tokenized/shards/{index}__{domain}__{stem}__{hash}.npy",
    }
    args.map_file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta))
    if missing:
        raise SystemExit(f"missing {len(missing)} manifest shards under {args.data_root}")


if __name__ == "__main__":
    main()
