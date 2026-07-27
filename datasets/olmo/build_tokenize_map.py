#!/usr/bin/env python3
"""Build tokenize map: input_shard|output_npy from local extract + trim outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    h = hashlib.sha1(rel.encode()).hexdigest()[:10]
    stem = path.name
    for suf in (".jsonl.zstd", ".jsonl.zst", ".json.gz", ".jsonl.gz", ".zstd"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    safe = rel.replace("/", "__").replace(" ", "_")
    if len(safe) > 180:
        safe = f"{stem}_{h}"
    return f"{safe}.npy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--trim-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--map-file", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Path, Path]] = []

    # Trimmed non-DCLM domains (final extract on scratch)
    for p in sorted(args.trim_root.glob("*/*-trimmed.json.gz")):
        out = args.out_dir / f"trim__{p.parent.name}.npy"
        rows.append((p, out))

    # DCLM shards from manifest (local copies under data-root)
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        domain = rec.get("domain") or rec.get("subset")
        if domain != "dclm":
            continue
        rel = rec.get("path") or rec.get("rel_path") or rec.get("local_path")
        if rel is None:
            continue
        rel_p = Path(rel)
        candidates = []
        if rel_p.is_absolute():
            candidates.append(rel_p)
        # manifest paths look like data/dclm/... ; local layout is often data-root/data/dclm/...
        candidates.append(args.data_root / rel_p)
        candidates.append(args.data_root / "data" / rel_p)
        if str(rel).startswith("data/"):
            candidates.append(args.data_root / str(rel)[len("data/") :])
            candidates.append(args.data_root / "data" / str(rel)[len("data/") :])
        src = next((c for c in candidates if c.is_file()), None)
        if src is None:
            base = Path(rel).name
            hits = list(args.data_root.rglob(base))
            if hits:
                src = hits[0]
        if src is None:
            print(f"WARN missing dclm shard: {rel}")
            continue
        out = args.out_dir / f"dclm__{Path(rel).name.replace('.jsonl.zstd', '').replace('.zstd','')}.npy"
        # avoid collisions on basename
        if any(o == out for _, o in rows):
            out = args.out_dir / stable_name(src, args.data_root)
        rows.append((src, out))

    # de-dupe by output
    seen = set()
    uniq = []
    for inp, out in rows:
        if out in seen:
            continue
        seen.add(out)
        uniq.append((inp, out))

    args.map_file.parent.mkdir(parents=True, exist_ok=True)
    with args.map_file.open("w", encoding="utf-8") as f:
        for inp, out in uniq:
            f.write(f"{inp}|{out}\n")

    meta = {
        "n_shards": len(uniq),
        "trim_shards": sum(1 for i, _ in uniq if "trim" in str(i)),
        "dclm_shards": sum(1 for i, _ in uniq if "dclm" in str(i).lower()),
    }
    args.map_file.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
