#!/usr/bin/env python3
"""Download Co-LMLM fineweb_with_fullwiki_entries.db from the HF storage bucket."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path


SRC = "hf://buckets/lil-lab/co-lmlm-360m-fw-fineweb-wiki-index/fineweb_with_fullwiki_entries.db"
EXPECTED_MIN_BYTES = 700_000_000_000  # ~700 GB floor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk-mb", type=int, default=64)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set; refusing anonymous 712GB download")
    print(f"auth token present len={len(token)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file() and args.out.stat().st_size >= EXPECTED_MIN_BYTES:
        print(f"already present size={args.out.stat().st_size}", flush=True)
        _sanity(args.out)
        return 0

    fs = HfFileSystem(token=token)
    info = fs.info(SRC)
    expected = int(info.get("size") or 0)
    print(f"downloading {SRC} size={expected} -> {args.out}", flush=True)

    partial = args.out.with_suffix(args.out.suffix + ".partial")
    # Resume if partial exists.
    start = partial.stat().st_size if partial.is_file() else 0
    mode = "ab" if start else "wb"
    chunk = args.chunk_mb * 1024 * 1024
    t0 = time.time()
    written = start
    with fs.open(SRC, "rb") as inf, open(partial, mode) as out:
        if start:
            print(f"resuming from {start}", flush=True)
            inf.seek(start)
        last_log = t0
        while True:
            buf = inf.read(chunk)
            if not buf:
                break
            out.write(buf)
            written += len(buf)
            now = time.time()
            if now - last_log >= 30:
                mbps = (written - start) / max(now - t0, 1) / 1e6
                pct = (100.0 * written / expected) if expected else 0.0
                print(
                    f"progress {written}/{expected} ({pct:.2f}%) ~{mbps:.1f} MB/s",
                    flush=True,
                )
                last_log = now
                out.flush()
    if expected and written != expected:
        raise SystemExit(f"size mismatch written={written} expected={expected}")
    partial.replace(args.out)
    print(f"done size={args.out.stat().st_size}", flush=True)
    _sanity(args.out)
    return 0


def _sanity(path: Path) -> None:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    rows = list(cur.execute("SELECT entry_id, substr(data,1,160) FROM entries LIMIT 3"))
    for entry_id, data in rows:
        print(f"sanity {entry_id} {data[:100]}", flush=True)
        if not (entry_id.startswith("<urn:uuid:") or "_fact" in entry_id):
            print("WARN unexpected entry_id shape", flush=True)
    n = cur.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    print(f"entries count={n}", flush=True)
    con.close()


if __name__ == "__main__":
    # Ensure token from env if present.
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        for p in (
            Path.home() / ".cache/huggingface/token",
            Path(os.environ.get("HF_HOME", "")) / "token",
        ):
            if p.is_file():
                os.environ["HF_TOKEN"] = p.read_text(encoding="utf-8").strip()
                os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
                break
    sys.exit(main())
