#!/usr/bin/env python3
"""Download FineWeb-Edu sample-100BT parquet shards with retries and optional sharding."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


HF_DATASET = "HuggingFaceFW/fineweb-edu"
HF_PREFIX = "sample/100BT"


def _ensure_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    for p in (
        Path.home() / ".cache/huggingface/token",
        Path(os.environ.get("HF_HOME", "")) / "token",
    ):
        if p.is_file():
            tok = p.read_text(encoding="utf-8").strip()
            if tok:
                os.environ["HF_TOKEN"] = tok
                os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
                return


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def list_parquet_files() -> list[dict]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=_token())
    root = f"hf://datasets/{HF_DATASET}/{HF_PREFIX}"
    entries = fs.ls(root, detail=True)
    files = []
    for e in entries:
        name = e.get("name") or e.get("path") or ""
        # name like datasets/HuggingFaceFW/fineweb-edu/sample/100BT/000_00000.parquet
        if not str(name).endswith(".parquet"):
            continue
        rel = str(name).split(f"{HF_DATASET}/", 1)[-1]
        if not rel.startswith(HF_PREFIX):
            # fallback: take last two components
            parts = str(name).rstrip("/").split("/")
            rel = f"{HF_PREFIX}/{parts[-1]}"
        files.append({"path": rel, "size": int(e.get("size") or 0)})
    files.sort(key=lambda x: x["path"])
    if not files:
        raise RuntimeError(f"no parquet files under {root}")
    return files


def download_one(
    rel_path: str,
    dest: Path,
    *,
    expected_size: int,
    retries: int,
    backoff: float,
) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected_size > 0 and dest.stat().st_size == expected_size:
        print(f"skip exists {dest.name} ({expected_size})", flush=True)
        return

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            local = hf_hub_download(
                repo_id=HF_DATASET,
                filename=rel_path,
                repo_type="dataset",
                local_dir=str(dest.parent / "_hf_cache"),
                token=_token(),
            )
            local_path = Path(local)
            got = local_path.stat().st_size
            if expected_size > 0 and got != expected_size:
                raise RuntimeError(f"size mismatch {got} != {expected_size} for {rel_path}")
            # Move/copy into flat out dir for DuckDB glob.
            if dest.exists():
                dest.unlink()
            local_path.replace(dest)
            print(f"ok {dest.name} ({got})", flush=True)
            return
        except Exception as e:  # noqa: BLE001 — retry transient HF/network errors
            last_err = e
            sleep = backoff * (2**attempt)
            print(f"retry {rel_path}: {e}; sleep {sleep:.0f}s", flush=True)
            time.sleep(sleep)
    raise RuntimeError(f"failed {rel_path}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--backoff", type=float, default=20.0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    _ensure_token()
    files = list_parquet_files()
    print(f"found {len(files)} parquet files", flush=True)
    if args.list_only:
        for f in files:
            print(f"{f['size']}\t{f['path']}")
        return 0

    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard_index]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(mine)} files -> {args.out}",
        flush=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for f in mine:
        dest = args.out / Path(f["path"]).name
        download_one(
            f["path"],
            dest,
            expected_size=f["size"],
            retries=args.retries,
            backoff=args.backoff,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
