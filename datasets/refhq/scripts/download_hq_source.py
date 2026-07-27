#!/usr/bin/env python3
"""Download one HQ reference domain source onto FarmShare scratch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from hq_reference_sources import HQ_SOURCES


def _download_repo_prefix(
    repo_id: str,
    prefix: str,
    dest: Path,
    token: str | None,
    repo_type: str = "dataset",
    max_files: int | None = None,
    seed: int = 42,
) -> list[str]:
    import random

    from huggingface_hub import hf_hub_download, list_repo_files

    dest.mkdir(parents=True, exist_ok=True)
    files = [
        f
        for f in list_repo_files(repo_id, repo_type=repo_type, token=token)
        if f.startswith(prefix) and not f.endswith("/")
    ]
    if not files:
        raise SystemExit(f"no files under {repo_id}:{prefix}")
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_files is not None and max_files > 0:
        files = files[:max_files]
        print(
            f"datadecide_npy capped_to={len(files)} max_files={max_files} seed={seed}",
            flush=True,
        )
    local: list[str] = []
    for i, rel in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {rel}", flush=True)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            repo_type=repo_type,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
        local.append(path)
    return local


def _download_olmo_mix_domain(
    repo_id: str,
    path_prefix: str,
    dest: Path,
    token: str | None,
) -> list[str]:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"\.(json\.gz|jsonl\.gz|jsonl\.zstd|jsonl\.zst)$")
    files = [
        f
        for f in list_repo_files(repo_id, repo_type="dataset", token=token)
        if f.startswith(path_prefix) and pat.search(f)
    ]
    if not files:
        raise SystemExit(f"no olmo-mix files under {path_prefix}")
    local: list[str] = []
    for i, rel in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {rel}", flush=True)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
        local.append(path)
    return local


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.domain not in plan["domains"]:
        raise SystemExit(f"domain {args.domain} missing from plan")
    domain_plan = plan["domains"][args.domain]
    src = HQ_SOURCES[args.domain]
    dest = Path(domain_plan["paths"]["raw"])
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if src.get("gated") and not token:
        raise SystemExit(f"{args.domain} is gated; set HF_TOKEN on FarmShare")

    print(f"download domain={args.domain} kind={src['kind']} -> {dest}", flush=True)
    meta: dict = {"domain": args.domain, "source": src, "files": []}

    if src["kind"] == "datadecide_npy":
        overrides = domain_plan.get("source_overrides") or {}
        max_files = overrides.get("max_files")
        if max_files is None:
            max_files = int(src.get("max_files") or 0) or None
        else:
            max_files = int(max_files) or None
        files = _download_repo_prefix(
            src["repo_id"],
            src["prefix"],
            dest,
            token,
            max_files=max_files,
            seed=int(domain_plan.get("seed", plan.get("seed", 42))),
        )
        meta["files"] = files
        meta["max_files"] = max_files
    elif src["kind"] == "olmo_mix_domain":
        files = _download_olmo_mix_domain(src["repo_id"], src["path_prefix"], dest, token)
        meta["files"] = files
    elif src["kind"] == "hf_dataset":
        # Large HF corpora are streamed at build time (FarmShare CPU).
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        info = api.dataset_info(src["repo_id"])
        meta["snapshot"] = None
        meta["stream_from_hub"] = True
        meta["repo_id"] = src["repo_id"]
        meta["config"] = src.get("config")
        meta["split"] = src.get("split")
        meta["hub_id"] = getattr(info, "id", src["repo_id"])
        meta["files"] = []
        meta["file_count"] = 0
        print(
            f"stream-from-hub domain={args.domain} repo={src['repo_id']} "
            "(no full snapshot; build will stream)",
            flush=True,
        )
    else:
        raise SystemExit(f"unknown source kind {src['kind']}")

    marker = dest / "_download_complete.json"
    marker.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"download complete: {marker}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
