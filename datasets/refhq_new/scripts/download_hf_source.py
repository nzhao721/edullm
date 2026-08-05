#!/usr/bin/env python3
"""Download one refhq-new HF source onto FarmShare scratch (array task)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from refhq_new.exclusion import load_exclusion_rules, skip_smoltalk_config  # noqa: E402


def _hf_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _download_snapshot(repo_id: str, dest: Path, token: str | None) -> list[str]:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    # Prefer high-throughput transfer when hf_transfer is installed.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    root = Path(local_dir)
    files = sorted(
        str(p.relative_to(root)).replace("\\", "/")
        for p in root.rglob("*")
        if p.is_file() and p.name != "_download_complete.json"
    )
    return files


def _list_smoltalk_configs(repo_id: str, token: str | None) -> list[str]:
    from datasets import get_dataset_config_names

    kwargs: dict = {"path": repo_id}
    if token:
        kwargs["token"] = token
    try:
        names = get_dataset_config_names(**kwargs)
    except TypeError:
        kwargs.pop("token", None)
        names = get_dataset_config_names(**kwargs)
    return sorted(str(n) for n in names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.source not in plan["sources"]:
        raise SystemExit(f"source {args.source} missing from plan")
    src_plan = plan["sources"][args.source]
    dest = Path(src_plan["paths"]["raw"])
    repo_id = src_plan["hf_repo"]
    token = _hf_token(args.token)

    if src_plan.get("gated") and not token:
        raise SystemExit(f"{args.source} is gated; set HF_TOKEN / {args.plan.parent.parent}/.hf_token")

    print(f"download source={args.source} repo={repo_id} -> {dest}", flush=True)
    files = _download_snapshot(repo_id, dest, token)

    meta: dict = {
        "source": args.source,
        "repo_id": repo_id,
        "raw_dir": str(dest),
        "file_count": len(files),
        "files_sample": files[:50],
        "multi_config": bool(src_plan.get("multi_config")),
    }

    if args.source == "smoltalk":
        rules = load_exclusion_rules()
        all_configs = _list_smoltalk_configs(repo_id, token)
        keep = [c for c in all_configs if not skip_smoltalk_config(c, rules)]
        skip = [c for c in all_configs if skip_smoltalk_config(c, rules)]
        meta["smoltalk_configs_all"] = all_configs
        meta["smoltalk_configs_keep"] = keep
        meta["smoltalk_configs_skip"] = skip
        print(f"smoltalk keep={keep} skip={skip}", flush=True)

    marker = dest / "_download_complete.json"
    marker.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"download complete: {marker} files={len(files)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
