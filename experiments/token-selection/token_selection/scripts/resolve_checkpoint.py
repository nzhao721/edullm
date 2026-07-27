#!/usr/bin/env python3
"""Resolve / write ``model.load_path`` for the CPT experiment.

Searches ``s3://edullm-checkpoints/`` for an early OLMo-2-1B candidate.
Fails closed if none found (do not invent a path). Use ``--set PATH`` to
write a known URI into the config.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.scripts import load_config

PREFERRED_PATTERNS = [
    re.compile(r"olmo[-_]?2.*1b", re.I),
    re.compile(r"OLMo-2-0425-1B", re.I),
    re.compile(r"early", re.I),
    re.compile(r"step[-_]?0*[1-9]\d{0,4}(?!\d)", re.I),  # early-ish step ids
]


def _aws_ls(profile: str, uri: str) -> List[str]:
    cmd = ["aws", "--profile", profile, "s3", "ls", uri, "--recursive"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output, file=sys.stderr)
        return []
    paths: List[str] = []
    bucket = uri[len("s3://") :].split("/", 1)[0]
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # DATE TIME SIZE KEY
        parts = line.split()
        if len(parts) < 4:
            continue
        key = parts[-1]
        paths.append(f"s3://{bucket}/{key}")
    return paths


def rank_candidates(paths: List[str]) -> List[str]:
    scored: List[tuple[int, str]] = []
    for p in paths:
        score = 0
        for i, pat in enumerate(PREFERRED_PATTERNS):
            if pat.search(p):
                score += len(PREFERRED_PATTERNS) - i
        # Prefer model.pt / config style checkpoint dirs
        if p.endswith(".pt") or p.endswith(".pth") or "model.safetensors" in p or p.endswith("/"):
            score += 1
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, p in scored]


def write_load_path(config_path: Path, load_path: str) -> None:
    cfg: Dict[str, Any] = load_config(config_path)
    cfg.setdefault("model", {})["load_path"] = load_path
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote model.load_path={load_path!r} -> {config_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_rho_10b.yaml")
    ap.add_argument("--set", dest="set_path", default=None, help="Explicit checkpoint URI to write")
    ap.add_argument("--dry-run", action="store_true", help="List candidates; do not write config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    # This tool is for continued pretraining (CPT). The REL-vs-full experiment is scratch,
    # where a non-null model.load_path is rejected downstream anyway -- so refuse up front
    # rather than let it poison the shared config with a checkpoint path.
    if (cfg.get("model") or {}).get("init_mode") == "scratch":
        raise SystemExit(
            f"{args.config} is a scratch (from-scratch) config; resolve_checkpoint.py only "
            "applies to CPT runs. Refusing to write model.load_path into a scratch config."
        )
    profile = str(cfg.get("s3", {}).get("profile", "sbsandbox"))
    bucket = str(cfg.get("s3", {}).get("checkpoint_bucket", "edullm-checkpoints"))

    if args.set_path:
        if args.dry_run:
            print(json.dumps({"would_set": args.set_path}, indent=2))
            return
        write_load_path(args.config, args.set_path)
        return

    uri = f"s3://{bucket}/"
    print(f"Searching {uri} (profile={profile}) ...")
    all_paths = _aws_ls(profile, uri)
    candidates = rank_candidates(all_paths)
    payload = {
        "bucket": uri,
        "n_objects": len(all_paths),
        "candidates": candidates[:20],
        "current_load_path": (cfg.get("model") or {}).get("load_path"),
    }
    print(json.dumps(payload, indent=2))

    if not candidates:
        raise SystemExit(
            "No early OLMo-2-1B checkpoint found under edullm-checkpoints. "
            "Upload one or pass --set s3://.../path (fail closed)."
        )

    chosen = candidates[0]
    # Prefer a directory prefix if we hit a file inside a ckpt folder
    if chosen.endswith(".pt") or chosen.endswith(".pth") or chosen.endswith(".safetensors"):
        chosen = chosen.rsplit("/", 1)[0] + "/"

    if args.dry_run:
        print(json.dumps({"would_set": chosen}, indent=2))
        return
    write_load_path(args.config, chosen)


if __name__ == "__main__":
    main()
