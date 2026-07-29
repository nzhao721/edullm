#!/usr/bin/env python3
"""Unshard a local olmo-core distcp checkpoint to a flat model.pt state dict."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Local step*/ directory containing model_and_optim/.metadata",
    )
    ap.add_argument("--output", type=Path, required=True, help="Destination model.pt")
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temp unshard dir (default: sibling unshard_tmp under checkpoint parent)",
    )
    ap.add_argument("--step", type=int, default=None, help="Step label for metadata JSON")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    model_and_optim = ckpt_dir / "model_and_optim"
    if not (model_and_optim / ".metadata").exists():
        raise SystemExit(f"Missing distcp metadata under {model_and_optim}")

    try:
        from olmo_core.distributed.checkpoint import unshard_checkpoint
    except ImportError as exc:
        raise SystemExit(f"olmo_core unshard_checkpoint required: {exc}") from exc

    work = args.work_dir or (ckpt_dir.parent / f"_unshard_tmp_{ckpt_dir.name}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    unshard_checkpoint(
        dir=str(model_and_optim),
        target_dir=str(work),
        optim=False,
        save_overwrite=True,
    )

    candidates = [
        work / "model.pt",
        work / "model.pth",
        work / "model_and_optim" / "model.pt",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        pts = sorted(work.rglob("*.pt"))
        if not pts:
            raise SystemExit(f"unshard finished but no .pt under {work}")
        src = pts[0]
        print(f"WARNING: using fallback {src}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.output)

    step = args.step
    if step is None:
        name = ckpt_dir.name.replace("step", "").split("-")[0]
        try:
            step = int(name)
        except ValueError:
            step = None

    meta = {
        "checkpoint_dir": str(ckpt_dir),
        "unsharded_src": str(src),
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "step": step,
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"READY {args.output}", flush=True)

    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
