#!/usr/bin/env python3
"""Export the RefHQ 5.5B OLMo2-370M final checkpoint to a flat model.pt for RHO.

The training checkpoints under
  s3://edullm-olmo-370m-ckpts/edullm-370M-refhq-5p5b/checkpoints/step*/model_and_optim/
are olmo-core distributed (``.distcp``). RHO's FrozenReference loader requires a
local ``.pt`` / ``model.pt`` state dict — this script unshards step1315 (planned
end of the 5.5B budget; total_steps=1314) by default.

Does not start training. Safe to run on CPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REF_S3 = (
    "s3://edullm-olmo-370m-ckpts/edullm-370M-refhq-5p5b/checkpoints/step1315/"
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--s3-uri",
        default=DEFAULT_REF_S3,
        help="Checkpoint directory containing model_and_optim/ (default: step1315)",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Local working directory for download + unshard intermediates",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination model.pt path for reference.load_path",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing work-dir download",
    )
    args = ap.parse_args()

    work = args.work_dir
    ckpt_dir = work / "step_ckpt"
    unshard_dir = work / "unsharded"
    work.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _run(["aws", "s3", "sync", args.s3_uri.rstrip("/") + "/", str(ckpt_dir)])

    model_and_optim = ckpt_dir / "model_and_optim"
    if not (model_and_optim / ".metadata").exists():
        raise SystemExit(
            f"Missing distcp metadata under {model_and_optim}; "
            "expected an olmo-core model_and_optim/ checkpoint"
        )

    try:
        from olmo_core.distributed.checkpoint import unshard_checkpoint
    except ImportError as exc:
        raise SystemExit(
            "olmo_core is required to unshard. Install ai2-olmo-core or the pinned "
            f"edu-llm/OLMo-core checkout first.\nOriginal error: {exc}"
        ) from exc

    if unshard_dir.exists():
        shutil.rmtree(unshard_dir)
    unshard_dir.mkdir(parents=True, exist_ok=True)

    # olmo-core expects the model_and_optim/ distcp folder (where .metadata lives),
    # not the parent step*/ directory that also holds train/ + config.json.
    result = unshard_checkpoint(
        dir=str(model_and_optim),
        target_dir=str(unshard_dir),
        optim=False,
        save_overwrite=True,
    )
    print(json.dumps({"unshard_result": str(result)}, indent=2), flush=True)

    candidates = [
        unshard_dir / "model.pt",
        unshard_dir / "model.pth",
        unshard_dir / "model_and_optim" / "model.pt",
        ckpt_dir / "model.pt",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        # Fall back: any .pt under unshard_dir.
        pts = sorted(unshard_dir.rglob("*.pt"))
        if not pts:
            raise SystemExit(
                f"unshard_checkpoint finished but no model.pt found under {unshard_dir}"
            )
        src = pts[0]
        print(f"WARNING: using fallback unsharded file {src}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, args.output)
    meta = {
        "source_s3": args.s3_uri,
        "local_distcp": str(ckpt_dir),
        "unsharded_src": str(src),
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "note": (
            "RefHQ 5.5B CE final planned checkpoint (step1315; planned total_steps=1314). "
            "Use as token_selection reference.load_path for rho_excess."
        ),
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"READY reference.load_path={args.output}", flush=True)


if __name__ == "__main__":
    main()
