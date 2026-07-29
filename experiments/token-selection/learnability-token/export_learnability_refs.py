#!/usr/bin/env python3
"""Export early + late-averaged RefHQ .pt files for the learnability-token arm.

Early: RefHQ step250 (single checkpoint).
Late: mean of RefHQ step1000 / step1125 / step1315 parameter tensors.

Reuses ``reference/export_refhq_reference.py`` for DistCP download + unshard,
then averages the late exports with ``average_reference_state_dicts``.

Does not start training. Safe to run on CPU. Does not mutate AWS state beyond
whatever the caller already authorized for ``aws s3 sync`` (same as the RHO
export helper). Prefer running where the RefHQ DistCP shards are already local
(``--skip-download``) when possible.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

_TS_ROOT = Path(__file__).resolve().parents[1]
_EXPORT_SINGLE = _TS_ROOT / "reference" / "export_refhq_reference.py"
_DEFAULT_BASE = (
    "s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints"
)
_EARLY_STEP = 250
_LATE_STEPS = (1000, 1125, 1315)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _export_one(*, s3_uri: str, work_dir: Path, output: Path, skip_download: bool) -> None:
    cmd = [
        sys.executable,
        str(_EXPORT_SINGLE),
        "--s3-uri",
        s3_uri,
        "--work-dir",
        str(work_dir),
        "--output",
        str(output),
    ]
    if skip_download:
        cmd.append("--skip-download")
    _run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Scratch directory for DistCP download + unshard intermediates",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for early/late model.pt (+ sidecars)",
    )
    ap.add_argument(
        "--s3-base",
        default=_DEFAULT_BASE,
        help="Parent S3 prefix containing step*/ DistCP dirs",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse existing work-dir downloads for each step",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing early/late outputs",
    )
    args = ap.parse_args()

    if str(_TS_ROOT) not in sys.path:
        sys.path.insert(0, str(_TS_ROOT))
    from token_selection.olmo_ext.train_module import (
        average_reference_state_dicts,
        load_reference_state_dict,
    )

    work = args.work_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    base = str(args.s3_base).rstrip("/")

    early_pt = out / "refhq_step250_early.pt"
    late_pt = out / "refhq_late_avg_1000_1125_1315.pt"

    if early_pt.exists() and not args.force:
        print(json.dumps({"event": "skip_existing", "path": str(early_pt)}), flush=True)
    else:
        _export_one(
            s3_uri=f"{base}/step{_EARLY_STEP}/",
            work_dir=work / f"step{_EARLY_STEP}",
            output=early_pt,
            skip_download=args.skip_download,
        )

    late_singles: list[Path] = []
    for step in _LATE_STEPS:
        single = out / f"refhq_step{step}.pt"
        late_singles.append(single)
        if single.exists() and not args.force:
            print(json.dumps({"event": "skip_existing", "path": str(single)}), flush=True)
            continue
        _export_one(
            s3_uri=f"{base}/step{step}/",
            work_dir=work / f"step{step}",
            output=single,
            skip_download=args.skip_download,
        )

    if late_pt.exists() and not args.force:
        print(json.dumps({"event": "skip_existing", "path": str(late_pt)}), flush=True)
    else:
        states = [load_reference_state_dict(p) for p in late_singles]
        averaged = average_reference_state_dicts(states)
        tmp = Path(str(late_pt) + ".tmp")
        torch.save(
            {
                "model": averaged,
                "averaged_checkpoints": [str(p) for p in late_singles],
                "steps": list(_LATE_STEPS),
                "note": (
                    "Late learnability reference = mean of RefHQ steps "
                    f"{list(_LATE_STEPS)}. Use as reference.late.load_path."
                ),
            },
            tmp,
        )
        tmp.replace(late_pt)
        meta = {
            "output": str(late_pt),
            "steps": list(_LATE_STEPS),
            "sources": [str(p) for p in late_singles],
            "n_tensors": len(averaged),
            "bytes": late_pt.stat().st_size,
        }
        late_pt.with_suffix(".json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "late_avg_ready", **meta}, indent=2), flush=True)

    print(
        json.dumps(
            {
                "READY": True,
                "reference.early.load_path": str(early_pt),
                "reference.late.load_path": str(late_pt),
                "score": "L_early - L_late (larger = larger improvement; top-k keep)",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
