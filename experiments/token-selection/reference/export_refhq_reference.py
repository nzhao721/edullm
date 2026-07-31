#!/usr/bin/env python3
"""Export a RefHQ OLMo2-370M DistCP checkpoint to a flat model.pt.

Thin CLI over ``token_selection.olmo_ext.refhq_materialize.export_distcp_to_pt``.

Default source is the durable reference-arm layout written by
``train_olmo3_370m_refhq.py`` / ``launch_train.sh``::

  s3://edullm-checkpoints/token-sel/reference/checkpoints/refhq-regmix-5p5b-v1/step1313/

Prefer letting ``--launch`` auto-materialize from YAML ``reference.s3_uri``; use
this script for a one-shot offline export.

Does not start training. Safe to run on CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TS_ROOT = Path(__file__).resolve().parents[1]
if str(_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TS_ROOT))

from token_selection.olmo_ext.refhq_materialize import (  # noqa: E402
    DEFAULT_REFERENCE_ARM_FINAL,
    export_distcp_to_pt,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--s3-uri",
        default=DEFAULT_REFERENCE_ARM_FINAL,
        help=(
            "Checkpoint directory containing model_and_optim/ "
            f"(default: {DEFAULT_REFERENCE_ARM_FINAL})"
        ),
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
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output .pt",
    )
    args = ap.parse_args()
    out = export_distcp_to_pt(
        args.s3_uri,
        work_dir=args.work_dir,
        output=args.output,
        skip_download=args.skip_download,
        force=args.force,
    )
    print(json.dumps({"READY": True, "reference.load_path": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
