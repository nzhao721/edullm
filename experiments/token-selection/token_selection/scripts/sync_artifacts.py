#!/usr/bin/env python3
"""Stage published training inputs from S3 into runtime scratch.

This command intentionally has no upload mode and no checkpoint, progress,
metrics, or eval targets. Run artifacts remain on scratch and are uploaded by
the trainers to W&B.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.scripts import load_config, resolve_output_dir, resolve_tokens_s3
from token_selection.scripts.edullm_data_tokens import (
    ensure_order_contract,
    ensure_train_tokens,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=ROOT / "rho-1/configs/run_rho_10b.yaml"
    )
    ap.add_argument("--direction", choices=["download"], default="download")
    ap.add_argument("--what", choices=["tokens"], default="tokens")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Restage published input shards into runtime scratch",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    profile = str((cfg.get("s3") or {}).get("profile", "sbsandbox"))
    remote = resolve_tokens_s3(cfg)
    print(f"=== stage training inputs from {remote}")
    ensure_train_tokens(cfg, out / "tokens", profile=profile, force=bool(args.force))
    ensure_order_contract(cfg, out)
    print(f"Inputs staged under {out}; outputs must use local scratch + W&B.")


if __name__ == "__main__":
    main()
