#!/usr/bin/env python3
"""Upload/download experiment artifacts to/from S3.

Layout:
  INPUT (read-only, published edullm-data):
    tokens/ <- data.dataset_id via edullm_data.read + ensure_train_tokens
              (``.u32le.bin`` train shards; local train manifest.json written here)
  LOCAL (produced on the train box):
    order/               <- ensure_order_contract / freeze_order after tokens exist
  OUTPUTS (this run) — published under edullm-checkpoints/token-sel/<arm>/:
    checkpoints/         -> s3://edullm-checkpoints/token-sel/<arm>/checkpoints/
    task_loss_results/   -> s3://edullm-checkpoints/token-sel/<arm>/task_loss_results/
    metrics/             -> s3://edullm-checkpoints/token-sel/<arm>/metrics/

``s3.prefix`` in each arm YAML must be ``token-sel/<arm>`` (see
``token_selection.olmo_ext.s3_layout``).

Required object tags (apply via bucket policy / sync tooling as org requires):
  Project, Environment, ManagedBy, Owner.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.scripts import (
    load_config,
    resolve_output_dir,
    resolve_tokens_s3,
    s3_uri,
)
from token_selection.scripts.edullm_data_tokens import (
    ensure_order_contract,
    ensure_train_tokens,
)


def _aws(profile: str, args: list[str]) -> None:
    # Empty / "none" profile → instance-role / env credentials (AWS B200 hosts).
    if profile and str(profile).strip().lower() not in {"none", "null", "-"}:
        cmd = ["aws", "--profile", profile, *args]
    else:
        cmd = ["aws", *args]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def sync_dir(
    local: Path,
    remote: str,
    *,
    profile: str,
    upload: bool,
    mirror: bool = False,
    keep: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
) -> None:
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    remote = remote if remote.endswith("/") else remote + "/"
    if upload:
        if not any(local.iterdir()):
            if mirror:
                raise SystemExit(
                    f"Refusing mirrored (--delete) upload of empty {local} to {remote}."
                )
            print(f"WARNING: {local} is empty; nothing to upload", file=sys.stderr)
            return
    cmd = ["s3", "sync", str(local), remote] if upload else ["s3", "sync", remote, str(local)]
    # For download of inputs we may want --delete; for upload of outputs we do not.
    if mirror:
        cmd.append("--delete")
    # Keep locally-derived files (e.g. manifest.json) from being deleted by --delete.
    for pattern in keep:
        cmd.extend(["--exclude", pattern])
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    cmd.extend(["--only-show-errors"])
    _aws(profile, cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "rho-1/configs/run_rho_10b.yaml")
    ap.add_argument("--direction", choices=["upload", "download"], required=True)
    ap.add_argument(
        "--what",
        choices=["tokens", "metrics", "checkpoints", "task_loss", "all"],
        default="all",
    )
    ap.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable --delete when downloading tokens (NOT recommended: risks stale shards)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    profile = str(cfg["s3"].get("profile", "sbsandbox"))
    upload = args.direction == "upload"

    if shutil.which("aws") is None:
        raise SystemExit("aws CLI not found on PATH")

    arm_root = s3_uri(cfg, bucket_key="checkpoint_bucket").rstrip("/")

    if args.what in ("tokens", "all") and not upload:
        # Preferred path: resolve + stage via edullm_data.read (excludes val, writes
        # local train manifest). Works on a clean machine.
        try:
            remote = resolve_tokens_s3(cfg)
        except Exception as exc:
            raise SystemExit(f"train corpus resolution failed: {exc}") from exc
        print(f"=== download tokens from {remote} via ensure_train_tokens")
        try:
            ensure_train_tokens(cfg, out / "tokens", profile=profile, force=True)
            ensure_order_contract(cfg, out)
        except Exception as exc:
            raise SystemExit(f"token staging failed: {exc}") from exc
        print(
            "\nTokens staged from edullm-data and order contract written.\n"
            "Train with: python -m token_selection.scripts.train_olmo_template "
            f"--config {args.config} --method <method> --olmo-root <OLMo-core> --launch"
        )

    # (name, local, remote, read_only_input, keep_through_mirror)
    targets: list[tuple[str, Path, str, bool, tuple[str, ...]]] = []
    if args.what in ("metrics", "all"):
        targets.append(
            ("metrics", out / "metrics", f"{arm_root}/metrics/", False, ())
        )
    if args.what in ("checkpoints", "all"):
        targets.append(
            ("checkpoints", out / "checkpoints", f"{arm_root}/checkpoints/", False, ())
        )
    if args.what in ("task_loss", "all"):
        tl_local = out / "task_loss_results"
        targets.append(
            ("task_loss_results", tl_local, f"{arm_root}/task_loss_results/", False, ())
        )

    for name, local, remote, read_only, keep in targets:
        if upload and read_only:
            raise SystemExit(
                f"Refusing to upload {name!r} to {remote}. Pre-tokenized inputs are "
                "read-only; only metrics, checkpoints, and task_loss_results are published."
            )
        if not upload and not read_only and args.what == "all":
            print(f"=== skip download {name} (run output; use --what {name} to pull)")
            continue
        if upload and name == "tokens":
            raise SystemExit(
                "Refusing to upload tokens. Pre-tokenized inputs are read-only under "
                "s3://edullm-data/; only metrics, checkpoints, and task_loss_results "
                "are published."
            )
        mirror = read_only and not upload and not args.no_mirror
        print(f"=== {args.direction} {name}: {local} <-> {remote}{' [mirror]' if mirror else ''}")
        sync_dir(local, remote, profile=profile, upload=upload, mirror=mirror, keep=keep)

    if args.what in ("tokens", "all") and upload:
        raise SystemExit(
            "Refusing to upload tokens. Pre-tokenized inputs are read-only under "
            "s3://edullm-data/; only metrics, checkpoints, and task_loss_results "
            "are published."
        )
    print("Done.")


if __name__ == "__main__":
    main()
