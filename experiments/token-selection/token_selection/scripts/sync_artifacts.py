#!/usr/bin/env python3
"""Upload/download experiment artifacts to/from S3.

Layout:
  INPUT (read-only, pre-tokenized elsewhere):
    tokens/ <- data.tokens_s3   (download only; never upload)
              per-domain <domain>/<domain>.npy + .json sidecars + paths.txt
  LOCAL (produced on the train box, not synced from the token bucket):
    tokens/manifest.json <- build_token_manifest, derived from the sidecars
    order/               <- freeze_order after the manifest exists
  OUTPUTS (this run):
    metrics/     -> s3://<dataset_bucket>/<prefix>/metrics/
    checkpoints/ -> s3://<checkpoint_bucket>/<prefix>/

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
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    # ``--delete`` mirrors destination to source. Used when downloading tokens so a
    # stale local shard cannot linger. Metrics/checkpoints stay additive.
    extra = ["--delete"] if mirror else []
    # Excluded destination files survive ``--delete``. The derived token manifest is
    # local-only, so without this a re-sync of the corpus would silently remove it and
    # invalidate the frozen order contract that fingerprints it.
    for pattern in keep if mirror else ():
        extra += ["--exclude", pattern]
    if upload:
        if not any(local.iterdir()):
            if mirror:
                raise SystemExit(
                    f"Refusing mirrored (--delete) upload of empty {local} to {remote}."
                )
            print(f"WARNING: {local} is empty; nothing to upload", file=sys.stderr)
        _aws(profile, ["s3", "sync", str(local), remote, *extra])
    else:
        _aws(profile, ["s3", "sync", remote, str(local), *extra])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_rho_10b.yaml")
    ap.add_argument("--direction", choices=["upload", "download"], required=True)
    ap.add_argument(
        "--what",
        choices=["tokens", "metrics", "checkpoints", "all"],
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

    # (name, local, remote, read_only_input, keep_through_mirror)
    targets: list[tuple[str, Path, str, bool, tuple[str, ...]]] = []
    if args.what in ("tokens", "all"):
        targets.append(
            ("tokens", out / "tokens", resolve_tokens_s3(cfg) + "/", True, ("manifest.json",))
        )
    if args.what in ("metrics", "all"):
        targets.append(("metrics", out / "metrics", s3_uri(cfg, "metrics"), False, ()))
    if args.what in ("checkpoints", "all"):
        remote = s3_uri(cfg, bucket_key="checkpoint_bucket")
        targets.append(("checkpoints", out / "checkpoints", remote.rstrip("/") + "/", False, ()))

    for name, local, remote, read_only, keep in targets:
        if upload and read_only:
            raise SystemExit(
                f"Refusing to upload {name!r} to {remote}. Pre-tokenized inputs are "
                "read-only; only metrics and checkpoints are published from a run."
            )
        if not upload and not read_only and args.what == "all":
            # ``--what all --direction download`` should not pull run outputs by default;
            # only tokens are an input. Skip metrics/checkpoints on download-all.
            print(f"=== skip download {name} (run output; use --what {name} to pull)")
            continue
        mirror = read_only and not upload and not args.no_mirror
        print(f"=== {args.direction} {name}: {local} <-> {remote}{' [mirror]' if mirror else ''}")
        sync_dir(local, remote, profile=profile, upload=upload, mirror=mirror, keep=keep)

    if args.what in ("tokens", "all") and not upload:
        print(
            "\nNext: derive the token manifest from the corpus sidecars, then freeze order:\n"
            "  python -m token_selection.scripts.build_token_manifest\n"
            "  python token_selection/scripts/freeze_order.py"
        )
    print("Done.")


if __name__ == "__main__":
    main()
