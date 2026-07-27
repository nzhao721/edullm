#!/usr/bin/env python3
"""Write broker-resolved temporary AWS credentials to a private shell file."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    return parser.parse_args()


def export_line(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}\n"


def resolve_credentials(profile: str, region: str) -> tuple[str, str, str]:
    import boto3

    credentials = boto3.Session(
        profile_name=profile,
        region_name=region,
    ).get_credentials()
    if credentials is None:
        raise RuntimeError("no credentials were resolved")
    frozen = credentials.get_frozen_credentials()
    if not frozen.token:
        raise RuntimeError("credentials are not temporary session credentials")
    return frozen.access_key, frozen.secret_key, frozen.token


def main() -> int:
    args = parse_args()
    try:
        access_key, secret_key, session_token = resolve_credentials(args.profile, args.region)
    except Exception:
        print(f"unable to resolve AWS credentials for profile {args.profile}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write("# Generated on the FarmShare login node; do not commit or log.\n")
            output.write("unset AWS_PROFILE\n")
            output.write(export_line("AWS_ACCESS_KEY_ID", access_key))
            output.write(export_line("AWS_SECRET_ACCESS_KEY", secret_key))
            output.write(export_line("AWS_SESSION_TOKEN", session_token))
            output.write(export_line("AWS_REGION", args.region))
            output.write(export_line("AWS_DEFAULT_REGION", args.region))
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, args.output)
        os.chmod(args.output, 0o600)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
