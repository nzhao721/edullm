#!/usr/bin/env python3
"""Write broker-resolved temporary AWS credentials to a private shell file."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="mint a fresh STS session via sts:AssumeRole (prefers env creds, else profile)",
    )
    return parser.parse_args()


def export_line(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}\n"


def _session_from_env_or_profile(profile: str, region: str):
    import boto3

    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SESSION_TOKEN"):
        return boto3.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=region,
        )
    return boto3.Session(profile_name=profile, region_name=region)


def resolve_via_profile(profile: str, region: str) -> tuple[str, str, str, str | None]:
    session = _session_from_env_or_profile(profile, region)
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("no credentials were resolved")
    frozen = credentials.get_frozen_credentials()
    if not frozen.token:
        raise RuntimeError("credentials are not temporary session credentials")
    expiry = None
    if getattr(credentials, "_expiry_time", None) is not None:
        expiry = str(credentials._expiry_time)
    return frozen.access_key, frozen.secret_key, frozen.token, expiry


def _resolve_via_profile_only(profile: str, region: str) -> tuple[str, str, str, str | None]:
    """Resolve via AWS_PROFILE/credential_process only (ignore env session keys)."""
    import boto3

    saved = {
        k: os.environ.pop(k, None)
        for k in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SECURITY_TOKEN",
        )
    }
    try:
        session = boto3.Session(profile_name=profile, region_name=region)
        credentials = session.get_credentials()
        if credentials is None:
            raise RuntimeError("no credentials were resolved via profile")
        frozen = credentials.get_frozen_credentials()
        if not frozen.token:
            raise RuntimeError("profile credentials are not temporary session credentials")
        expiry = None
        if getattr(credentials, "_expiry_time", None) is not None:
            expiry = str(credentials._expiry_time)
        return frozen.access_key, frozen.secret_key, frozen.token, expiry
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def resolve_force_new(profile: str, region: str) -> tuple[str, str, str, str | None]:
    """Mint a brand-new STS session (AssumeRole, else profile/credential_process)."""
    import botocore.exceptions as _boto_exc

    # Prefer reminting through the broker profile so we get a new ~1h window.
    try:
        access_key, secret_key, session_token, expiration = _resolve_via_profile_only(
            profile, region
        )
        print(
            f"force-new via profile key=...{access_key[-4:]} expiry={expiration}",
            file=sys.stderr,
        )
        return access_key, secret_key, session_token, expiration
    except Exception as profile_exc:
        print(
            f"profile remint failed ({profile_exc}); trying AssumeRole chain",
            file=sys.stderr,
        )

    session = _session_from_env_or_profile(profile, region)
    sts = session.client("sts")
    ident = sts.get_caller_identity()
    arn = ident["Arn"]
    if ":assumed-role/" not in arn:
        return resolve_via_profile(profile, region)
    role_name = arn.split(":assumed-role/")[1].split("/")[0]
    account = ident["Account"]
    role_arn = f"arn:aws:iam::{account}:role/{role_name}"
    try:
        resp = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=f"farmshare-refresh-{int(time.time())}",
            DurationSeconds=3600,
        )
    except _boto_exc.ClientError as exc:
        print(
            f"assume-role failed ({exc}); falling back to existing credentials",
            file=sys.stderr,
        )
        return resolve_via_profile(profile, region)
    creds = resp["Credentials"]
    return (
        creds["AccessKeyId"],
        creds["SecretAccessKey"],
        creds["SessionToken"],
        creds["Expiration"].isoformat(),
    )


def main() -> int:
    args = parse_args()
    try:
        if args.force_new:
            access_key, secret_key, session_token, expiration = resolve_force_new(
                args.profile, args.region
            )
        else:
            access_key, secret_key, session_token, expiration = resolve_via_profile(
                args.profile, args.region
            )
    except Exception as exc:
        print(
            f"unable to resolve AWS credentials for profile {args.profile}: {exc}",
            file=sys.stderr,
        )
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
            if expiration:
                output.write(f"# Expiration={expiration}\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, args.output)
        os.chmod(args.output, 0o600)
        print(
            f"aws_session_written key=...{access_key[-4:]} expiry={expiration}",
            file=sys.stderr,
        )
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
