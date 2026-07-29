#!/usr/bin/env python3
"""Upload RegMix heuristic + LM label trees to s3://edullm-datasets/regmix/regmix-10b/."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def provision_bucket(bucket: str, region: str) -> None:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket exists: s3://{bucket}", flush=True)
        return
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status not in {404, 403}:
            raise
    print(f"creating bucket s3://{bucket} in {region}", flush=True)
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )


def tree_stats(root: Path) -> dict:
    """Walk a label tree and return file count + total bytes (files only)."""
    n_files = 0
    n_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            n_files += 1
            n_bytes += path.stat().st_size
    return {"n_files": n_files, "bytes": n_bytes}


def read_ready(ready_path: Path) -> dict:
    raw = ready_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw.strip()}
    mtime = datetime.fromtimestamp(ready_path.stat().st_mtime, tz=timezone.utc)
    return {
        "path": str(ready_path),
        "mtime_utc": mtime.isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }


def resolve_lm_labels_root(run: Path) -> Path:
    """Prefer nested lm_labels/labels/ (FarmShare layout); else flat lm_labels/."""
    nested = run / "lm_labels" / "labels"
    flat = run / "lm_labels"
    if (nested / "READY").is_file():
        return nested
    if (flat / "READY").is_file():
        return flat
    # Prefer the canonical nested path in error messages when the tree exists.
    if nested.is_dir():
        raise SystemExit(f"missing READY for LM labels: {nested / 'READY'}")
    raise SystemExit(f"missing READY for LM labels: {flat / 'READY'}")


def sync_tree(local: Path, s3_uri: str) -> None:
    cmd = ["aws", "s3", "sync", str(local), s3_uri, "--only-show-errors"]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dst-bucket", default="edullm-datasets")
    parser.add_argument("--dst-prefix", default="regmix/regmix-10b")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip heuristic labels/ tree upload.",
    )
    parser.add_argument(
        "--skip-lm-labels",
        action="store_true",
        help="Skip lm_labels/ tree upload.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.skip_labels and args.skip_lm_labels:
        raise SystemExit("nothing to upload: both --skip-labels and --skip-lm-labels set")

    run = args.run_dir
    prefix = args.dst_prefix.strip("/")
    uploads: list[dict] = []

    if not args.skip_labels:
        labels_root = run / "labels"
        ready = labels_root / "READY"
        if not ready.is_file():
            raise SystemExit(f"missing READY for heuristic labels: {ready}")
        stats = tree_stats(labels_root)
        s3_uri = f"s3://{args.dst_bucket}/{prefix}/labels/"
        uploads.append(
            {
                "kind": "heuristic_labels",
                "local_root": str(labels_root),
                "s3_uri": s3_uri,
                "dst_key_prefix": f"{prefix}/labels",
                **stats,
                "ready": read_ready(ready),
            }
        )

    if not args.skip_lm_labels:
        lm_root = resolve_lm_labels_root(run)
        ready = lm_root / "READY"
        stats = tree_stats(lm_root)
        s3_uri = f"s3://{args.dst_bucket}/{prefix}/lm_labels/"
        uploads.append(
            {
                "kind": "lm_labels",
                "local_root": str(lm_root),
                "s3_uri": s3_uri,
                "dst_key_prefix": f"{prefix}/lm_labels",
                "local_layout": (
                    "lm_labels/labels"
                    if lm_root.name == "labels" and lm_root.parent.name == "lm_labels"
                    else "lm_labels"
                ),
                **stats,
                "ready": read_ready(ready),
            }
        )

    manifest = {
        "kind": "regmix-labels-upload",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(run),
        "destination": {
            "bucket": args.dst_bucket,
            "prefix": prefix,
            "region": args.region,
        },
        "trees": uploads,
        "bytes_total": sum(u["bytes"] for u in uploads),
        "n_files_total": sum(u["n_files"] for u in uploads),
        "dry_run": bool(args.dry_run),
    }

    out_path = run / "labels_upload_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)

    if args.dry_run:
        for row in uploads:
            print(f"dry-run would sync {row['local_root']} -> {row['s3_uri']}", flush=True)
        return 0

    provision_bucket(args.dst_bucket, args.region)
    for row in uploads:
        sync_tree(Path(row["local_root"]), row["s3_uri"])
        print(f"uploaded {row['kind']} to {row['s3_uri']}", flush=True)

    # Re-write manifest after successful sync (mark uploaded).
    manifest["uploaded_at"] = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    manifest["dry_run"] = False
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Also publish the manifest next to the corpus plan metadata.
    manifest_uri = f"s3://{args.dst_bucket}/{prefix}/labels_upload_manifest.json"
    cmd = ["aws", "s3", "cp", str(out_path), manifest_uri, "--only-show-errors"]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f"uploaded manifest to {manifest_uri}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
