#!/usr/bin/env python3
"""Append top-up shards to olmohq S3 manifests without modifying regmix-10b.

Uploads new raw + tokenized objects, then rewrites plan manifests to include
both old and new shards. Existing objects are left in place (append-only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def list_s3_sizes(bucket: str, prefix: str) -> dict[str, int]:
    """Map relative key (under prefix) → ContentLength via list-objects-v2."""
    out: dict[str, int] = {}
    token: str | None = None
    pfx = prefix.strip("/") + "/"
    while True:
        cmd = [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            pfx,
            "--output",
            "json",
        ]
        if token:
            cmd.extend(["--continuation-token", token])
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(proc.stdout or "{}")
        for obj in payload.get("Contents") or []:
            key = obj["Key"]
            if key.startswith(pfx):
                rel = key[len(pfx) :]
            else:
                rel = key
            out[rel] = int(obj["Size"])
        if not payload.get("IsTruncated"):
            break
        token = payload.get("NextContinuationToken")
        if not token:
            break
    return out


def cp_if_needed(
    local: Path,
    uri: str,
    *,
    remote_sizes: dict[str, int],
    root: str,
    dry_run: bool,
) -> None:
    """Upload only when remote is missing or size differs (resume-safe)."""
    local_size = local.stat().st_size
    assert uri.startswith(root + "/")
    rel = uri[len(root) + 1 :]
    remote = remote_sizes.get(rel)
    if remote == local_size:
        print(f"skip (exists size={local_size}): {uri}", flush=True)
        return
    if dry_run:
        print(f"dry-run would upload {local} → {uri}", flush=True)
        return
    run(["aws", "s3", "cp", str(local), uri, "--only-show-errors"])
    remote_sizes[rel] = local_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--bucket", default="edullm-datasets")
    ap.add_argument("--prefix", default="olmo100b/olmo-mix-1124-30b")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir
    prefix = args.prefix.strip("/")
    root = f"s3://{args.bucket}/{prefix}"

    topup_rows = [
        json.loads(l)
        for l in (run_dir / "plan/topup_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    tok_index = [
        json.loads(l)
        for l in (run_dir / "plan/topup_tokenize_index.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    print("listing existing S3 objects for resume...", flush=True)
    remote_sizes = list_s3_sizes(args.bucket, prefix)
    print(f"listed {len(remote_sizes)} objects under s3://{args.bucket}/{prefix}/", flush=True)

    # Upload raw data shards (append; skip already-uploaded for resume).
    for row in topup_rows:
        local = run_dir / "data" / row["path"]
        if not local.is_file():
            raise SystemExit(f"missing local shard {local}")
        uri = f"{root}/{row['path']}"
        cp_if_needed(local, uri, remote_sizes=remote_sizes, root=root, dry_run=args.dry_run)

    # Upload tokenized shards + sidecar json (resume-safe).
    shard_reports = []
    for row in tok_index:
        npy = run_dir / "tokenized" / row["npy"]
        meta = npy.with_suffix(".json")
        if not npy.is_file():
            raise SystemExit(f"missing tokenized {npy}")
        tokens = 0
        if meta.is_file():
            tokens = int(json.loads(meta.read_text(encoding="utf-8")).get("tokens") or 0)
        else:
            tokens = npy.stat().st_size // 4
            meta.write_text(
                json.dumps({"tokens": tokens, "path": row["npy"]}, indent=2) + "\n",
                encoding="utf-8",
            )
        cp_if_needed(
            npy,
            f"{root}/tokenized/{row['npy']}",
            remote_sizes=remote_sizes,
            root=root,
            dry_run=args.dry_run,
        )
        meta_uri = f"{root}/tokenized/{Path(row['npy']).with_suffix('.json').as_posix()}"
        cp_if_needed(
            meta, meta_uri, remote_sizes=remote_sizes, root=root, dry_run=args.dry_run
        )
        shard_reports.append(
            {
                "path": row["npy"],
                "manifest_path": row["manifest_path"],
                "domain": row["domain"],
                "tokens": tokens,
                "bytes": npy.stat().st_size,
                "topup": True,
            }
        )

    # Merge manifests.
    old_manifest = [
        json.loads(l)
        for l in (run_dir / "plan/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    existing_paths = {r["path"] for r in old_manifest}
    merged_manifest = list(old_manifest)
    for row in topup_rows:
        if row["path"] in existing_paths:
            continue
        merged_manifest.append(
            {
                "path": row["path"],
                "size": row["size"],
                "domain": row["domain"],
                "est_tokens": row.get("est_tokens"),
                "topup": True,
            }
        )

    old_tok = json.loads((run_dir / "plan/tokenized_manifest.json").read_text(encoding="utf-8"))
    old_paths = {s["path"] for s in old_tok["shards"]}
    new_shards = list(old_tok["shards"])
    for s in shard_reports:
        if s["path"] not in old_paths:
            new_shards.append(s)
    total_tokens = sum(int(s.get("tokens") or 0) for s in new_shards)
    new_tok = dict(old_tok)
    new_tok["shards"] = new_shards
    new_tok["total_content_tokens"] = total_tokens
    new_tok["topup_appended_at"] = datetime.now(timezone.utc).isoformat()
    new_tok["topup_domains"] = sorted({s["domain"] for s in shard_reports})

    by_domain: dict[str, int] = {}
    for s in new_shards:
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + int(s.get("tokens") or 0)

    (run_dir / "plan/manifest_merged.jsonl").write_text(
        "\n".join(json.dumps(r) for r in merged_manifest) + "\n", encoding="utf-8"
    )
    (run_dir / "plan/tokenized_manifest_merged.json").write_text(
        json.dumps(new_tok, indent=2) + "\n", encoding="utf-8"
    )
    availability = {
        "updated_at": new_tok["topup_appended_at"],
        "measured_tokens_by_domain": by_domain,
        "planned_available": {
            "dclm": 28_600_000_000,
            "arxiv": 20_800_000_000,
            "starcoder": 20_300_000_000,
            "pes2o": 26_300_000_000,
            "open-web-math": 12_200_000_000,
            "algebraic-stack": 11_800_000_000,
            "wiki": 3_660_000_000,
        },
    }
    for d, planned in availability["planned_available"].items():
        meas = by_domain.get(d, 0)
        err = abs(planned - meas) / meas if meas else None
        availability.setdefault("rel_err", {})[d] = err
        availability.setdefault("within_10pct", {})[d] = bool(err is not None and err <= 0.10)
    (run_dir / "plan/availability_after_topup.json").write_text(
        json.dumps(availability, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(availability, indent=2))

    if args.dry_run:
        print("dry-run: skipping manifest publish")
        return 0

    # Backup prior manifests BEFORE overwriting.
    run(
        [
            "aws",
            "s3",
            "cp",
            str(run_dir / "plan/manifest.jsonl"),
            f"{root}/plan/manifest.pre_topup.jsonl",
        ]
    )
    run(
        [
            "aws",
            "s3",
            "cp",
            str(run_dir / "plan/tokenized_manifest.json"),
            f"{root}/plan/tokenized_manifest.pre_topup.json",
        ]
    )
    run(["aws", "s3", "cp", str(run_dir / "plan/manifest_merged.jsonl"), f"{root}/plan/manifest.jsonl"])
    run(
        [
            "aws",
            "s3",
            "cp",
            str(run_dir / "plan/tokenized_manifest_merged.json"),
            f"{root}/plan/tokenized_manifest.json",
        ]
    )
    run(
        [
            "aws",
            "s3",
            "cp",
            str(run_dir / "plan/availability_after_topup.json"),
            f"{root}/plan/availability_after_topup.json",
        ]
    )
    run(
        [
            "aws",
            "s3",
            "cp",
            str(run_dir / "plan/topup_summary.json"),
            f"{root}/plan/topup_summary.json",
        ]
    )
    print("olmohq top-up published (regmix-10b untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
