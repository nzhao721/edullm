#!/usr/bin/env python3
"""Upload mixlaw validation corpora to s3://edullm-datasets/mixlaw/.

Never writes to regmix-10b. For mix01, performs a server-side copy *from*
regmix-10b into mixlaw/mixes/mix01/ (destination only).
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--mixtures-json", type=Path, required=True)
    ap.add_argument("--dst-bucket", default="edullm-datasets")
    ap.add_argument("--dst-prefix", default="mixlaw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mixes = json.loads(args.mixtures_json.read_text(encoding="utf-8"))
    dst_root = f"s3://{args.dst_bucket}/{args.dst_prefix.strip('/')}"
    slices = args.run_dir / "slices"
    uploaded: list[dict] = []

    # README + recipe first (small).
    readme = args.run_dir / "MIXLAW_README.md"
    if not readme.is_file():
        readme.write_text(
            "# mixlaw validation corpora\n\n"
            "10B-token mixtures for skill-dag 370M scale-up.\n"
            "mix01 is a server-side copy of regmix-10b (source untouched).\n",
            encoding="utf-8",
        )

    if not args.dry_run:
        run(["aws", "s3", "cp", str(args.mixtures_json), f"{dst_root}/validation_mixtures_10b.json"])
        run(["aws", "s3", "cp", str(readme), f"{dst_root}/README.md"])

    for row in mixes["mixtures"]:
        name = row["run_name"]
        dest = f"{dst_root}/mixes/{name}"
        if row.get("reuse_s3"):
            src = row["reuse_s3"].rstrip("/")
            print(f"server-side copy {src}/ → {dest}/ (regmix source read-only)", flush=True)
            if not args.dry_run:
                # Copy tokenized + paths only; never sync back to source.
                run(
                    [
                        "aws",
                        "s3",
                        "sync",
                        f"{src}/tokenized/",
                        f"{dest}/tokenized/",
                        "--only-show-errors",
                    ]
                )
                # Best-effort small metadata if present.
                for rel in ("README.md", "plan/summary.json"):
                    subprocess.run(
                        ["aws", "s3", "cp", f"{src}/{rel}", f"{dest}/{rel}", "--only-show-errors"],
                        check=False,
                    )
            uploaded.append(
                {
                    "run_name": name,
                    "source": "server_side_copy_from_regmix_10b",
                    "src": src,
                    "dst": dest,
                    "regmix_modified": False,
                }
            )
            continue

        mix_dir = slices / name
        if not mix_dir.is_dir():
            raise SystemExit(f"missing slice dir {mix_dir}")
        print(f"upload {mix_dir} → {dest}/", flush=True)
        if not args.dry_run:
            run(["aws", "s3", "sync", str(mix_dir), f"{dest}/", "--only-show-errors"])
        uploaded.append({"run_name": name, "source": "materialized", "dst": dest})

    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dst": dst_root,
        "mixtures": uploaded,
        "regmix_10b_policy": "read_only_never_modify",
    }
    receipt_path = args.run_dir / "mixlaw_upload_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if not args.dry_run:
        run(["aws", "s3", "cp", str(receipt_path), f"{dst_root}/mixlaw_upload_receipt.json"])
        # Publish READY last.
        ready = args.run_dir / "READY"
        ready.write_text(receipt["created_at"] + "\n", encoding="utf-8")
        run(["aws", "s3", "cp", str(ready), f"{dst_root}/READY"])

    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
