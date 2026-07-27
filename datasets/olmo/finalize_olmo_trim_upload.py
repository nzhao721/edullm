#!/usr/bin/env python3
"""After domain trims finish: replace overshot shards in S3 + update plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig


OVERSHOT = [
    "algebraic-stack",
    "arxiv",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bucket", default="edullm-dataset-olmo")
    parser.add_argument("--prefix", default="olmo-mix-1124-30b")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    run = args.run_dir
    summary = json.loads((run / "plan" / "summary.json").read_text(encoding="utf-8"))
    old_manifest = [
        json.loads(line)
        for line in (run / "plan" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    s3 = boto3.client("s3", region_name=args.region)
    cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=16,
        use_threads=True,
    )

    # Delete old overshot objects from S3.
    to_delete = [m for m in old_manifest if m["domain"] in OVERSHOT]
    print(f"deleting {len(to_delete)} old overshot objects from s3", flush=True)
    for i in range(0, len(to_delete), 1000):
        chunk = to_delete[i : i + 1000]
        s3.delete_objects(
            Bucket=args.bucket,
            Delete={
                "Objects": [{"Key": f"{args.prefix}/{m['path']}"} for m in chunk],
                "Quiet": True,
            },
        )

    new_manifest = [m for m in old_manifest if m["domain"] not in OVERSHOT]
    trim_meta = []

    for domain in OVERSHOT:
        result_path = run / "trim" / domain / "trim_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        local = Path(result["output_shard"])
        rel = f"data/{domain}/{domain}-trimmed.json.gz"
        key = f"{args.prefix}/{rel}"
        print(f"upload {local} -> s3://{args.bucket}/{key}", flush=True)
        s3.upload_file(
            str(local),
            args.bucket,
            key,
            ExtraArgs={
                "Metadata": {
                    "domain": domain,
                    "tokens": str(result["tokens_after"]),
                    "tokenizer": result["tokenizer"],
                    "trimmed": "true",
                }
            },
            Config=cfg,
        )
        new_manifest.append(
            {
                "path": rel,
                "size": local.stat().st_size,
                "domain": domain,
                "est_tokens": result["tokens_after"],
                "measured_tokens": result["tokens_after"],
                "tokenizer": result["tokenizer"],
                "trimmed": True,
                "docs_after": result["docs_after"],
            }
        )
        # Update summary domain block with measured tokens.
        summary["domains"][domain]["files_selected"] = 1
        summary["domains"][domain]["bytes_selected"] = local.stat().st_size
        summary["domains"][domain]["est_tokens_selected"] = result["tokens_after"]
        summary["domains"][domain]["measured_tokens"] = result["tokens_after"]
        summary["domains"][domain]["docs_before"] = result["docs_before"]
        summary["domains"][domain]["docs_after"] = result["docs_after"]
        summary["domains"][domain]["docs_scanned"] = result.get("docs_scanned")
        summary["domains"][domain]["tokens_before_measured"] = result.get("tokens_before")
        summary["domains"][domain]["relative_error"] = result.get("relative_error")
        trim_meta.append(result)

    new_manifest.sort(key=lambda x: x["path"])
    for i, row in enumerate(new_manifest):
        row["index"] = i

    summary["files_selected"] = len(new_manifest)
    summary["bytes_selected"] = sum(m["size"] for m in new_manifest)
    summary["est_tokens_selected"] = sum(
        m.get("measured_tokens", m.get("est_tokens", 0)) for m in new_manifest
    )
    summary["tokenizer"] = "allenai/OLMo-2-0425-1B"
    summary["trim"] = {
        "method": "document-shuffle-then-keep-to-target",
        "seed": 42,
        "domains": OVERSHOT,
        "results": trim_meta,
    }

    plan_dir = run / "plan"
    man_path = plan_dir / "manifest.jsonl"
    sum_path = plan_dir / "summary.json"
    with man_path.open("w", encoding="utf-8") as fh:
        for row in new_manifest:
            fh.write(json.dumps(row) + "\n")
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for rel, path in [
        ("plan/manifest.jsonl", man_path),
        ("plan/summary.json", sum_path),
        ("plan/trim_results.json", run / "trim" / "trim_results.json"),
    ]:
        if rel.endswith("trim_results.json"):
            (run / "trim" / "trim_results.json").write_text(
                json.dumps(trim_meta, indent=2), encoding="utf-8"
            )
        key = f"{args.prefix}/{rel}"
        s3.upload_file(str(path if not rel.endswith("trim_results.json") else run / "trim" / "trim_results.json"), args.bucket, key)
        print(f"uploaded s3://{args.bucket}/{key}", flush=True)

    readme = f"""# edullm OLMo-mix-1124 ~30B sample (document-trimmed)

- Source: allenai/olmo-mix-1124
- Shard sample seed: 42
- Overshot domains trimmed with tokenizer: allenai/OLMo-2-0425-1B (OLMo-2-1B)
- Method: shuffle documents within domain, keep until measured token target
- Trimmed domains: {', '.join(OVERSHOT)}
- See plan/summary.json for measured token counts
"""
    s3.put_object(Bucket=args.bucket, Key=f"{args.prefix}/README.md", Body=readme.encode())
    print(json.dumps({"files": len(new_manifest), "est_tokens": summary["est_tokens_selected"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
