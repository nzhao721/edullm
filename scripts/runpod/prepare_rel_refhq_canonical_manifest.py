#!/usr/bin/env python3
"""Reuse staged REL-RefHQ inputs with the canonical OLMo token-selection runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.tokens_dir.resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    uri_root = str(manifest["source_uri"]).rstrip("/")
    objects: list[dict[str, object]] = []
    logical_paths: list[str] = []
    for shard in manifest["shards"]:
        relative = str(shard["path"])
        path = (source / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        uri = f"{uri_root}/{relative}"
        objects.append(
            {
                "uri": uri,
                "path": str(path),
                "size": path.stat().st_size,
                "etag": "staged-edullm-data",
                "version_id": None,
            }
        )
        logical_paths.append(uri)

    reference = args.reference.resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    payload = {
        "schema_version": 1,
        "family": "token-selection",
        "arm": "rel-ema-refhq",
        "object_list_sha256": hashlib.sha256(
            "\n".join(logical_paths).encode("utf-8")
        ).hexdigest(),
        "total_bytes": sum(int(record["size"]) for record in objects)
        + reference.stat().st_size,
        "corpora": {
            "pretrain/regmix-10b": {
                "dataset_id": "pretrain/regmix-10b",
                "version": str(manifest["dataset_version"]),
                "dtype": str(manifest["dtype"]),
                "rows": int(manifest["n_tokens"]),
                "logical_paths": logical_paths,
                "objects": objects,
            }
        },
        "references": {"reference": str(reference)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "objects": len(objects),
                "total_bytes": payload["total_bytes"],
                "reference": str(reference),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
