#!/usr/bin/env python3
"""Copy a deterministic, bounded raw Hugging Face dataset subset to S3.

This is intentionally an import-only utility: it never reads document text,
changes document format, filters, tags, deduplicates, or tokenizes data.
It chooses source files by their Hub-reported compressed sizes and writes the
original files unchanged, along with an import manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


DATA_SUFFIXES = (".parquet", ".json", ".jsonl", ".jsonl.gz", ".json.gz", ".zst", ".arrow")


@dataclass(frozen=True)
class Source:
    name: str
    repo_id: str
    domain: str
    target_tokens: int
    compressed_bytes_per_token: float
    include_any: tuple[str, ...] = ()

    @property
    def raw_token_target(self) -> int:
        return int(self.target_tokens * 1.30)

    @property
    def byte_budget(self) -> int:
        return int(self.raw_token_target * self.compressed_bytes_per_token)


SOURCES: dict[str, Source] = {
    "starcoderdata": Source("starcoderdata", "bigcode/starcoderdata", "code", 25_000_000_000, 3.15),
    "pes2o": Source("pes2o", "allenai/peS2o", "stem", 15_000_000_000, 2.30),
    "arxiv_s2orc": Source("arxiv_s2orc", "AlgorithmicResearchGroup/arxiv_s2orc_parsed", "stem", 10_000_000_000, 2.30),
    "openwebmath": Source("openwebmath", "open-web-math/open-web-math", "math", 12_000_000_000, 1.90),
    "algebraic_stack": Source("algebraic_stack", "typeof/algebraic-stack", "math", 3_000_000_000, 2.20),
    "olmocr_hist_geo": Source("olmocr_hist_geo", "adorkin/olmocr_science_pdfs-history_and_geography", "history", 12_000_000_000, 2.40),
    "pg19": Source("pg19", "deepmind/pg19", "literature", 2_000_000_000, 3.00),
    "project_gutenberg": Source(
        "project_gutenberg",
        "common-pile/project_gutenberg_filtered",
        "literature",
        3_000_000_000,
        2.50,
    ),
    "stackexchange": Source(
        "stackexchange",
        "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
        "conversational",
        10_000_000_000,
        2.30,
        (
            "english/", "ell/", "parenting/", "workplace/", "history/",
            "philosophy/", "cooking/", "travel/", "interpersonal/", "money/",
            "law/", "politics/", "skeptics/", "cogsci/", "diy/", "gardening/",
            "fitness/",
        ),
    ),
    "wikipedia": Source(
        "wikipedia",
        "wikimedia/wikipedia",
        "encyclopedic",
        8_000_000_000,
        2.20,
        ("20231101.en/",),
    ),
}


def configure_logging(source: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)sZ {source} %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


def is_data_file(path: str, source: Source) -> bool:
    path_lower = path.lower()
    if not path_lower.endswith(DATA_SUFFIXES):
        return False
    if source.include_any and not any(marker in path for marker in source.include_any):
        return False
    return True


def object_exists(bucket: str, key: str) -> bool:
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def upload_file(local_path: str, bucket: str, key: str) -> None:
    subprocess.run(
        [
            "aws", "s3", "cp", local_path, f"s3://{bucket}/{key}",
            "--only-show-errors",
        ],
        check=True,
    )


def write_manifest(bucket: str, key: str, manifest: dict[str, Any], work_dir: Path) -> None:
    local_manifest = work_dir / "manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    upload_file(str(local_manifest), bucket, key)


def select_files(api: HfApi, source: Source, revision: str) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    for entry in api.list_repo_tree(source.repo_id, repo_type="dataset", revision=revision, recursive=True):
        path = getattr(entry, "path", "")
        size = getattr(entry, "size", None)
        if path and size and is_data_file(path, source):
            files.append((path, int(size)))

    if not files and source.include_any:
        logging.warning("No files matched configured prefixes; falling back to all data files")
        for entry in api.list_repo_tree(source.repo_id, repo_type="dataset", revision=revision, recursive=True):
            path = getattr(entry, "path", "")
            size = getattr(entry, "size", None)
            if path and size and path.lower().endswith(DATA_SUFFIXES):
                files.append((path, int(size)))

    files.sort(key=lambda item: hashlib.sha256(f"{source.repo_id}:{item[0]}".encode()).digest())
    selected: list[tuple[str, int]] = []
    cumulative = 0
    for path, size in files:
        selected.append((path, size))
        cumulative += size
        if cumulative >= source.byte_budget:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="raw-import-v1")
    parser.add_argument("--work-dir", default="/mnt/ingest")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    source = SOURCES[args.source]
    worker_name = f"{source.name}-{args.shard_index:03d}-of-{args.shard_count:03d}"
    configure_logging(worker_name)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_info = api.repo_info(source.repo_id, repo_type="dataset")
    revision = repo_info.sha
    logging.info("start repo=%s revision=%s raw_token_target=%d byte_budget=%d",
                 source.repo_id, revision, source.raw_token_target, source.byte_budget)

    selected_all = select_files(api, source, revision)
    selected = selected_all[args.shard_index::args.shard_count]
    if not selected_all:
        raise RuntimeError(f"No importable data files found in {source.repo_id}")

    planned_bytes = sum(size for _, size in selected)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_unix": int(time.time()),
        "source": asdict(source),
        "raw_token_target": source.raw_token_target,
        "compressed_byte_budget": source.byte_budget,
        "source_planned_compressed_bytes": sum(size for _, size in selected_all),
        "worker_planned_compressed_bytes": planned_bytes,
        "revision": revision,
        "worker": {
            "name": worker_name,
            "index": args.shard_index,
            "count": args.shard_count,
        },
        "selected_files": [{"path": path, "size": size} for path, size in selected],
        "completed_files": [],
        "completed_compressed_bytes": 0,
        "processing_performed": False,
    }
    manifest_key = f"{args.prefix}/manifests/{source.name}/{worker_name}.json"
    write_manifest(args.bucket, manifest_key, manifest, work_dir)

    for index, (path, size) in enumerate(selected, start=1):
        key = f"{args.prefix}/datasets/{source.name}/{revision}/{path}"
        if object_exists(args.bucket, key):
            logging.info("skip %d/%d existing=%s", index, len(selected), path)
        else:
            logging.info("download %d/%d bytes=%d path=%s", index, len(selected), size, path)
            local_path = hf_hub_download(
                repo_id=source.repo_id,
                repo_type="dataset",
                revision=revision,
                filename=path,
                local_dir=str(work_dir / "hf"),
            )
            logging.info("upload %d/%d s3_key=%s", index, len(selected), key)
            upload_file(local_path, args.bucket, key)
            Path(local_path).unlink(missing_ok=True)

        manifest["completed_files"].append({"path": path, "size": size})
        manifest["completed_compressed_bytes"] += size
        manifest["updated_at_unix"] = int(time.time())
        write_manifest(args.bucket, manifest_key, manifest, work_dir)
        logging.info(
            "progress files=%d/%d bytes=%d/%d",
            index, len(selected), manifest["completed_compressed_bytes"], planned_bytes,
        )

    manifest["status"] = "completed"
    manifest["completed_at_unix"] = int(time.time())
    write_manifest(args.bucket, manifest_key, manifest, work_dir)
    logging.info("complete files=%d bytes=%d", len(selected), manifest["completed_compressed_bytes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
