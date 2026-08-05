#!/usr/bin/env python3
"""Tokenize one holdout document shard (Slurm array over tokenize_tasks.txt).

Task line: ``source domain split documents-NNNNN.jsonl.gz``
Writes: tokenized/<source>/<domain>/<split>.parts/<stem>.npy (+ .json)
Downstream merge_tokenized.py concatenates parts into <split>.npy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from olmo_shard_utils import doc_text, iter_docs  # noqa: E402
from refhq_new_sources import EOS_TOKEN_ID, TOKENIZER_ID  # noqa: E402
from trim_and_tokenize_regmix import TokenWriter, _encode_batch, _worker_init  # noqa: E402


def tokenize_shard(
    *,
    source: str,
    domain: str,
    split: str,
    shard_path: Path,
    parts_dir: Path,
    tokenizer: str,
    eos_token_id: int,
    workers: int,
    batch_size: int,
) -> dict:
    if not shard_path.is_file():
        raise SystemExit(f"missing holdout shard: {shard_path}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    stem = shard_path.name
    for suffix in (".jsonl.gz", ".json.gz", ".gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out_npy = parts_dir / f"{stem}.npy"
    out_meta = parts_dir / f"{stem}.json"

    writer = TokenWriter(out_npy, eos_token_id)
    docs = 0
    content_tokens = 0

    def _iter_text_batches():
        nonlocal docs
        batch_texts: list[str] = []
        print(f"read {shard_path}", flush=True)
        for obj in iter_docs(shard_path):
            text = doc_text(obj)
            if not text:
                continue
            batch_texts.append(text)
            docs += 1
            if len(batch_texts) >= batch_size:
                yield batch_texts
                batch_texts = []
            if docs % 50_000 == 0:
                print(
                    f"progress {source}/{domain}/{split}/{stem} docs={docs:,} "
                    f"content_tokens={content_tokens:,}",
                    flush=True,
                )
        if batch_texts:
            yield batch_texts

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(tokenizer,),
    ) as pool:
        for batch_texts in _iter_text_batches():
            id_batches = pool.submit(_encode_batch, batch_texts).result()
            for ids in id_batches:
                content_tokens += len(ids)
                writer.write_doc_ids(ids)

    stream_tokens = writer.finalize()
    meta = {
        "source": source,
        "domain": domain,
        "split": split,
        "shard": shard_path.name,
        "tokenizer": tokenizer,
        "eos_token_id": eos_token_id,
        "docs": docs,
        "content_tokens": content_tokens,
        "stream_tokens_with_eos": stream_tokens,
        "tokenized_npy": str(out_npy),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def _load_task(tasks_file: Path, task_index: int) -> tuple[str, str, str, str]:
    lines = [ln.strip() for ln in tasks_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if task_index < 0 or task_index >= len(lines):
        raise SystemExit(f"task_index {task_index} out of range (n={len(lines)})")
    parts = lines[task_index].split()
    # New: source domain split shard_name
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    # Legacy: source domain split  → process ALL shards for that split (serial)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], ""
    raise SystemExit(f"bad tokenize task line: {lines[task_index]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--tasks-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    scratch = Path(plan["scratch_root"])
    tasks_file = args.tasks_file or (scratch / "manifests" / "tokenize_tasks.txt")
    source, domain, split, shard_name = _load_task(tasks_file, args.task_index)

    if source not in plan["sources"]:
        raise SystemExit(f"source {source} missing from plan")
    src_plan = plan["sources"][source]
    holdout_root = Path(src_plan["paths"]["holdout"])
    tok_root = Path(src_plan["paths"]["tokenized"])
    parts_dir = tok_root / domain / f"{split}.parts"

    if shard_name:
        shard_path = holdout_root / domain / split / shard_name
        meta = tokenize_shard(
            source=source,
            domain=domain,
            split=split,
            shard_path=shard_path,
            parts_dir=parts_dir,
            tokenizer=args.tokenizer,
            eos_token_id=args.eos_token_id,
            workers=max(1, args.workers),
            batch_size=args.batch_size,
        )
    else:
        # Legacy 3-field task: tokenize every shard under the split.
        split_dir = holdout_root / domain / split
        shards = sorted(split_dir.glob("documents-*.jsonl.gz"))
        if not shards:
            shards = sorted(split_dir.glob("documents-*.json.gz"))
        if not shards:
            raise SystemExit(f"no holdout shards under {split_dir}")
        metas = []
        for shard_path in shards:
            metas.append(
                tokenize_shard(
                    source=source,
                    domain=domain,
                    split=split,
                    shard_path=shard_path,
                    parts_dir=parts_dir,
                    tokenizer=args.tokenizer,
                    eos_token_id=args.eos_token_id,
                    workers=max(1, args.workers),
                    batch_size=args.batch_size,
                )
            )
        meta = {"source": source, "domain": domain, "split": split, "parts": len(metas)}

    # Mirror parts tree into tok/ for finalize discovery
    mirror_parts = scratch / "tok" / source / domain / f"{split}.parts"
    mirror_parts.mkdir(parents=True, exist_ok=True)
    for p in parts_dir.glob("*.npy"):
        dest = mirror_parts / p.name
        if dest.resolve() != p.resolve():
            shutil.copy2(p, dest)
        j = p.with_suffix(".json")
        if j.is_file():
            shutil.copy2(j, mirror_parts / j.name)
    print(json.dumps({"mirrored_parts": str(mirror_parts), "meta": meta}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
