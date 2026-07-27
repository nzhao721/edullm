#!/usr/bin/env python3
"""Build one HQ reference domain on FarmShare.

Randomly stream documents; for each doc (or Dolma mini-batch) apply filters,
tokenize passers, and stop as soon as the token budget is reached. No
predetermined headroom — ingest continues only until the budget is filled.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from hq_reference_sources import (  # noqa: E402
    DOLMA_MINI_BATCH_DOCS,
    HQ_SOURCES,
    TOKENIZER_ID,
    within_budget,
)
from olmo_shard_utils import count_batch, doc_text, iter_docs, worker_init  # noqa: E402


def _write_docs_jsonl_gz(docs: list[dict], out_dir: Path, prefix: str = "documents") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    shard_idx = 0
    handle = None
    n = 0
    max_per = 50_000
    try:
        for doc in docs:
            if handle is None or n >= max_per:
                if handle is not None:
                    handle.close()
                path = out_dir / f"{prefix}-{shard_idx:05d}.json.gz"
                handle = gzip.open(path, "wt", encoding="utf-8")
                shards.append(path)
                shard_idx += 1
                n = 0
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    finally:
        if handle is not None:
            handle.close()
    return shards


def _count_tokens(texts: list[str], batch_size: int = 64) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), batch_size):
        out.extend(count_batch(texts[i : i + batch_size]))
    return out


def _iter_hf_stream(src: dict[str, Any], token: str | None) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:  # noqa: BLE001
            print(f"huggingface_hub.login warning: {exc!r}", flush=True)

    kwargs: dict[str, Any] = {
        "path": src["repo_id"],
        "split": src.get("split") or "train",
        "streaming": True,
    }
    if token:
        kwargs["token"] = token
    if src.get("config"):
        kwargs["name"] = src["config"]
    # Script-based datasets (peS2o, older hubs) need trust_remote_code on datasets 2.x.
    kwargs["trust_remote_code"] = True
    try:
        ds = load_dataset(**kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        ds = load_dataset(**kwargs)
    text_field = src.get("text_field") or "text"
    for i, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        text = row.get(text_field) or row.get("content") or row.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        yield {
            "id": str(row.get("id") or row.get("url") or f"{src['repo_id']}-{i}"),
            "text": text,
            "metadata": row.get("metadata") or {},
            "source": src["repo_id"],
        }


def _list_local_shard_paths(raw_dir: Path) -> list[Path]:
    return sorted(
        list(raw_dir.rglob("*.json.gz"))
        + list(raw_dir.rglob("*.jsonl.gz"))
        + list(raw_dir.rglob("*.jsonl.zstd"))
        + list(raw_dir.rglob("*.jsonl.zst"))
    )


def _iter_local_shards_shuffled(raw_dir: Path, seed: int) -> Iterator[dict[str, Any]]:
    paths = _list_local_shard_paths(raw_dir)
    rng = random.Random(seed)
    rng.shuffle(paths)
    for path in paths:
        for obj in iter_docs(path):
            text = doc_text(obj)
            if not text:
                continue
            yield {
                "id": str(obj.get("id") or path.name),
                "text": text,
                "metadata": obj.get("metadata") or obj.get("meta") or {},
                "source": str(path),
            }


def _shuffled_doc_stream(docs_iter: Iterator[dict], seed: int, buffer_size: int = 10_000) -> Iterator[dict]:
    """Reservoir-shuffle windows so acceptance order is not pure stream order."""
    rng = random.Random(seed)
    buf: list[dict] = []
    for doc in docs_iter:
        buf.append(doc)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            yield from buf
            buf = []
    if buf:
        rng.shuffle(buf)
        yield from buf


def _keep_openwebmath_hq(doc: dict) -> bool:
    from refhq.math_quality import keep_openwebmath_hq_record

    record = {"metadata": doc.get("metadata"), "text": doc["text"]}
    return keep_openwebmath_hq_record(record, doc["text"])


def _keep_algebraic(doc: dict) -> bool:
    from refhq.math_quality import keep_algebraic_stack_text

    return keep_algebraic_stack_text(doc["text"])


def fill_until_budget(
    docs_iter: Iterator[dict],
    *,
    budget: float,
    seed: int,
    keep: Callable[[dict], bool] | None,
    progress_every: int = 5_000,
) -> tuple[list[dict], dict[str, Any]]:
    """Filter each doc as sampled; tokenize passers until budget is hit."""
    worker_init(TOKENIZER_ID)
    selected: list[dict] = []
    tokens = 0
    seen = 0
    kept = 0
    rejected = 0
    pending: list[dict] = []

    def _flush_pending() -> None:
        nonlocal tokens, pending
        if not pending or tokens >= budget:
            pending = []
            return
        counts = _count_tokens([d["text"] for d in pending])
        for d, ntok in zip(pending, counts):
            if tokens >= budget:
                break
            d = dict(d)
            d["n_tokens"] = int(ntok)
            selected.append(d)
            tokens += int(ntok)
        pending = []

    for doc in _shuffled_doc_stream(docs_iter, seed):
        seen += 1
        if keep is not None and not keep(doc):
            rejected += 1
        else:
            kept += 1
            pending.append(doc)
            if len(pending) >= 256:
                _flush_pending()
        if seen % progress_every == 0:
            print(
                f"progress seen={seen} kept={kept} rejected={rejected} tokens={tokens:.3e}",
                flush=True,
            )
        if tokens >= budget:
            break

    _flush_pending()
    while selected and tokens > budget * 1.02:
        last = selected.pop()
        tokens -= int(last.get("n_tokens", 0))

    stats = {
        "seen_docs": seen,
        "kept_docs": kept,
        "rejected_docs": rejected,
        "filter_pass_rate": (kept / seen) if seen else 0.0,
        "realized_tokens": tokens,
        "doc_count": len(selected),
        "fill_until_budget": True,
    }
    return selected, stats


def _dolma_filter_batch(docs: list[dict], work_dir: Path, batch_id: int, processes: int) -> list[dict]:
    """Run Dolma code-hq tag+mix on one mini-batch; return surviving docs."""
    from refhq.process import apply_dolma_pre_mix_domain

    batch_dir = work_dir / f"batch-{batch_id:05d}"
    # Dolma substitutes ".../documents/..." → ".../attributes/..."; require that path segment.
    docs_dir = batch_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    shard = docs_dir / "part-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(
                json.dumps(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "source": doc.get("source", "starcoder"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    try:
        apply_dolma_pre_mix_domain(
            domain="code-hq",
            shard_path=shard,
            work_root=batch_dir / "dolma",
            processes=processes,
        )
    except RuntimeError as exc:
        print(f"dolma unavailable for batch {batch_id}: {exc}; keeping batch unfiltered", flush=True)
        return docs

    out: list[dict] = []
    for i, obj in enumerate(iter_docs(shard)):
        text = doc_text(obj)
        if not text:
            continue
        out.append({"id": str(obj.get("id") or i), "text": text, "source": "starcoder-hq"})
    return out


def fill_until_budget_dolma(
    docs_iter: Iterator[dict],
    *,
    budget: float,
    seed: int,
    work_dir: Path,
    processes: int,
    batch_docs: int = DOLMA_MINI_BATCH_DOCS,
) -> tuple[list[dict], dict[str, Any]]:
    """Random stream → Dolma mini-batches → tokenize passers until budget."""
    worker_init(TOKENIZER_ID)
    selected: list[dict] = []
    tokens = 0
    seen = 0
    kept = 0
    rejected = 0
    batch_id = 0
    buf: list[dict] = []

    def _consume_passers(passers: list[dict]) -> None:
        nonlocal tokens, kept
        if not passers or tokens >= budget:
            return
        counts = _count_tokens([d["text"] for d in passers])
        for d, ntok in zip(passers, counts):
            if tokens >= budget:
                break
            d = dict(d)
            d["n_tokens"] = int(ntok)
            selected.append(d)
            tokens += int(ntok)
            kept += 1

    for doc in _shuffled_doc_stream(docs_iter, seed):
        seen += 1
        buf.append(doc)
        if len(buf) < batch_docs and tokens < budget:
            continue
        batch_id += 1
        before = len(buf)
        passers = _dolma_filter_batch(buf, work_dir, batch_id, processes)
        rejected += before - len(passers)
        _consume_passers(passers)
        buf = []
        print(
            f"dolma batch={batch_id} seen={seen} kept={kept} rejected={rejected} tokens={tokens:.3e}",
            flush=True,
        )
        if tokens >= budget:
            break

    if buf and tokens < budget:
        batch_id += 1
        before = len(buf)
        passers = _dolma_filter_batch(buf, work_dir, batch_id, processes)
        rejected += before - len(passers)
        _consume_passers(passers)

    while selected and tokens > budget * 1.02:
        last = selected.pop()
        tokens -= int(last.get("n_tokens", 0))
        kept = max(kept - 1, 0)

    stats = {
        "seen_docs": seen,
        "kept_docs": kept,
        "rejected_docs": rejected,
        "filter_pass_rate": (kept / seen) if seen else 0.0,
        "realized_tokens": tokens,
        "doc_count": len(selected),
        "dolma_batches": batch_id,
        "fill_until_budget": True,
    }
    return selected, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--processes", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--max-stream-docs", type=int, default=0, help="Optional cap for dry runs")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    domain_plan = plan["domains"][args.domain]
    src = HQ_SOURCES[args.domain]
    budget = float(domain_plan["budget_tokens"])
    seed = int(domain_plan["seed"])
    raw_dir = Path(domain_plan["paths"]["raw"])
    work_dir = Path(domain_plan["paths"]["work"])
    out_dir = Path(domain_plan["paths"]["out"])
    stats_path = Path(domain_plan["paths"]["stats"])
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(
        f"build domain={args.domain} budget={budget:.3e} filter={src['filter']} "
        f"mode=fill-until-budget",
        flush=True,
    )

    if args.domain == "dclm":
        from sample_datadecide_dclm import main as dclm_main

        sys.argv = [
            "sample_datadecide_dclm.py",
            "--raw-dir",
            str(raw_dir),
            "--out-dir",
            str(out_dir),
            "--target-tokens",
            str(int(budget)),
            "--seed",
            str(seed),
            "--tokenizer",
            plan.get("tokenizer_id", TOKENIZER_ID),
            "--source-tokenizer",
            str(src.get("source_tokenizer_id", "allenai/gpt-neox-olmo-dolma-v1_5")),
            "--stats-out",
            str(stats_path),
        ]
        return dclm_main()

    keep: Callable[[dict], bool] | None
    if src["filter"] == "openwebmath-hq":
        keep = _keep_openwebmath_hq
    elif src["filter"] == "algebraic-stack-heuristic":
        keep = _keep_algebraic
    else:
        keep = None

    if src["kind"] == "olmo_mix_domain":
        docs_iter: Iterator[dict] = _iter_local_shards_shuffled(raw_dir, seed)
    else:
        docs_iter = _iter_hf_stream(src, token)

    if args.max_stream_docs > 0:
        base_iter = docs_iter

        def _limited() -> Iterator[dict]:
            for i, d in enumerate(base_iter):
                if i >= args.max_stream_docs:
                    break
                yield d

        docs_iter = _limited()

    if src["filter"] == "dolma-code-hq":
        selected, stats = fill_until_budget_dolma(
            docs_iter,
            budget=budget,
            seed=seed,
            work_dir=work_dir,
            processes=args.processes,
        )
    else:
        selected, stats = fill_until_budget(
            docs_iter, budget=budget, seed=seed, keep=keep
        )

    shards = _write_docs_jsonl_gz(selected, out_dir)
    stats.update(
        {
            "domain": args.domain,
            "budget_tokens": budget,
            "within_budget": within_budget(stats["realized_tokens"], budget),
            "seed": seed,
            "tokenizer_id": plan.get("tokenizer_id", TOKENIZER_ID),
            "filter": src["filter"],
            "hf_dataset_id": src.get("repo_id"),
            "shards": [str(p) for p in shards],
        }
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    if not stats["within_budget"]:
        print(
            f"WARNING: realized {stats['realized_tokens']} outside ±2% of {budget}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
