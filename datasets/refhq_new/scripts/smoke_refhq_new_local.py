#!/usr/bin/env python3
"""Local offline smoke for refhq-new: fixture → normalize → holdout → fake-tokenize → finalize.

No HuggingFace Hub, Dolma, FarmShare, or S3. Uses --fixture-jsonl for normalize and a
deterministic fake encoder (no dolma2 download) for tokenize.

Usage (from repo root):
  py -3 datasets/refhq_new/scripts/smoke_refhq_new_local.py
  py -3 datasets/refhq_new/scripts/smoke_refhq_new_local.py --scratch-root /tmp/refhq-smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from finalize_upload import build_manifest, resolve_tok_root, stage_tokens_from_tok  # noqa: E402
from holdout_docs import run_holdout  # noqa: E402
from normalize_filter_source import _iter_fixture_rows, normalize_rows  # noqa: E402
from olmo_shard_utils import doc_text, iter_docs  # noqa: E402
from refhq_new_sources import EOS_TOKEN_ID  # noqa: E402
from trim_and_tokenize_regmix import TokenWriter  # noqa: E402


FIXTURE_ROWS = [
    # Kept: general chat
    {
        "id": "keep-1",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "dataset": "sharegpt",
    },
    {
        "id": "keep-2",
        "messages": [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Attraction between masses."},
        ],
        "dataset": "cot",
    },
    {
        "id": "keep-3",
        "messages": [
            {"role": "user", "content": "Write a Python hello world."},
            {"role": "assistant", "content": "print('hello')"},
        ],
        "dataset": "code_alpaca",
    },
    # Dropped by openhermes language filter (when used with openhermes-25)
    {"id": "drop-lang", "language": "fr", "conversations": [{"from": "human", "value": "Bonjour"}]},
]


def _write_fixture(path: Path, *, source: str, n_docs: int) -> None:
    """Write enough synthetic rows that holdout carve is non-zero for large n."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if source == "tulu-v2":
            for i in range(n_docs):
                row = {
                    "id": f"tv2-{i}",
                    "dataset": "code_alpaca" if i % 3 == 0 else "sharegpt",
                    "messages": [
                        {"role": "user", "content": f"Question {i}?"},
                        {"role": "assistant", "content": f"Answer {i}."},
                    ],
                }
                handle.write(json.dumps(row) + "\n")
        elif source == "tulu-3":
            for i in range(n_docs):
                # Mix keep + drop rows
                if i % 20 == 0:
                    row = {
                        "id": f"t3-drop-{i}",
                        "source": "allenai/wildguardmix",
                        "messages": [{"role": "user", "content": "x"}],
                    }
                else:
                    row = {
                        "id": f"t3-{i}",
                        "source": "ai2-adapt-dev/flan_v2_converted",
                        "messages": [
                            {"role": "user", "content": f"flan {i}"},
                            {"role": "assistant", "content": f"ok {i}"},
                        ],
                    }
                handle.write(json.dumps(row) + "\n")
        else:
            for row in FIXTURE_ROWS:
                handle.write(json.dumps(row) + "\n")


def _copy_docs_to_out(docs_root: Path, out_root: Path) -> int:
    """Simulate Dolma English pass-through (offline)."""
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for domain_dir in sorted(p for p in docs_root.iterdir() if p.is_dir()):
        dest = out_root / domain_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for shard in sorted(domain_dir.glob("documents-*.jsonl.gz")):
            shutil.copy2(shard, dest / shard.name)
            n += 1
    return n


def _fake_tokenize_split(
    *,
    holdout_root: Path,
    tok_root: Path,
    source: str,
    domain: str,
    split: str,
) -> dict:
    split_dir = holdout_root / domain / split
    shards = sorted(split_dir.glob("documents-*.jsonl.gz"))
    if not shards:
        raise SystemExit(f"no holdout shards under {split_dir}")
    out_dir = tok_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"{split}.npy"
    writer = TokenWriter(out_npy, EOS_TOKEN_ID)
    docs = 0
    content_tokens = 0
    for shard in shards:
        for obj in iter_docs(shard):
            text = doc_text(obj)
            if not text:
                continue
            # Deterministic fake ids (no HF tokenizer): hash words → uint16 range.
            ids = [(sum(ord(c) for c in w) % 50_000) + 1 for w in text.split()[:64]]
            if not ids:
                ids = [1]
            content_tokens += len(ids)
            writer.write_doc_ids(ids)
            docs += 1
    stream_tokens = writer.finalize()
    meta = {
        "source": source,
        "domain": domain,
        "split": split,
        "tokenizer": "fake-local-smoke",
        "eos_token_id": EOS_TOKEN_ID,
        "docs": docs,
        "content_tokens": content_tokens,
        "stream_tokens_with_eos": stream_tokens,
        "tokenized_npy": str(out_npy),
    }
    (out_dir / f"{split}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Scratch dir (default: <repo>/.tmp/refhq_new_smoke)",
    )
    parser.add_argument("--n-docs", type=int, default=2000, help="Synthetic docs per source")
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    scratch = args.scratch_root or (repo_root / ".tmp" / "refhq_new_smoke")
    if scratch.exists() and not args.keep_scratch:
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    sources = ["tulu-v2", "tulu-3"]
    # 1) plan
    plan_proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "plan_refhq_new.py"),
            "--scratch-root",
            str(scratch),
            "--sources",
            *sources,
            "--seed",
            "42",
        ],
        check=False,
    )
    if plan_proc.returncode != 0:
        return plan_proc.returncode
    plan_path = scratch / "manifests" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # 2) normalize from local fixtures (no HF)
    fixtures_dir = scratch / "fixtures"
    for source in sources:
        fixture = fixtures_dir / f"{source}.jsonl"
        _write_fixture(fixture, source=source, n_docs=args.n_docs)
        src_plan = plan["sources"][source]
        docs_root = Path(src_plan["paths"]["docs"])
        if docs_root.exists():
            shutil.rmtree(docs_root)
        docs_root.mkdir(parents=True, exist_ok=True)
        stats = normalize_rows(
            source=source,
            repo_id=src_plan["hf_repo"],
            docs_root=docs_root,
            rows=_iter_fixture_rows(fixture),
        )
        print(f"normalize {source}: {json.dumps(stats['domain_counts'])}", flush=True)
        # 3) skip Dolma — copy docs → out
        n_shards = _copy_docs_to_out(docs_root, Path(src_plan["paths"]["out"]))
        print(f"english-passthrough {source}: copied {n_shards} shard(s)", flush=True)

    # 4) holdout
    summary = run_holdout(plan)
    tasks_path = scratch / "manifests" / "tokenize_tasks.txt"
    tasks = [ln for ln in tasks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(
        f"holdout pairs={len(summary['pairs'])} tokenize_tasks={len(tasks)}",
        flush=True,
    )
    if not tasks:
        print("ERROR: no tokenize tasks after holdout", flush=True)
        return 1

    # 5) fake tokenize (no HF dolma2)
    for line in tasks:
        source, domain, split = line.split()
        src_plan = plan["sources"][source]
        meta = _fake_tokenize_split(
            holdout_root=Path(src_plan["paths"]["holdout"]),
            tok_root=Path(src_plan["paths"]["tokenized"]),
            source=source,
            domain=domain,
            split=split,
        )
        print(
            f"tokenize {source}/{domain}/{split}: docs={meta['docs']} "
            f"tokens={meta['stream_tokens_with_eos']}",
            flush=True,
        )

    # 6) finalize dry-run (stage locally, no S3)
    tok_root = resolve_tok_root(scratch)
    stage_dir = scratch / "publish-stage"
    staged = stage_tokens_from_tok(
        tok_root=tok_root,
        out_root=stage_dir,
        shard_bytes=1_073_741_824,
        force=True,
    )
    manifest = build_manifest(
        scratch_root=scratch,
        tok_root=tok_root,
        bucket="edullm-datasets",
        prefix="refhq/refhq-new",
    )
    man_path = scratch / "manifests" / "tokenized_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not manifest["accepted"]:
        print(f"ACCEPTANCE FAILED: {manifest['failures']}", flush=True)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "scratch_root": str(scratch),
                "tokenize_tasks": len(tasks),
                "staged_pairs": sorted(staged),
                "total_stream_tokens_with_eos": manifest["total_stream_tokens_with_eos"],
                "by_source": manifest["by_source"],
                "stage_dir": str(stage_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
