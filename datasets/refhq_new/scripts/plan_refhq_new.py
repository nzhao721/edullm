#!/usr/bin/env python3
"""Write refhq-new plan manifests for FarmShare (6 instruct sources)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from _bootstrap import setup_paths  # noqa: E402

setup_paths()

from refhq_new.domain_map import DOMAINS, SOURCES  # noqa: E402
from refhq_new.exclusion import load_exclusion_rules  # noqa: E402
from refhq_new_sources import (  # noqa: E402
    DEFAULT_S3_BUCKET,
    DEFAULT_S3_PREFIX,
    DEFAULT_SCRATCH_ROOT,
    DEFAULT_SEED,
    EOS_TOKEN_ID,
    HOLDOUT_FRACTION,
    SPLITS,
    TOKENIZER_ID,
    scratch_layout,
    source_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Subset of sources to plan (default: all six)",
    )
    parser.add_argument(
        "--source-list",
        default=None,
        help="Whitespace-separated source list (submit_refhq_new.sh contract)",
    )
    args = parser.parse_args()

    if args.source_list:
        selected = args.source_list.split()
    elif args.sources is not None:
        selected = args.sources
    else:
        selected = list(SOURCES)

    if not (0.0 < args.holdout_fraction < 0.5):
        raise SystemExit(f"holdout fraction must be in (0, 0.5); got {args.holdout_fraction}")

    unknown = sorted(set(selected) - set(SOURCES))
    if unknown:
        raise SystemExit(f"unknown sources: {unknown}")
    sources = [s for s in SOURCES if s in set(selected)]

    layout = scratch_layout(args.scratch_root)
    for key in ("root", "raw", "docs", "out", "holdout", "work", "tokenized", "manifests", "logs"):
        layout[key].mkdir(parents=True, exist_ok=True)

    rules = load_exclusion_rules()
    specs = source_specs(rules)

    plan: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scratch_root": str(args.scratch_root),
        "tokenizer_id": args.tokenizer,
        "eos_token_id": args.eos_token_id,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "holdout_note": (
            "Reserve holdout_fraction of documents per (source, domain) BEFORE tokenize "
            "(seed above). Uniform fraction keeps mix weights."
        ),
        "s3_bucket": args.s3_bucket,
        "s3_prefix": args.s3_prefix,
        "domains": list(DOMAINS),
        "splits": list(SPLITS),
        "sources": {},
        "layout": {
            "docs": "docs/<source>/<domain>/documents-*.jsonl.gz",
            "out": "out/<source>/<domain>/documents/documents-*.jsonl.gz  # after Dolma English",
            "holdout": "holdout/<source>/<domain>/{train,val}/documents-*.jsonl.gz",
            "tokenized_npy": "tokenized/<source>/<domain>/<split>.npy",
            "tokens_publish": "tokens/<source>/<domain>/<split>-NNNNN.u32le.bin",
        },
    }

    for source in sources:
        src = specs[source]
        plan["sources"][source] = {
            "hf_repo": src["hf_repo"],
            "repo_type": src["repo_type"],
            "split": src["split"],
            "multi_config": src["multi_config"],
            "gated": src["gated"],
            "paths": {
                "raw": str(layout["raw"] / source),
                "docs": str(layout["docs"] / source),
                "out": str(layout["out"] / source),
                "holdout": str(layout["holdout"] / source),
                "work": str(layout["work"] / source),
                "tokenized": str(layout["tokenized"] / source),
                "stats": str(layout["manifests"] / f"{source}-stats.json"),
            },
            "seed": args.seed + list(SOURCES).index(source),
        }
        for key in ("raw", "docs", "out", "holdout", "work", "tokenized"):
            Path(plan["sources"][source]["paths"][key]).mkdir(parents=True, exist_ok=True)

    plan_path = layout["manifests"] / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    source_list = layout["manifests"] / "sources.txt"
    source_list.write_text("\n".join(sources) + "\n", encoding="utf-8")

    # Placeholder tokenize tasks (source domain split); holdout refreshes with real pairs.
    task_lines: list[str] = []
    for source in sources:
        for domain in DOMAINS:
            for split in SPLITS:
                task_lines.append(f"{source} {domain} {split}")
    tasks_path = layout["manifests"] / "tokenize_tasks.txt"
    tasks_path.write_text("\n".join(task_lines) + "\n", encoding="utf-8")

    print(f"wrote {plan_path}", flush=True)
    print(f"wrote {source_list}", flush=True)
    print(f"wrote {tasks_path} ({len(task_lines)} placeholder tasks)", flush=True)
    print(
        f"sources={len(sources)} holdout_fraction={args.holdout_fraction} "
        f"tokenizer={args.tokenizer} eos={args.eos_token_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
