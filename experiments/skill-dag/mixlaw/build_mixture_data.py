#!/usr/bin/env python3
"""DEPRECATED / DO-NOT-USE for new training runs.

Exact per-mix slice materialization is **not a supported training path**.
All new mixlaw pilot and 370M work must use ``DomainMixtureStream`` over a
peak-sized pool staged from ``edullm-data`` (see ``prepare_*_data.py``,
``domain_stream.py``, ``olmo_domain_stream_patch.py``). Do not re-materialize
slices to bit-match checked-in ``pilot_runs/`` curves.

This module remains only for historical reference and
``preflight_checks.py`` helpers (e.g. ``_blocks_for_domain``).

---
Historical behavior (do not use for new runs):

Plan and materialize the per-mixture training data for the 24 mixing-law probes.

Source memmaps are the *working pool* built by ``tokenize_working_pool.py`` from
``s3://edullm-datasets/olmo100b/olmo-mix-1124-30b`` raw shards
(``tokenized/<domain>/<domain>.npy``). For each mixture this script draws a
*random* subsample of that domain stream and writes it to its own memmap, sized
so that

    tokens_written(domain) / tokens_written(all domains) == mixture weight

to sequence granularity. Because OLMo's ``MemMapDataset`` indexes the
concatenation of all training paths as non-overlapping ``seq_len`` chunks and the
sampler shuffles that index, one epoch over these slices *is* the mixture: no
sampling weights, no repeats, and no run-to-run drift in what "37.5% dclm" means.

Randomness is drawn in contiguous blocks (default 256 sequences = 524k tokens)
rather than per sequence. Blocks are chosen without replacement within a mixture,
which keeps reads sequential — the copy is bandwidth-bound, not seek-bound —
while still spreading each run across the whole domain. Blocks are *not*
disjoint across mixtures: the 24 runs together need more wiki and pes2o tokens
than exist, and since each run trains an independent model, overlap between runs
is harmless.

Subcommands:
    plan   inspect the domain memmaps and write slice_plan.json (no data copied)
    build  materialize the slices named by slice_plan.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from mixlaw_common import (
    DOMAINS,
    MEMMAP_DTYPE,
    SEQ_LEN,
    DEFAULT_TOKENS_PER_PARAM,
    allocate_sequences,
    domain_npy_name,
    load_mixtures,
    memmap_tokens,
    realized_weights,
    token_budget,
    token_budget_fixed,
)

PLAN_NAME = "slice_plan.json"


def _blocks_for_domain(
    rng: np.random.Generator,
    n_seqs_available: int,
    n_seqs_needed: int,
    block_seqs: int,
) -> list[tuple[int, int]]:
    """Choose (start_seq, n_seqs) blocks totalling exactly ``n_seqs_needed``.

    Blocks tile the domain and are drawn without replacement, so no sequence is
    used twice within one mixture.
    """
    if n_seqs_needed <= 0:
        return []
    if n_seqs_needed > n_seqs_available:
        raise SystemExit(
            f"need {n_seqs_needed} sequences but only {n_seqs_available} available"
        )

    n_blocks_total = n_seqs_available // block_seqs
    if n_blocks_total == 0:
        # Domain smaller than one block: take a single random contiguous run.
        start = int(rng.integers(0, n_seqs_available - n_seqs_needed + 1))
        return [(start, n_seqs_needed)]

    n_blocks_needed = min(n_blocks_total, -(-n_seqs_needed // block_seqs))
    chosen = rng.choice(n_blocks_total, size=n_blocks_needed, replace=False)
    chosen.sort()  # sequential read order

    blocks: list[tuple[int, int]] = []
    remaining = n_seqs_needed
    for b in chosen:
        take = min(block_seqs, remaining)
        blocks.append((int(b) * block_seqs, int(take)))
        remaining -= take
        if remaining == 0:
            break

    if remaining > 0:
        # Only reachable when the block grid leaves a ragged tail; top up from it.
        tail_start = n_blocks_total * block_seqs
        tail_avail = n_seqs_available - tail_start
        if tail_avail < remaining:
            raise SystemExit("block plan could not satisfy sequence budget")
        blocks.append((tail_start, remaining))
    return blocks


def resolve_mixtures_json(explicit: Path | None) -> Path | None:
    """Resolve mixtures JSON: CLI flag, then SKILLIT_PROBES_JSON / MIXTURES_JSON env."""
    if explicit is not None:
        return explicit
    for key in ("SKILLIT_PROBES_JSON", "MIXTURES_JSON"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw)
    return None


def cmd_plan(args: argparse.Namespace) -> None:
    args.mixtures_json = resolve_mixtures_json(args.mixtures_json)
    mixtures = load_mixtures(args.mixtures_json)
    # Skip mixtures that reuse an existing S3 corpus (e.g. mix01 → regmix-10b).
    if args.mixtures_json is not None:
        payload = json.loads(args.mixtures_json.read_text(encoding="utf-8"))
        reuse = {
            int(r["id"])
            for r in payload["mixtures"]
            if r.get("reuse_s3")
        }
        mixtures = [m for m in mixtures if m.id not in reuse]

    if args.total_tokens is not None:
        total_seqs, total_steps, total_tokens = token_budget_fixed(args.total_tokens)
        tpp = None
    else:
        tpp = args.tokens_per_param if args.tokens_per_param is not None else DEFAULT_TOKENS_PER_PARAM
        total_seqs, total_steps, total_tokens = token_budget(tpp)

    available: dict[str, int] = {}
    for domain in DOMAINS:
        src = args.tokenized_dir / domain / domain_npy_name(domain)
        if not src.is_file():
            raise SystemExit(f"missing domain memmap: {src}")
        available[domain] = memmap_tokens(src) // SEQ_LEN

    plan_mixtures = []
    for mix in mixtures:
        counts = allocate_sequences(mix.weights, total_seqs)
        rng = np.random.default_rng([args.seed, mix.id])

        domain_plan = {}
        for domain in DOMAINS:
            need = counts[domain]
            if need > available[domain]:
                raise SystemExit(
                    f"mix {mix.id} needs {need} {domain} sequences, "
                    f"only {available[domain]} available "
                    f"(lower --tokens-per-param/--total-tokens or raise the domain budget)"
                )
            domain_plan[domain] = {
                "seqs": need,
                "tokens": need * SEQ_LEN,
                "blocks": _blocks_for_domain(rng, available[domain], need, args.block_seqs),
            }

        plan_mixtures.append(
            {
                "id": mix.id,
                "tag": mix.tag,
                "run_name": mix.run_name,
                "target_weights": mix.weights,
                "realized_weights": realized_weights(counts),
                "max_weight_error": max(
                    abs(realized_weights(counts)[d] - mix.weights[d]) for d in DOMAINS
                ),
                "domains": domain_plan,
            }
        )

    plan = {
        "seed": args.seed,
        "seq_len": SEQ_LEN,
        "dtype": MEMMAP_DTYPE,
        "block_seqs": args.block_seqs,
        "tokens_per_param": tpp,
        "total_tokens_requested": args.total_tokens,
        "total_seqs_per_mix": total_seqs,
        "total_steps_per_mix": total_steps,
        "total_tokens_per_mix": total_tokens,
        "tokenized_dir": str(args.tokenized_dir),
        "mixtures_json": str(args.mixtures_json) if args.mixtures_json else None,
        "domain_seqs_available": available,
        "mixtures": plan_mixtures,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / PLAN_NAME
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    worst = max(m["max_weight_error"] for m in plan_mixtures)
    gib = total_tokens * 4 / 2**30
    print(f"wrote {out}")
    print(
        f"{len(plan_mixtures)} mixtures | {total_tokens:,} tokens each "
        f"({gib:.2f} GiB, {total_steps:,} steps) | {len(plan_mixtures) * gib:.1f} GiB total"
    )
    print(f"worst realized-vs-target weight error: {worst:.2e}")


def _build_one(
    tokenized_dir: str,
    out_dir: str,
    mix: dict,
    seq_len: int,
) -> tuple[int, int]:
    """Materialize one mixture's slices. Returns (mix_id, tokens_written)."""
    mix_dir = Path(out_dir) / mix["run_name"]
    mix_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    written = 0
    for domain, spec in mix["domains"].items():
        if spec["seqs"] <= 0:
            continue
        src = Path(tokenized_dir) / domain / f"{domain}.npy"
        src_mm = np.memmap(src, mode="r", dtype=np.uint32)

        dst = mix_dir / f"{domain}.npy"
        tmp = dst.with_suffix(".npy.tmp")
        dst_mm = np.memmap(tmp, mode="w+", dtype=np.uint32, shape=(spec["tokens"],))

        offset = 0
        for start_seq, n_seqs in spec["blocks"]:
            lo = start_seq * seq_len
            n = n_seqs * seq_len
            dst_mm[offset : offset + n] = src_mm[lo : lo + n]
            offset += n
        if offset != spec["tokens"]:
            raise SystemExit(f"mix {mix['id']} {domain}: wrote {offset} of {spec['tokens']}")

        dst_mm.flush()
        del dst_mm
        del src_mm
        os.replace(tmp, dst)

        paths.append(str(dst))
        written += spec["tokens"]

    (mix_dir / "paths_train.txt").write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
    (mix_dir / "mix_meta.json").write_text(
        json.dumps(
            {
                "id": mix["id"],
                "tag": mix["tag"],
                "run_name": mix["run_name"],
                "target_weights": mix["target_weights"],
                "realized_weights": mix["realized_weights"],
                "tokens": written,
                "seqs": written // seq_len,
                "paths": sorted(paths),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(mix["id"]), written


def cmd_build(args: argparse.Namespace) -> None:
    plan = json.loads((args.plan_dir / PLAN_NAME).read_text(encoding="utf-8"))
    tokenized_dir = args.tokenized_dir or Path(plan["tokenized_dir"])

    wanted = set(args.mix_ids) if args.mix_ids else None
    todo = [m for m in plan["mixtures"] if wanted is None or m["id"] in wanted]
    if not todo:
        raise SystemExit("no mixtures selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = int(plan["seq_len"])

    if args.workers <= 1:
        for mix in todo:
            mix_id, written = _build_one(str(tokenized_dir), str(args.out_dir), mix, seq_len)
            print(f"mix{mix_id:02d}: {written:,} tokens")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_build_one, str(tokenized_dir), str(args.out_dir), mix, seq_len): mix["id"]
            for mix in todo
        }
        for fut in as_completed(futures):
            mix_id, written = fut.result()
            print(f"mix{mix_id:02d}: {written:,} tokens")


def main() -> None:
    import sys

    print(
        "WARNING: build_mixture_data.py is DEPRECATED / DO-NOT-USE for new "
        "training. Use DomainMixtureStream + edullm-data peak pool instead.",
        file=sys.stderr,
    )
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("plan", help="write slice_plan.json")
    p.set_defaults(func=cmd_plan)
    p.add_argument("--tokenized-dir", type=Path, required=True, help="Local working-pool/tokenized")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--mixtures-json",
        type=Path,
        default=None,
        help=(
            "Override mixtures.json (e.g. ../skillit/probes.json or "
            "validation_mixtures_10b.json). Also reads env SKILLIT_PROBES_JSON "
            "or MIXTURES_JSON when unset."
        ),
    )
    p.add_argument(
        "--total-tokens",
        type=int,
        default=None,
        help="Fixed token budget per mixture (e.g. 10000000000 for 370M validation).",
    )
    p.add_argument(
        "--tokens-per-param",
        type=float,
        default=None,
        help=(
            "Token budget per mixture as a multiple of the 57.1M DataDecide 60M size. "
            "Default 5 (~285M tokens) targets ≈12 B200 GPU-hours for all 24 mixtures "
            "including final task-loss eval. Ignored when --total-tokens is set."
        ),
    )
    p.add_argument(
        "--block-seqs",
        type=int,
        default=256,
        help="Contiguous sequences per random block (256 = 524k tokens)",
    )
    p.add_argument("--seed", type=int, default=6198)

    b = sub.add_parser("build", help="materialize slices from slice_plan.json")
    b.set_defaults(func=cmd_build)
    b.add_argument("--plan-dir", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--tokenized-dir", type=Path, default=None, help="Override plan's source dir")
    b.add_argument("--mix-ids", type=int, nargs="*", default=None, help="Subset of mixture ids")
    b.add_argument("--workers", type=int, default=8, help="Mixtures materialized in parallel")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
