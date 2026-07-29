#!/usr/bin/env python3
"""Build a peak-sized contiguous working pool from olmohq tokenized shards.

Reads ``tokenized_manifest.json``, selects enough shards per domain to cover
``--peak-tokens`` (or auto from a validation mixtures JSON + budget), downloads
the selected ``.npy`` files from S3, and concatenates them into

    <out-dir>/<domain>/<domain>.npy

aligned to ``seq_len`` tokens. Does not touch regmix-10b.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from mixlaw_common import DOMAINS, SEQ_LEN, allocate_sequences, load_mixtures


def peak_tokens_from_mixtures(mixtures_json: Path, budget: int) -> dict[str, int]:
    mixes = load_mixtures(mixtures_json)
    payload = json.loads(mixtures_json.read_text(encoding="utf-8"))
    reuse = {int(r["id"]) for r in payload["mixtures"] if r.get("reuse_s3")}
    mixes = [m for m in mixes if m.id not in reuse]
    total_seqs = budget // SEQ_LEN
    peak = {d: 0 for d in DOMAINS}
    for mix in mixes:
        counts = allocate_sequences(mix.weights, total_seqs)
        for d in DOMAINS:
            peak[d] = max(peak[d], counts[d] * SEQ_LEN)
    # Add 2% headroom for alignment / rounding.
    return {d: int(peak[d] * 1.02) for d in DOMAINS}


def select_shards(
    manifest: dict,
    peak: dict[str, int],
    seed: int,
) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for s in manifest["shards"]:
        d = s["domain"]
        if d in by_domain:
            by_domain[d].append(s)

    selected: dict[str, list[dict]] = {}
    for d in DOMAINS:
        need = peak[d]
        shards = by_domain[d][:]
        order = rng.permutation(len(shards))
        chosen: list[dict] = []
        got = 0
        for i in order:
            if got >= need:
                break
            s = shards[int(i)]
            chosen.append(s)
            got += int(s["tokens"])
        if got < need:
            raise SystemExit(
                f"{d}: need {need:,} tokens but pool only has "
                f"{sum(int(s['tokens']) for s in shards):,} in selected shards "
                f"(got {got:,})"
            )
        selected[d] = chosen
        print(f"{d}: need={need/1e9:.3f}B selected={got/1e9:.3f}B shards={len(chosen)}")
    return selected


def aws_cp(uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", uri, str(dest), "--only-show-errors"]
    subprocess.run(cmd, check=True)


def concat_domain(
    shards: list[dict],
    shard_dir: Path,
    out_npy: Path,
    seq_len: int,
) -> int:
    """Concatenate downloaded shard npys into one contiguous seq-aligned memmap."""
    total = sum(int(s["tokens"]) for s in shards)
    usable = (total // seq_len) * seq_len
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npy.with_suffix(".npy.tmp")
    dst = np.memmap(tmp, mode="w+", dtype=np.uint32, shape=(usable,))
    offset = 0
    for s in shards:
        local = shard_dir / Path(s["path"]).name
        if not local.is_file():
            raise SystemExit(f"missing downloaded shard {local}")
        src = np.memmap(local, mode="r", dtype=np.uint32)
        n = min(len(src), usable - offset)
        if n <= 0:
            break
        dst[offset : offset + n] = src[:n]
        offset += n
        del src
        if offset >= usable:
            break
    dst.flush()
    del dst
    tmp.replace(out_npy)
    return usable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenized-manifest", type=Path, required=True)
    ap.add_argument("--s3-tokenized-prefix", required=True,
                    help="e.g. s3://edullm-datasets/olmo100b/olmo-mix-1124-30b/tokenized")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mixtures-json", type=Path, default=None)
    ap.add_argument("--budget-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--peak-json", type=Path, default=None, help="Optional {domain: tokens}")
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-concat", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.tokenized_manifest.read_text(encoding="utf-8"))
    if args.peak_json:
        peak = {d: int(v) for d, v in json.loads(args.peak_json.read_text()).items()}
    elif args.mixtures_json:
        peak = peak_tokens_from_mixtures(args.mixtures_json, args.budget_tokens)
    else:
        raise SystemExit("provide --mixtures-json or --peak-json")

    plan_dir = args.out_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "peak_tokens.json").write_text(json.dumps(peak, indent=2) + "\n")
    selected = select_shards(manifest, peak, args.seed)
    (plan_dir / "selected_shards.json").write_text(
        json.dumps({d: [{"path": s["path"], "tokens": s["tokens"]} for s in rows]
                    for d, rows in selected.items()}, indent=2)
        + "\n"
    )

    shard_dir = args.out_dir / "shards"
    prefix = args.s3_tokenized_prefix.rstrip("/")
    if not args.skip_download:
        for d, rows in selected.items():
            for s in rows:
                rel = s["path"]  # shards/....npy
                uri = f"{prefix}/{rel}"
                dest = shard_dir / Path(rel).name
                if dest.is_file() and dest.stat().st_size == int(s["bytes"]):
                    continue
                print(f"download {uri}", flush=True)
                aws_cp(uri, dest)

    tok_dir = args.out_dir / "tokenized"
    if not args.skip_concat:
        for d, rows in selected.items():
            out = tok_dir / d / f"{d}.npy"
            n = concat_domain(rows, shard_dir, out, SEQ_LEN)
            meta = {"domain": d, "tokens": n, "seq_len": SEQ_LEN, "n_shards": len(rows)}
            (tok_dir / d / f"{d}.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"wrote {out} ({n/1e9:.3f}B tokens)", flush=True)

    print("working pool ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
