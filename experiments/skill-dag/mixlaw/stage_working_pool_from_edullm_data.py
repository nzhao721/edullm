#!/usr/bin/env python3
"""Stage a peak-sized working pool from published ``edullm-data`` for DataDecide-60M.

Resolves ``pretrain/olmo-127b`` (or ``--dataset-id``) via ``resolve_latest`` /
``dataset_paths`` (and optionally ``build_mixture`` for whole-shard selection),
downloads train shards for each domain's peak demand, and concatenates them into

    <out-dir>/tokenized/<domain>/<domain>.npy

plus ``<out-dir>/edullm_data_source.json`` provenance so training refuses orphan
scratch/local pools. Does **not** read ``s3://edullm-datasets/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from mixlaw_common import (
    DOMAINS,
    EDULLM_DATA_DATASET_ID,
    EDULLM_DATA_SOURCE_LABEL,
    POOL_PROVENANCE_NAME,
    SEQ_LEN,
    allocate_sequences,
    load_mixtures,
)


def peak_tokens_from_mixtures(mixtures_json: Path, budget: int) -> dict[str, int]:
    mixes = load_mixtures(mixtures_json)
    total_seqs = budget // SEQ_LEN
    peak = {d: 0 for d in DOMAINS}
    for mix in mixes:
        counts = allocate_sequences(mix.weights, total_seqs)
        for d in DOMAINS:
            peak[d] = max(peak[d], counts[d] * SEQ_LEN)
    return {d: int(peak[d] * 1.02) for d in DOMAINS}


def peak_tokens_from_recipe_files(mixture_paths: list[Path], budget: int) -> dict[str, int]:
    peak = {d: 0 for d in DOMAINS}
    for path in mixture_paths:
        part = peak_tokens_from_mixtures(path, budget)
        for d in DOMAINS:
            peak[d] = max(peak[d], part[d])
    return peak


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise SystemExit(f"not an s3 uri: {uri}")
    return p.netloc, p.path.lstrip("/")


def aws_cp(uri: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", uri, str(dest), "--only-show-errors"]
    subprocess.run(cmd, check=True)


def resolve_domain_shards(
    *,
    dataset_id: str,
    version: str,
    domain: str,
    need_tokens: int,
    seed: int,
    s3: Any,
) -> dict[str, Any]:
    """Pick enough whole train shards for ``need_tokens`` from one source label."""
    from edullm_data.read import MixtureSource, build_mixture, dataset_paths

    # Confirm the labelled split exists and is validated before mixture selection.
    probe = dataset_paths(
        dataset_id,
        version,
        split="train",
        s3=s3,
        labels={EDULLM_DATA_SOURCE_LABEL: domain},
    )
    if not probe.paths:
        raise SystemExit(
            f"{dataset_id}/{version}: no train shards for "
            f"labels.{{{EDULLM_DATA_SOURCE_LABEL}={domain!r}}}"
        )
    if probe.dtype and probe.dtype not in ("uint32", "u32", "<u4", ">u4"):
        raise SystemExit(
            f"{dataset_id}/{version} source={domain}: unexpected dtype {probe.dtype!r} "
            "(expected uint32)"
        )

    mix = build_mixture(
        dataset_id,
        version,
        sources=[MixtureSource(labels={EDULLM_DATA_SOURCE_LABEL: domain}, ratio=1.0)],
        total=int(need_tokens),
        seed=int(seed),
        s3=s3,
        split="train",
    )
    shortfall = int(mix.shortfall.get(f"{EDULLM_DATA_SOURCE_LABEL}={domain}", 0) or 0)
    if shortfall > 0:
        raise SystemExit(
            f"{domain}: need {need_tokens:,} tokens but mixture shortfall={shortfall:,} "
            f"(actual={mix.counts_by_source})"
        )
    return {
        "domain": domain,
        "need_tokens": need_tokens,
        "paths": list(mix.paths),
        "dtype": mix.dtype or probe.dtype or "uint32",
        "header_bytes": int(mix.header_bytes or 0),
        "selected_tokens": int(
            mix.counts_by_source.get(f"{EDULLM_DATA_SOURCE_LABEL}={domain}", mix.total)
        ),
        "byte_order": mix.byte_order,
    }


def concat_shards(
    shards: list[Path],
    out_npy: Path,
    *,
    seq_len: int,
    header_bytes: int,
) -> int:
    """Concatenate uint32 token shards into one seq-aligned memmap (.npy name, raw payload)."""
    total_tokens = 0
    for p in shards:
        size = p.stat().st_size
        if size < header_bytes:
            raise SystemExit(f"{p}: smaller than header_bytes={header_bytes}")
        payload = size - header_bytes
        if payload % 4 != 0:
            raise SystemExit(f"{p}: payload {payload} not uint32-aligned")
        total_tokens += payload // 4
    usable = (total_tokens // seq_len) * seq_len
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npy.with_suffix(".npy.tmp")
    dst = np.memmap(tmp, mode="w+", dtype=np.uint32, shape=(usable,))
    offset = 0
    for p in shards:
        raw = np.memmap(p, mode="r", dtype=np.uint8)
        body = raw[header_bytes:]
        if body.nbytes % 4 != 0:
            raise SystemExit(f"{p}: body not uint32-aligned after header skip")
        src = np.frombuffer(body, dtype=np.uint32)
        n = min(len(src), usable - offset)
        if n <= 0:
            break
        dst[offset : offset + n] = src[:n]
        offset += n
        del src, body, raw
        if offset >= usable:
            break
    dst.flush()
    del dst
    tmp.replace(out_npy)
    return usable


def write_provenance(out_dir: Path, payload: dict[str, Any]) -> Path:
    path = out_dir / POOL_PROVENANCE_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-id", default=EDULLM_DATA_DATASET_ID)
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin a version (default: resolve_latest)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mixtures-json", type=Path, default=None)
    ap.add_argument("--extra-mixtures-json", type=Path, nargs="*", default=())
    ap.add_argument("--budget-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--peak-json", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-concat", action="store_true")
    args = ap.parse_args()

    try:
        from edullm_data.read import resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:
        raise SystemExit(
            "edullm-data package is required "
            '(install: uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0")'
        ) from exc

    s3 = Boto3S3.default()
    version = args.dataset_version or resolve_latest(args.dataset_id, s3=s3)
    if not version:
        raise SystemExit(f"no published versions for {args.dataset_id} in edullm-data catalog")

    if args.peak_json:
        peak = {d: int(v) for d, v in json.loads(args.peak_json.read_text()).items()}
    elif args.mixtures_json:
        paths = [args.mixtures_json, *list(args.extra_mixtures_json)]
        peak = peak_tokens_from_recipe_files(paths, args.budget_tokens)
    else:
        raise SystemExit("provide --mixtures-json or --peak-json")

    plan_dir = args.out_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "peak_tokens.json").write_text(json.dumps(peak, indent=2) + "\n")

    selections: dict[str, dict[str, Any]] = {}
    for i, domain in enumerate(DOMAINS):
        need = int(peak.get(domain, 0))
        if need <= 0:
            raise SystemExit(f"{domain}: peak demand is 0")
        sel = resolve_domain_shards(
            dataset_id=args.dataset_id,
            version=version,
            domain=domain,
            need_tokens=need,
            seed=args.seed + i,
            s3=s3,
        )
        selections[domain] = sel
        print(
            f"{domain}: need={need/1e9:.3f}B selected={sel['selected_tokens']/1e9:.3f}B "
            f"shards={len(sel['paths'])}",
            flush=True,
        )

    (plan_dir / "selected_shards.json").write_text(
        json.dumps(
            {
                d: {"tokens": s["selected_tokens"], "paths": s["paths"]}
                for d, s in selections.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    shard_dir = args.out_dir / "shards"
    local_by_domain: dict[str, list[Path]] = {}
    if not args.skip_download:
        for domain, sel in selections.items():
            local_by_domain[domain] = []
            for uri in sel["paths"]:
                _, key = _parse_s3_uri(uri)
                dest = shard_dir / domain / Path(key).name
                if not dest.is_file():
                    print(f"download {uri}", flush=True)
                    aws_cp(uri, dest)
                local_by_domain[domain].append(dest)
    else:
        for domain, sel in selections.items():
            local_by_domain[domain] = [
                shard_dir / domain / Path(_parse_s3_uri(uri)[1]).name for uri in sel["paths"]
            ]

    tok_dir = args.out_dir / "tokenized"
    domain_meta: dict[str, Any] = {}
    if not args.skip_concat:
        for domain, sel in selections.items():
            out = tok_dir / domain / f"{domain}.npy"
            n = concat_shards(
                local_by_domain[domain],
                out,
                seq_len=SEQ_LEN,
                header_bytes=int(sel["header_bytes"]),
            )
            meta = {
                "domain": domain,
                "tokens": n,
                "seq_len": SEQ_LEN,
                "n_shards": len(sel["paths"]),
                "dtype": sel["dtype"],
            }
            (tok_dir / domain / f"{domain}.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
            domain_meta[domain] = meta
            print(f"wrote {out} ({n/1e9:.3f}B tokens)", flush=True)

    for sel in selections.values():
        for uri in sel["paths"]:
            if "edullm-datasets" in uri:
                raise SystemExit(f"refusing legacy edullm-datasets uri: {uri}")

    provenance = {
        "dataset_id": args.dataset_id,
        "dataset_version": version,
        "data_bucket": "edullm-data",
        "edullm_data_uri": f"s3://edullm-data/{args.dataset_id}/{version}/",
        "label_key": EDULLM_DATA_SOURCE_LABEL,
        "domains": list(DOMAINS),
        "budget_tokens": int(args.budget_tokens),
        "seed": int(args.seed),
        "peak_tokens": peak,
        "domain_meta": domain_meta,
        "selected": {
            d: {"tokens": s["selected_tokens"], "n_shards": len(s["paths"])}
            for d, s in selections.items()
        },
        "ephemeral_scratch": True,
    }
    prov_path = write_provenance(args.out_dir, provenance)
    print(f"wrote {prov_path}", flush=True)
    print("working pool ready (edullm-data)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
