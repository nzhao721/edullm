#!/usr/bin/env python3
"""Stage a peak-sized mixlaw validation working pool from published edullm-data.

Resolves ``pretrain/olmo-127b`` (or ``--dataset-id``) via ``edullm_data.read``
(``resolve_latest`` + ``dataset_paths`` with ``labels={"source": <domain>}``),
selects enough train shards per domain for the validation recipe peak demand,
downloads them, and concatenates seq-aligned uint32 memmaps under::

    <out-dir>/tokenized/<domain>/<domain>.u32le.bin

Does **not** read ``s3://edullm-datasets/`` or assume a pre-existing FarmShare /
laptop pool, ladder-run venv, or leftover scratch checkpoints. Safe on
ephemeral empty scratch: stages only from published ``edullm-data``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from mixlaw_common import DOMAINS, SEQ_LEN, allocate_sequences, load_mixtures

DEFAULT_DATASET_ID = "pretrain/olmo-127b"
SOURCE_LABEL_KEY = "source"
BYTES_PER_TOKEN = 4

_S3_URI = re.compile(r"^s3://([^/]+)/(.+)$")


def peak_tokens_from_mixtures(mixtures_json: Path, budget: int) -> dict[str, int]:
    mixes = load_mixtures(mixtures_json)
    total_seqs = budget // SEQ_LEN
    peak = {d: 0 for d in DOMAINS}
    for mix in mixes:
        counts = allocate_sequences(mix.weights, total_seqs)
        for d in DOMAINS:
            peak[d] = max(peak[d], counts[d] * SEQ_LEN)
    return {d: int(peak[d] * 1.02) for d in DOMAINS}


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    m = _S3_URI.match(uri.strip())
    if not m:
        raise SystemExit(f"not an s3 URI: {uri!r}")
    return m.group(1), m.group(2)


def _token_count(entry_bytes: int, *, header_bytes: int = 0) -> int:
    usable = max(0, int(entry_bytes) - int(header_bytes))
    if usable % BYTES_PER_TOKEN != 0:
        raise SystemExit(f"shard size {entry_bytes} not uint32-aligned after header={header_bytes}")
    return usable // BYTES_PER_TOKEN


def resolve_dataset(
    dataset_id: str,
    version: str | None,
    *,
    s3: Any,
) -> tuple[str, str]:
    from edullm_data.read import resolve_latest

    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(f"no published version for {dataset_id} in edullm-data")
    return dataset_id, ver


def domain_train_shards(
    dataset_id: str,
    version: str,
    domain: str,
    *,
    s3: Any,
) -> dict[str, Any]:
    """Validated train shard URIs + dtype for one mixlaw domain (``labels.source``)."""
    from edullm_data.read import dataset_paths

    resolved = dataset_paths(
        dataset_id,
        version,
        split="train",
        s3=s3,
        labels={SOURCE_LABEL_KEY: domain},
    )
    if not resolved.paths:
        raise SystemExit(
            f"{dataset_id}/{version}: no train shards for {SOURCE_LABEL_KEY}={domain!r}"
        )
    if resolved.dtype not in (None, "uint32"):
        raise SystemExit(
            f"{dataset_id}/{version} source={domain}: expected dtype uint32, got {resolved.dtype!r}"
        )
    header_bytes = int(getattr(resolved, "header_bytes", 0) or 0)
    shards: list[dict[str, Any]] = []
    for uri in resolved.paths:
        bucket, key = _parse_s3_uri(uri)
        meta = s3.head(bucket, key)
        nbytes = int(meta["size"])
        shards.append(
            {
                "uri": uri,
                "bucket": bucket,
                "key": key,
                "bytes": nbytes,
                "tokens": _token_count(nbytes, header_bytes=header_bytes),
                "name": Path(urlparse(uri).path).name,
            }
        )
    return {
        "domain": domain,
        "dtype": resolved.dtype or "uint32",
        "byte_order": getattr(resolved, "byte_order", "little"),
        "header_bytes": header_bytes,
        "rows": resolved.rows,
        "shards": shards,
    }


def select_shards(
    inventory: dict[str, dict[str, Any]],
    peak: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for d in DOMAINS:
        need = int(peak[d])
        shards = list(inventory[d]["shards"])
        order = rng.permutation(len(shards))
        chosen: list[dict[str, Any]] = []
        got = 0
        for i in order:
            if got >= need:
                break
            s = shards[int(i)]
            chosen.append(s)
            got += int(s["tokens"])
        if got < need:
            raise SystemExit(
                f"{d}: need {need:,} tokens but edullm-data only has "
                f"{sum(int(s['tokens']) for s in shards):,} (got {got:,})"
            )
        selected[d] = chosen
        print(
            f"{d}: need={need / 1e9:.3f}B selected={got / 1e9:.3f}B shards={len(chosen)}",
            flush=True,
        )
    return selected


def download_shard(s3: Any, shard: dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == int(shard["bytes"]):
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    # Stream via boto3 client underlying Boto3S3 when available; else range-get loop.
    client = getattr(s3, "_c", None)
    if client is not None:
        client.download_file(shard["bucket"], shard["key"], str(tmp))
    else:
        body = s3.get(shard["bucket"], shard["key"])
        tmp.write_bytes(body)
    if tmp.stat().st_size != int(shard["bytes"]):
        raise SystemExit(
            f"download size mismatch for {shard['uri']}: "
            f"{tmp.stat().st_size} != {shard['bytes']}"
        )
    tmp.replace(dest)


def concat_domain(
    shards: list[dict[str, Any]],
    shard_dir: Path,
    out_bin: Path,
    seq_len: int,
    *,
    header_bytes: int = 0,
) -> int:
    total = sum(int(s["tokens"]) for s in shards)
    usable = (total // seq_len) * seq_len
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_bin.with_suffix(out_bin.suffix + ".tmp")
    dst = np.memmap(tmp, mode="w+", dtype=np.uint32, shape=(usable,))
    offset = 0
    hdr = int(header_bytes)
    for s in shards:
        local = shard_dir / s["name"]
        if not local.is_file():
            raise SystemExit(f"missing downloaded shard {local}")
        nbytes = local.stat().st_size - hdr
        if nbytes < 0 or nbytes % BYTES_PER_TOKEN != 0:
            raise SystemExit(f"{local}: not uint32-aligned after header={hdr}")
        n_tok = nbytes // BYTES_PER_TOKEN
        src = np.memmap(local, mode="r", dtype=np.uint32, offset=hdr, shape=(n_tok,))
        n = min(n_tok, usable - offset)
        if n <= 0:
            break
        dst[offset : offset + n] = src[:n]
        offset += n
        del src
        if offset >= usable:
            break
    dst.flush()
    del dst
    tmp.replace(out_bin)
    return usable


def pool_is_ready(pool_dir: Path, domains: tuple[str, ...] = DOMAINS) -> bool:
    """True when every domain memmap exists under the working-pool layout."""
    for d in domains:
        candidates = [
            pool_dir / "tokenized" / d / f"{d}.u32le.bin",
            pool_dir / "tokenized" / d / f"{d}.npy",
            pool_dir / d / f"{d}.u32le.bin",
            pool_dir / d / f"{d}.npy",
            pool_dir / f"{d}.u32le.bin",
            pool_dir / f"{d}.npy",
        ]
        if not any(p.is_file() and p.stat().st_size >= SEQ_LEN * BYTES_PER_TOKEN for p in candidates):
            return False
    return True


def stage_pool(
    *,
    out_dir: Path,
    mixtures_json: Path,
    budget_tokens: int,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_version: str | None = None,
    seed: int = 6198,
    skip_download: bool = False,
    skip_concat: bool = False,
    s3: Any | None = None,
) -> dict[str, Any]:
    from edullm_data.s3 import Boto3S3

    s3 = s3 or Boto3S3.default()
    dataset_id, version = resolve_dataset(dataset_id, dataset_version, s3=s3)
    peak = peak_tokens_from_mixtures(mixtures_json, budget_tokens)

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_dir = out_dir / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)

    inventory: dict[str, dict[str, Any]] = {}
    for d in DOMAINS:
        print(f"resolve {dataset_id}/{version} source={d}", flush=True)
        inventory[d] = domain_train_shards(dataset_id, version, d, s3=s3)

    (plan_dir / "peak_tokens.json").write_text(json.dumps(peak, indent=2) + "\n")
    (plan_dir / "dataset_resolve.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "version": version,
                "label_key": SOURCE_LABEL_KEY,
                "domains": {
                    d: {
                        "rows": inventory[d]["rows"],
                        "dtype": inventory[d]["dtype"],
                        "header_bytes": inventory[d]["header_bytes"],
                        "n_shards": len(inventory[d]["shards"]),
                    }
                    for d in DOMAINS
                },
            },
            indent=2,
        )
        + "\n"
    )

    selected = select_shards(inventory, peak, seed)
    (plan_dir / "selected_shards.json").write_text(
        json.dumps(
            {
                d: [{"uri": s["uri"], "tokens": s["tokens"], "bytes": s["bytes"]} for s in rows]
                for d, rows in selected.items()
            },
            indent=2,
        )
        + "\n"
    )

    shard_dir = out_dir / "shards"
    if not skip_download:
        for d, rows in selected.items():
            for s in rows:
                dest = shard_dir / s["name"]
                print(f"download {s['uri']}", flush=True)
                download_shard(s3, s, dest)

    tok_dir = out_dir / "tokenized"
    written: dict[str, int] = {}
    if not skip_concat:
        for d, rows in selected.items():
            out = tok_dir / d / f"{d}.u32le.bin"
            header = int(inventory[d]["header_bytes"])
            n = concat_domain(rows, shard_dir, out, SEQ_LEN, header_bytes=header)
            written[d] = n
            meta = {
                "domain": d,
                "tokens": n,
                "seq_len": SEQ_LEN,
                "n_shards": len(rows),
                "dataset_id": dataset_id,
                "version": version,
                "dtype": inventory[d]["dtype"],
            }
            (tok_dir / d / f"{d}.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"wrote {out} ({n / 1e9:.3f}B tokens)", flush=True)

    summary = {
        "dataset_id": dataset_id,
        "version": version,
        "mixtures_json": str(mixtures_json.resolve()),
        "budget_tokens": budget_tokens,
        "peak_tokens": peak,
        "domain_tokens": written,
        "pool_dir": str(out_dir.resolve()),
        "layout": "tokenized/<domain>/<domain>.u32le.bin",
        "mode": "domain_stratified_stream",
    }
    (out_dir / "pool_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("working pool ready", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mixtures-json", type=Path, required=True)
    ap.add_argument("--budget-tokens", type=int, default=10_000_000_000)
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--dataset-version", default=None, help="Pin version; default resolve_latest")
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-concat", action="store_true")
    args = ap.parse_args()

    stage_pool(
        out_dir=args.out_dir,
        mixtures_json=args.mixtures_json,
        budget_tokens=args.budget_tokens,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        seed=args.seed,
        skip_download=args.skip_download,
        skip_concat=args.skip_concat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
