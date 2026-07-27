#!/usr/bin/env python3
"""Prepare RegMix train + RefHQ validation memmap path lists for BLADE 370M.

Expects local copies of the already-tokenized corpora:

  <train-tokenized-root>/<domain>/<domain>.npy   # s3://edullm-dataset-regmix/regmix-10b/tokenized
  <ref-tokenized-root>/<domain>/<domain>.npy     # s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/tokenized

Writes:
  <work>/train_tokenized/paths_train.txt
  <work>/ref_tokenized/paths_refhq.txt
  <work>/length_tokens.txt
  <work>/blade_data_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KNOWN_DOMAINS = (
    "dclm",
    "starcoder",
    "pes2o",
    "arxiv",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)

# Measured content tokens (S3_DATASETS.md / manifests).
REGMIX_TOKENS = 10_000_058_051
REFHQ_TOKENS = 5_514_030_574


def discover_domain_npys(tokenized_root: Path) -> list[Path]:
    found: list[Path] = []
    for domain in KNOWN_DOMAINS:
        candidate = tokenized_root / domain / f"{domain}.npy"
        if candidate.is_file():
            found.append(candidate)
            continue
        domain_dir = tokenized_root / domain
        if domain_dir.is_dir():
            found.extend(sorted(domain_dir.glob("*.npy")))
    for child in sorted(tokenized_root.iterdir()):
        if not child.is_dir() or child.name in {"holdout", "train", "val"}:
            continue
        if child.name in KNOWN_DOMAINS:
            continue
        found.extend(sorted(child.glob("*.npy")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def write_paths(paths: list[Path], out_file: Path) -> dict:
    if not paths:
        raise SystemExit(f"No domain .npy memmaps for {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")
    totals = {p.parent.name: p.stat().st_size // 4 for p in paths}
    return {
        "n_files": len(paths),
        "domains": totals,
        "total_tokens_on_disk": sum(totals.values()),
        "paths_file": str(out_file.resolve()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument(
        "--train-tokenized-root",
        type=Path,
        required=True,
        help="Local RegMix tokenized/ root",
    )
    ap.add_argument(
        "--ref-tokenized-root",
        type=Path,
        required=True,
        help="Local RefHQ tokenized/ root",
    )
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=REGMIX_TOKENS,
        help="Proxy training token budget (default: measured RegMix 10B)",
    )
    args = ap.parse_args()

    work = args.work
    work.mkdir(parents=True, exist_ok=True)

    train_paths = discover_domain_npys(args.train_tokenized_root)
    ref_paths = discover_domain_npys(args.ref_tokenized_root)

    train_info = write_paths(train_paths, work / "train_tokenized" / "paths_train.txt")
    ref_info = write_paths(ref_paths, work / "ref_tokenized" / "paths_refhq.txt")

    (work / "length_tokens.txt").write_text(str(int(args.length_tokens)) + "\n")

    summary = {
        "train": {
            **train_info,
            "s3": "s3://edullm-dataset-regmix/regmix-10b/tokenized",
            "tokenized_root": str(args.train_tokenized_root.resolve()),
            "published_tokens": REGMIX_TOKENS,
        },
        "refhq": {
            **ref_info,
            "s3": "s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/tokenized",
            "tokenized_root": str(args.ref_tokenized_root.resolve()),
            "published_tokens": REFHQ_TOKENS,
        },
        "length_tokens": int(args.length_tokens),
    }
    (work / "blade_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
