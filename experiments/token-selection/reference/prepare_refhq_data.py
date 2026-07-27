#!/usr/bin/env python3
"""Prepare RefHQ tokenized memmaps for OLMo-ladder 370M training.

Discovers ``tokenized/<domain>/<domain>.npy`` under ``--tokenized-root`` (or under
``<work>/tokenized``), writes ``paths.txt``, then runs a mix-proportional ~30M-token
validation holdout via ``make_val_holdout.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def discover_domain_npys(tokenized_root: Path) -> list[Path]:
    found: list[Path] = []
    for domain in KNOWN_DOMAINS:
        candidate = tokenized_root / domain / f"{domain}.npy"
        if candidate.is_file():
            found.append(candidate)
            continue
        # Fallback: any .npy directly under the domain folder.
        domain_dir = tokenized_root / domain
        if domain_dir.is_dir():
            found.extend(sorted(domain_dir.glob("*.npy")))
    # Also pick up unexpected domain folders that still look like RefHQ layout.
    for child in sorted(tokenized_root.iterdir()):
        if not child.is_dir() or child.name in {"holdout", "train", "val"}:
            continue
        if child.name in KNOWN_DOMAINS:
            continue
        found.extend(sorted(child.glob("*.npy")))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True, help="Run directory (contains tokenized/)")
    ap.add_argument(
        "--tokenized-root",
        type=Path,
        default=None,
        help="Override path to tokenized/<domain>/*.npy (default: <work>/tokenized)",
    )
    ap.add_argument("--target-val-tokens", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--skip-holdout", action="store_true", help="Train on all tokens; no val split")
    args = ap.parse_args()

    work = args.work
    tok = args.tokenized_root or (work / "tokenized")
    tok.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    paths = discover_domain_npys(tok)
    if not paths:
        raise SystemExit(f"No domain .npy memmaps under {tok}")

    paths_file = tok / "paths.txt"
    paths_file.write_text("\n".join(str(p.resolve()) for p in paths) + "\n")

    totals = {p.parent.name: p.stat().st_size // 4 for p in paths}
    summary = {
        "tokenized_root": str(tok.resolve()),
        "n_files": len(paths),
        "domains": totals,
        "total_tokens": sum(totals.values()),
        "paths_file": str(paths_file.resolve()),
    }
    (tok / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    if args.skip_holdout:
        train_file = tok / "paths_train.txt"
        train_file.write_text(paths_file.read_text())
        (tok / "paths_val.txt").write_text("")
        (work / "length_tokens.txt").write_text(str(summary["total_tokens"]) + "\n")
        print("skip-holdout: paths_train.txt = paths.txt; no val", flush=True)
        return 0

    holdout_script = Path(__file__).resolve().parent / "make_val_holdout.py"
    cmd = [
        sys.executable,
        str(holdout_script),
        str(work),
        "--target-tokens",
        str(args.target_val_tokens),
        "--seed",
        str(args.seed),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
