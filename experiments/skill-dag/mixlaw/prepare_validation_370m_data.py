#!/usr/bin/env python3
"""Prepare local train path lists for mixlaw 370M validation mixes.

Reads domain weights / mix names from ``validation_mixtures_10b.json`` (the
canonical recipe). Expects each mix already materialized under::

    <tokenized-root>/<run_name>/{dclm,arxiv,...}.npy
    # or <tokenized-root>/<run_name>/tokenized/<domain>/<domain>.npy
    # source: s3://edullm-datasets/mixlaw/mixes/<run_name>/

Writes per mix::

    <work>/<run_name>/paths_train.txt
    <work>/<run_name>/mix_weights.json
    <work>/validation_arms.json   # index of all prepared arms
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mixlaw_common import DOMAINS

DEFAULT_RECIPE = Path(__file__).resolve().parent / "validation_mixtures_10b.json"


def find_domain_npys(mix_dir: Path) -> list[Path]:
    """Locate domain memmaps for one mix (flat or tokenized/ layout)."""
    found: list[Path] = []
    candidates = [
        mix_dir,
        mix_dir / "tokenized",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for domain in DOMAINS:
            p = root / domain / f"{domain}.npy"
            if p.is_file():
                found.append(p)
        if found:
            return found
        # Flat layout: <mix_dir>/<domain>.npy (build_mixture_data local slices)
        for domain in DOMAINS:
            p = root / f"{domain}.npy"
            if p.is_file():
                found.append(p)
        if found:
            return found
    return found


def prepare_mix(mix: dict, tokenized_root: Path, work: Path) -> dict:
    name = mix["run_name"]
    mix_src = tokenized_root / name
    if not mix_src.is_dir():
        raise SystemExit(
            f"missing mix directory {mix_src} "
            f"(sync from s3://edullm-datasets/mixlaw/mixes/{name}/)"
        )
    npys = find_domain_npys(mix_src)
    if not npys:
        raise SystemExit(f"no domain .npy files under {mix_src}")

    out_dir = work / name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_file = out_dir / "paths_train.txt"
    paths_file.write_text("\n".join(str(p.resolve()) for p in sorted(npys)) + "\n")

    weights = {d: float(w) for d, w in zip(DOMAINS, mix["weights"])}
    weights_path = out_dir / "mix_weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "run_name": name,
                "id": mix["id"],
                "tag": mix.get("tag"),
                "source": mix.get("source"),
                "domain_order": list(DOMAINS),
                "weights": weights,
                "recipe": "validation_mixtures_10b.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "run_name": name,
        "id": mix["id"],
        "paths_train": str(paths_file.resolve()),
        "mix_weights": str(weights_path.resolve()),
        "n_files": len(npys),
        "weights": weights,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE,
        help="validation_mixtures_10b.json (domain-weight source of truth)",
    )
    ap.add_argument(
        "--tokenized-root",
        type=Path,
        required=True,
        help="Local root containing one subdirectory per run_name",
    )
    ap.add_argument("--work", type=Path, required=True, help="Output directory for path lists")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of run_name values (default: all recipe mixes)",
    )
    args = ap.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    wanted = set(args.only) if args.only else None
    arms = []
    for mix in recipe["mixtures"]:
        if wanted is not None and mix["run_name"] not in wanted:
            continue
        arms.append(prepare_mix(mix, args.tokenized_root, args.work))
        print(f"prepared {mix['run_name']}: {arms[-1]['n_files']} files")

    if not arms:
        raise SystemExit("no mixtures prepared")

    index = {
        "recipe": str(args.recipe.resolve()),
        "budget_tokens": recipe.get("budget_tokens"),
        "domain_order": recipe.get("domain_order"),
        "arms": arms,
    }
    out = args.work / "validation_arms.json"
    out.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(arms)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
