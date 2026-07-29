#!/usr/bin/env python3
"""Plot Skill-It domain-weight trajectories from skillit_updates.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Optional PNG path")
    args = ap.parse_args()

    rows = []
    for line in args.jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"empty {args.jsonl}")

    domains = list(rows[0]["domain_order"])
    for index, row in enumerate(rows[1:], start=2):
        if list(row.get("domain_order", ())) != domains:
            raise SystemExit(
                f"{args.jsonl}: row {index} has a different domain_order; "
                "cannot plot incomparable weight vectors"
            )
    steps = [int(r["step"]) for r in rows]
    P = np.array([[float(r["p_after"][d]) for d in domains] for r in rows])

    print(f"{'step':>6}", *[f"{d[:8]:>8}" for d in domains])
    for s, row in zip(steps, P):
        print(f"{s:6d}", *[f"{v:8.4f}" for v in row])

    if args.out is not None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for j, d in enumerate(domains):
            ax.plot(steps, P[:, j], marker="o", label=d)
        ax.set_xlabel("step")
        ax.set_ylabel("domain weight p_after")
        ax.set_title(args.jsonl.parent.parent.name if args.jsonl.parent else "skillit")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
