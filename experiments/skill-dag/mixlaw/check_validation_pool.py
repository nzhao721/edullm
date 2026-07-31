#!/usr/bin/env python3
"""Check validation-mix peak domain demand vs olmohq pool at 10B budget.

Domain weights come solely from ``validation_mixtures_10b.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

from mixlaw_common import DOMAINS

ROOT = Path(__file__).parent
PLAN = json.loads((ROOT / "validation_mixtures_10b.json").read_text(encoding="utf-8"))

PLAN_AVAIL = {
    "dclm": 28_600_000_000,
    "arxiv": 20_800_000_000,
    "starcoder": 20_300_000_000,
    "pes2o": 26_300_000_000,
    "open-web-math": 12_200_000_000,
    "algebraic-stack": 11_800_000_000,
    "wiki": 3_660_000_000,
}
MEASURED_AVAIL = {
    "dclm": 29_691_000_000,
    "arxiv": 22_148_000_000,
    "starcoder": 6_139_000_000,
    "pes2o": 12_309_000_000,
    "open-web-math": 13_238_000_000,
    "algebraic-stack": 12_902_000_000,
    "wiki": 3_752_000_000,
}

BUDGET = int(PLAN.get("budget_tokens", 10_000_000_000))


def main() -> None:
    rows = [
        (m["run_name"], dict(zip(DOMAINS, m["weights"])))
        for m in PLAN["mixtures"]
    ]

    peak = {d: 0.0 for d in DOMAINS}
    binding_mix = {d: "" for d in DOMAINS}
    print("Per-mixture domain token needs at 10B budget (billions):")
    hdr = f"{'mix':<16}" + "".join(f"{d[:4]:>8}" for d in DOMAINS)
    print(hdr)
    for name, w in rows:
        needs = {d: w[d] * BUDGET for d in DOMAINS}
        line = f"{name:<16}" + "".join(f"{needs[d] / 1e9:8.3f}" for d in DOMAINS)
        print(line)
        for d in DOMAINS:
            if needs[d] > peak[d]:
                peak[d] = needs[d]
                binding_mix[d] = name

    for label, avail in (
        ("plan (mixlaw_common)", PLAN_AVAIL),
        ("measured (S3 manifest)", MEASURED_AVAIL),
    ):
        print()
        print(f"=== Peak demand vs {label} ===")
        print(
            f"{'domain':<18}{'peak need':>12}{'available':>12}{'margin':>12}{'status':>8}  binding mix"
        )
        ok_all = True
        for d in DOMAINS:
            need = peak[d]
            av = avail[d]
            margin = av - need
            ok = margin >= 0
            ok_all = ok_all and ok
            status = "OK" if ok else "SHORT"
            print(
                f"{d:<18}{need / 1e9:12.3f}{av / 1e9:12.3f}{margin / 1e9:12.3f}{status:>8}  {binding_mix[d]}"
            )
        print("ALL OK" if ok_all else "FAILURES PRESENT")

    print()
    print("=== Per-mixture failures vs measured S3 pool ===")
    any_fail = False
    for name, w in rows:
        fails: list[tuple[str, float, float]] = []
        for d in DOMAINS:
            need = w[d] * BUDGET
            if need > MEASURED_AVAIL[d]:
                fails.append((d, need / 1e9, MEASURED_AVAIL[d] / 1e9))
        if fails:
            any_fail = True
            print(name + ":")
            for d, need, av in fails:
                print(f"  {d}: need {need:.3f}B, avail {av:.3f}B, short {need - av:.3f}B")
    if not any_fail:
        print(f"All {len(rows)} mixtures fit measured per-domain availability.")


if __name__ == "__main__":
    main()
