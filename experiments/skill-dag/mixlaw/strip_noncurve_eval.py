#!/usr/bin/env python3
"""Drop non-curve task families from collected pilot eval JSON files."""
from __future__ import annotations

import json
from pathlib import Path

from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, macro_curve

ROOT = Path(__file__).parent


def strip_final(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = {k: v for k, v in payload["labels"].items() if k in CURVE_TASK_LOSS_LABELS}
    families = {k: v for k, v in payload["task_families"].items() if k in CURVE_FAMILIES}
    if set(families) != set(CURVE_FAMILIES):
        raise SystemExit(f"{path}: missing curve families")
    payload["labels"] = labels
    payload["task_families"] = families
    payload["macro_mean"] = macro_curve(families)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> None:
    n = 0
    for path in sorted((ROOT / "pilot_runs").glob("mix*/progress/task_loss_final.json")):
        strip_final(path)
        n += 1
    print(f"stripped {n} task_loss_final.json files")


if __name__ == "__main__":
    main()
