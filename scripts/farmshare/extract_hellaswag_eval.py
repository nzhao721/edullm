#!/usr/bin/env python3
"""Copy HellaSwag metrics from a combined ARC+HellaSwag eval JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    d = json.loads(src.read_text(encoding="utf-8"))
    out = {
        **d,
        "tasks": ["HellaSwag"],
        "labels": {
            "hellaswag_val_rc_5shot_bpb": d["labels"]["hellaswag_val_rc_5shot_bpb"],
        },
        "accuracy_labels": {
            "hellaswag_val_rc_5shot_acc": d["accuracy_labels"]["hellaswag_val_rc_5shot_acc"],
        },
        "task_families": {"hellaswag": d["task_families"]["hellaswag"]},
        "accuracy_families": {"hellaswag": d["accuracy_families"]["hellaswag"]},
        "macro_mean": d["task_families"]["hellaswag"],
        "macro_mean_accuracy": d["accuracy_families"]["hellaswag"],
    }
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
