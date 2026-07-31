#!/usr/bin/env python3
import json
import pathlib

d = pathlib.Path(
    "/scratch/users/nzhao2/agent-runs/smollm2-135m-500m-40ep-20260730-092920/output/evals"
)
print("step\tbpb\tacc")
for p in sorted(d.glob("step*_hellaswag.json")):
    j = json.loads(p.read_text())
    step = int(p.name.split("_")[0].replace("step", ""))
    bpb = j["labels"]["hellaswag_val_rc_5shot_bpb"]
    acc = j["accuracy_labels"]["hellaswag_val_rc_5shot_acc"]
    print(f"{step}\t{bpb:.6f}\t{acc:.4f}")
