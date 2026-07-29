#!/usr/bin/env python3
"""Scan CE state.pt for sharded vs full tensors."""
import torch
from pathlib import Path

ck = torch.load(
    Path("/scratch/users/nzhao2/agent-runs/ce-regmix10b-models-20260727T212156Z/checkpoints/step2384/state.pt"),
    map_location="cpu",
    weights_only=False,
)
model = ck["train_module"]["model"]
sharded = []
full = []
for k, v in model.items():
    if not (torch.is_tensor(v) or type(v).__name__ == "DTensor"):
        continue
    local = v.to_local() if hasattr(v, "to_local") else v
    gshape = tuple(v.shape) if hasattr(v, "shape") else tuple(local.shape)
    lshape = tuple(local.shape)
    if gshape != lshape:
        sharded.append((k, gshape, lshape))
    else:
        full.append((k, gshape))
print("sharded", len(sharded))
for row in sharded[:15]:
    print(" ", row)
print("full", len(full))
