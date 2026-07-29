#!/usr/bin/env python3
import torch
from pathlib import Path

def inspect(path: Path):
    print("===", path, "===")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tm = ck["train_module"]
    model = tm["model"] if isinstance(tm, dict) and "model" in tm else tm
    emb = model["embeddings.weight"]
    print("emb type", type(emb), "shape", tuple(emb.shape))
    for attr in ("full_tensor", "to_local", "shape", "device_mesh", "placements"):
        print(" ", attr, getattr(emb, attr, None))
    ft = getattr(emb, "full_tensor", None)
    if callable(ft):
        try:
            print(" full_tensor()", tuple(ft().shape))
        except Exception as e:
            print(" full_tensor err", e)
    print(" format", ck.get("checkpoint_format"))
    print(" arch", ck.get("architecture"))

inspect(Path("/scratch/users/nzhao2/agent-runs/ce-regmix10b-models-20260727T212156Z/checkpoints/step2384/state.pt"))
