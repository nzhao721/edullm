#!/usr/bin/env python3
import torch
from pathlib import Path
paths = [
    ("refhq", Path("/scratch/users/nzhao2/agent-runs/refhq-unshard-20260727/reference/refhq_step1315_model.pt")),
    ("ce", Path("/scratch/users/nzhao2/agent-runs/ce-regmix10b-models-20260727T212156Z/models/step2384/model.pt")),
    ("blade", Path("/scratch/users/nzhao2/agent-runs/blade-regmix10b-models-20260727T212156Z/models/step2384/model.pt")),
]
for name, p in paths:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        sd = obj["model"]
    else:
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
    emb = sd.get("embeddings.weight")
    print(name, "n_tensors", len(sd), "emb", tuple(emb.shape) if emb is not None else None)
