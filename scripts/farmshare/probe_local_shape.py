#!/usr/bin/env python3
import torch
from pathlib import Path
ck = torch.load(Path("/scratch/users/nzhao2/agent-runs/ce-regmix10b-models-20260727T212156Z/checkpoints/step2384/state.pt"), map_location="cpu", weights_only=False)
emb = ck["train_module"]["model"]["embeddings.weight"]
print("dtensor shape", emb.shape)
print("local shape", emb.to_local().shape)
print("type", type(emb))
