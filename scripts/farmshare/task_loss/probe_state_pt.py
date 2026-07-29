#!/usr/bin/env python3
"""Probe state.pt load with DTensor compatibility shims."""
import importlib
import sys
import types
import torch
import torch.distributed.tensor as tdt

# Shim missing torch.distributed.tensor submodules from newer torch pickles.
def _ensure_mod(name):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        p = _ensure_mod(parent)
        setattr(p, child, mod)
    return mod

_dt_mod = importlib.import_module("torch.distributed._tensor")
DT = getattr(_dt_mod, "DTensor")
tdt.DTensor = DT
setattr(_ensure_mod("torch.distributed.tensor"), "DTensor", DT)

# Common newer-torch modules referenced by pickled DTensors
for _name in [
    "torch.distributed.tensor._dtensor_spec",
    "torch.distributed.tensor._api",
    "torch.distributed.tensor._utils",
    "torch.distributed.tensor._ops",
]:
    _ensure_mod(_name)

# Provide DTensorSpec alias if present under legacy path
try:
    from torch.distributed._tensor.placement_types import DTensorSpec as _Spec
    setattr(sys.modules["torch.distributed.tensor._dtensor_spec"], "DTensorSpec", _Spec)
except Exception as exc:
    print("spec alias failed", type(exc).__name__, exc, flush=True)

p = "/scratch/users/nzhao2/checkpoints/token-selection-370m/blade/checkpoints/step250/state.pt"
print("loading", p, flush=True)
try:
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    print("LOADED", sorted(ckpt.keys()), flush=True)
    tm = ckpt["train_module"]
    print("tm nkeys", len(tm), "sample", list(tm.keys())[:20], flush=True)
except Exception as exc:
    print("FAIL", type(exc).__name__, exc, flush=True)
    raise

