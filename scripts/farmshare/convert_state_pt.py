#!/usr/bin/env python3
"""Convert BLADE/CE state.pt (HSDP DTensor pickle) to a plain model.pt."""
from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import torch


def install_dtensor_stubs() -> None:
    class _Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._args = args
            self._kwargs = kwargs

        def __setstate__(self, state: Any) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self._state = state

        def __repr__(self) -> str:
            return f"_Stub({self.__class__.__name__})"

    def ensure(name: str) -> types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        try:
            return importlib.import_module(name)
        except Exception:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(ensure(parent), child, mod)
            return mod

    dt_mod = ensure("torch.distributed.tensor")
    if not hasattr(dt_mod, "DTensor"):
        try:
            legacy = importlib.import_module("torch.distributed._tensor")
            setattr(dt_mod, "DTensor", getattr(legacy, "DTensor"))
        except Exception:
            setattr(dt_mod, "DTensor", type("DTensor", (_Stub,), {}))

    needed = {
        "torch.distributed._mesh_layout": ["_MeshLayout", "_FlatLayout"],
        "torch.distributed._pycute": ["Layout", "IntTuple"],
        "torch.distributed.tensor._dtensor_spec": [
            "DTensorSpec",
            "ShardOrderEntry",
            "TensorMeta",
        ],
    }
    for mod_name, attrs in needed.items():
        mod = ensure(mod_name)
        for attr in attrs:
            if not hasattr(mod, attr):
                setattr(mod, attr, type(attr, (_Stub,), {}))


def to_local(t: Any) -> torch.Tensor:
    full = getattr(t, "full_tensor", None)
    if callable(full):
        try:
            return full().detach().cpu()
        except Exception:
            pass
    if torch.is_tensor(t) and type(t).__name__ == "Tensor":
        return t.detach().cpu()
    d = getattr(t, "__dict__", {}) or {}
    for key in ("_local_tensor", "_local_tensor_data", "local_tensor"):
        v = d.get(key, getattr(t, key, None))
        if torch.is_tensor(v):
            return v.detach().cpu()
    local = getattr(t, "to_local", None)
    if callable(local):
        try:
            return local().detach().cpu()
        except Exception:
            pass
    if torch.is_tensor(t):
        return t.detach().cpu()
    raise TypeError(f"cannot localize {type(t)} attrs={list(d)[:20]}")


def extract_model_state(tm_sd: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" in tm_sd and isinstance(tm_sd["model"], dict):
        nested = tm_sd["model"]
        return {k: to_local(v) for k, v in nested.items()}
    prefixed = {
        k[len("model.") :]: to_local(v)
        for k, v in tm_sd.items()
        if k.startswith("model.")
    }
    if prefixed:
        return prefixed
    out: dict[str, torch.Tensor] = {}
    for k, v in tm_sd.items():
        try:
            out[k] = to_local(v)
        except Exception:
            continue
    if not out:
        raise RuntimeError(f"no tensors found; keys={list(tm_sd.keys())[:40]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    install_dtensor_stubs()
    src = args.checkpoint / "state.pt"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    print(f"loading {src}", flush=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step") or (args.checkpoint / "step.txt").read_text().strip())
    model_sd = extract_model_state(ckpt["train_module"])
    emb = model_sd.get("embeddings.weight")
    if emb is not None and tuple(emb.shape) != (100_352, 1024):
        raise SystemExit(
            f"embeddings.weight shape {tuple(emb.shape)} != (100352, 1024); "
            "legacy HSDP rank-0 shard — use gather_hsdp_state_pt_to_model.py "
            "with torchrun --nproc_per_node=2 instead of this converter."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "model": model_sd}
    torch.save(payload, args.out)
    print(f"wrote {args.out} n_tensors={len(model_sd)} step={step}", flush=True)


if __name__ == "__main__":
    main()
