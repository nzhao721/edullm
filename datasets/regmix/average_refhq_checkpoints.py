#!/usr/bin/env python3
"""Average RefHQ checkpoint weights into one model_eval.pt file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping) and value:
                return {str(k): v.detach().cpu() for k, v in value.items() if torch.is_tensor(v)}
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return {str(k): v.detach().cpu() for k, v in payload.items()}
    raise TypeError(f"could not find model tensor state in {path}")


def average_states(paths: list[Path]) -> dict[str, torch.Tensor]:
    if not paths:
        raise ValueError("at least one checkpoint is required")
    first = load_model_state(paths[0])
    sums: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    for key, tensor in first.items():
        dtypes[key] = tensor.dtype
        if tensor.is_floating_point():
            sums[key] = tensor.to(dtype=torch.float32)
        else:
            sums[key] = tensor.clone()

    for path in paths[1:]:
        state = load_model_state(path)
        if set(state) != set(sums):
            missing = sorted(set(sums) - set(state))[:8]
            extra = sorted(set(state) - set(sums))[:8]
            raise KeyError(f"state keys differ for {path}: missing={missing} extra={extra}")
        for key, acc in sums.items():
            tensor = state[key]
            if tuple(tensor.shape) != tuple(acc.shape):
                raise ValueError(
                    f"shape mismatch for {key} in {path}: {tuple(tensor.shape)} != {tuple(acc.shape)}"
                )
            if tensor.is_floating_point():
                acc.add_(tensor.to(dtype=torch.float32))

    denom = float(len(paths))
    averaged: dict[str, torch.Tensor] = {}
    for key, acc in sums.items():
        if acc.is_floating_point():
            averaged[key] = (acc / denom).to(dtype=dtypes[key])
        else:
            averaged[key] = acc.to(dtype=dtypes[key])
    return averaged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(json.dumps({"event": "skip_existing", "out": str(args.out)}, sort_keys=True))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    state = average_states(args.checkpoint)
    tmp = Path(str(args.out) + ".tmp")
    torch.save(
        {
            "step": "avg(" + ",".join(path.parent.name for path in args.checkpoint) + ")",
            "model": state,
            "averaged_checkpoints": [str(path) for path in args.checkpoint],
        },
        tmp,
    )
    tmp.replace(args.out)
    print(
        json.dumps(
            {
                "event": "averaged_checkpoint_ready",
                "out": str(args.out),
                "n_checkpoints": len(args.checkpoint),
                "n_tensors": len(state),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
