#!/usr/bin/env python3
"""Build model_eval.pt from an unsharded model.pt checkpoint directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

EMBEDDING_SIZE = 100_352


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: <checkpoint-dir>/model_eval.pt",
    )
    args = ap.parse_args()

    ckpt = args.checkpoint_dir
    src = ckpt / "model.pt"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    out = args.out or (ckpt / "model_eval.pt")
    if out.is_file():
        print(f"exists {out}")
        return

    obj = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        sd = obj["model"]
        step = obj.get("step")
    else:
        sd = obj
        step = None
    if step is None:
        step_txt = ckpt / "step.txt"
        if step_txt.is_file():
            step = int(step_txt.read_text().strip())
        else:
            step = int(ckpt.name.replace("step", "").split("-")[0])

    emb = sd.get("embeddings.weight")
    if emb is None:
        raise SystemExit("missing embeddings.weight in model.pt")
    emb_shape = tuple(emb.shape)
    if emb_shape != (EMBEDDING_SIZE, 1024):
        raise SystemExit(
            f"bad embeddings.weight shape {emb_shape}; expected ({EMBEDDING_SIZE}, 1024)"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step), "model": sd}, out)
    print(f"wrote {out} step={step} emb={emb_shape}")


if __name__ == "__main__":
    main()
