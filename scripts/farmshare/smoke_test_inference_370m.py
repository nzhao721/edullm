#!/usr/bin/env python3
"""Smoke-test GPU inference for olmo2_370M weights saved as model.pt."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

EMBEDDING_SIZE = 100_352


def build_model() -> torch.nn.Module:
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig

    try:
        backend = AttentionBackendName.torch
    except Exception:
        backend = None
    kwargs: dict[str, Any] = {"vocab_size": EMBEDDING_SIZE}
    if backend is not None:
        kwargs["attn_backend"] = backend
    cfg = TransformerConfig.olmo2_370M(**kwargs)
    model = cfg.build(init_device="cpu")
    model.eval()
    return model


def load_model_state(path: Path) -> tuple[dict[str, torch.Tensor], int | None]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    step = None
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
        step = int(obj["step"]) if obj.get("step") is not None else None
    elif isinstance(obj, dict):
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if not sd:
            raise TypeError(f"{path} is not a tensor state dict")
    else:
        raise TypeError(f"unexpected checkpoint type: {type(obj)}")
    clean = {k: v.detach().cpu() for k, v in sd.items()}
    emb = clean.get("embeddings.weight")
    if emb is not None and tuple(emb.shape) != (EMBEDDING_SIZE, 1024):
        raise RuntimeError(
            f"embeddings.weight shape {tuple(emb.shape)} != ({EMBEDDING_SIZE}, 1024)"
        )
    return clean, step


def mean_ce_loss(model: torch.nn.Module, device: torch.device, *, batch: int, seq: int) -> float:
    torch.manual_seed(0)
    input_ids = torch.randint(1, 100_000, (batch, seq), device=device, dtype=torch.long)
    with torch.no_grad():
        if device.type == "cuda":
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        else:
            ctx = nullcontext()
        with ctx:
            out = model(input_ids, return_logits=True)
    logits = out.logits if hasattr(out, "logits") else out
    if not torch.is_tensor(logits):
        raise RuntimeError(f"unexpected forward output: {type(out)}")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return float(loss.item())


def smoke_one(label: str, path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"=== {label} ===", flush=True)
    print(f"path={path} bytes={path.stat().st_size}", flush=True)
    sd, step = load_model_state(path)
    model = build_model()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"warning: missing {len(missing)} keys (first 5): {missing[:5]}", flush=True)
    if unexpected:
        print(f"warning: unexpected {len(unexpected)} keys (first 5): {unexpected[:5]}", flush=True)
    if len(missing) > max(4, int(0.05 * (len(sd) + len(missing)))):
        raise RuntimeError(f"too many missing keys ({len(missing)}) for {label}")
    model.to(device)
    loss = mean_ce_loss(model, device, batch=2, seq=128)
    result = {
        "label": label,
        "path": str(path),
        "step": step,
        "n_tensors": len(sd),
        "smoke_ce_loss": loss,
        "device": str(device),
        "ok": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refhq", type=Path)
    ap.add_argument("--ce", type=Path)
    ap.add_argument("--blade", type=Path)
    ap.add_argument("--relema", type=Path)
    ap.add_argument(
        "--only-refhq",
        action="store_true",
        help="Run RefHQ smoke test only (requires --refhq)",
    )
    ap.add_argument(
        "--only-relema",
        action="store_true",
        help="Run REL-EMA smoke test only (requires --relema)",
    )
    ap.add_argument(
        "--only-ce-blade",
        action="store_true",
        help="Run CE + BLADE smoke tests only (requires --ce and --blade)",
    )
    ap.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="auto",
        help="Inference device (auto prefers cuda when available)",
    )
    args = ap.parse_args()

    only_flags = sum(bool(x) for x in (args.only_refhq, args.only_relema, args.only_ce_blade))
    if only_flags > 1:
        ap.error("use at most one of --only-refhq / --only-relema / --only-ce-blade")
    if args.only_refhq:
        if args.refhq is None:
            ap.error("--only-refhq requires --refhq")
        runs = (("refhq_step1315", args.refhq),)
    elif args.only_relema:
        if args.relema is None:
            ap.error("--only-relema requires --relema")
        runs = (("relema_step2386", args.relema),)
    elif args.only_ce_blade:
        if args.ce is None or args.blade is None:
            ap.error("--only-ce-blade requires --ce and --blade")
        runs = (
            ("ce_regmix10b_step2384", args.ce),
            ("blade_regmix10b_step2384", args.blade),
        )
    else:
        if args.refhq is None or args.ce is None or args.blade is None:
            ap.error("--refhq, --ce, and --blade are required unless --only-* is set")
        runs = [
            ("refhq_step1315", args.refhq),
            ("ce_regmix10b_step2384", args.ce),
            ("blade_regmix10b_step2384", args.blade),
        ]
        if args.relema is not None:
            runs.append(("relema_step2386", args.relema))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")

    if device.type == "cuda":
        print(f"cuda={torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("device=cpu", flush=True)

    results = []
    for label, path in runs:
        results.append(smoke_one(label, path, device))

    payload = {"models": results, "all_ok": all(r["ok"] for r in results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
