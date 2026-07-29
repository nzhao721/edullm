#!/usr/bin/env python3
"""Gather legacy HSDP-sharded state.pt into a full inference model.pt (2-GPU).

Legacy CE/BLADE checkpoints were saved rank-0-only with local DTensor shards.
Loading them into a live 2-rank HSDP module and gathering ``full_state_dict``
can recover the full weights when both ranks participate.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"torch.distributed.checkpoint.state_dict unavailable: {exc}") from exc

log = logging.getLogger("gather_hsdp_state_pt")
SEQ_LEN = 2048
EMBEDDING_SIZE = 100_352
MICROBATCH_TOKENS = 65_536


def resolve_attn_backend() -> AttentionBackendName | None:
    try:
        return AttentionBackendName.torch
    except Exception:
        return None


def build_train_module() -> TransformerTrainModule:
    backend = resolve_attn_backend()
    kwargs: dict[str, Any] = {"vocab_size": EMBEDDING_SIZE}
    if backend is not None:
        kwargs["attn_backend"] = backend
    model_cfg = TransformerConfig.olmo2_370M(**kwargs)
    try:
        scheduler = CosWithWarmup(warmup_steps=2000, alpha_f=0.1)
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=2000)
    tm_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=MICROBATCH_TOKENS,
        max_sequence_length=SEQ_LEN,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    model = model_cfg.build(init_device="cuda")
    return tm_cfg.build(model)


def plainify(obj: Any) -> Any:
    if torch.is_tensor(obj) and type(obj).__name__ == "Tensor":
        return obj.detach().cpu()
    full = getattr(obj, "full_tensor", None)
    if callable(full):
        return full().detach().cpu()
    local = getattr(obj, "to_local", None)
    if callable(local):
        return local().detach().cpu()
    if isinstance(obj, dict):
        return {k: plainify(v) for k, v in obj.items()}
    return obj


def load_legacy_checkpoint(train_module: TransformerTrainModule, checkpoint_dir: Path) -> int:
    path = checkpoint_dir / "state.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step") or (checkpoint_dir / "step.txt").read_text().strip())
    tm_sd = ckpt["train_module"]
    fmt = ckpt.get("checkpoint_format")
    log.info("checkpoint_format=%s step=%s", fmt, step)
    if (
        fmt == "full_state_dict_v1"
        and isinstance(tm_sd, dict)
        and "model" in tm_sd
    ):
        opts = StateDictOptions(full_state_dict=True, strict=True)
        set_model_state_dict(train_module.model, tm_sd["model"], options=opts)
    else:
        train_module.load_state_dict(tm_sd)
    return step


def gather_model_state(train_module: TransformerTrainModule) -> dict[str, torch.Tensor]:
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    sd = get_model_state_dict(train_module.model, options=opts)
    return {k: plainify(v) for k, v in sd.items()}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [rank %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prepare_training_environment()
    try:
        world = get_world_size()
        rank = get_rank()
        if world != 2:
            raise SystemExit(f"expected world_size=2 for HSDP gather, got {world}")
        if rank == 0:
            log.info("gather from %s -> %s", args.checkpoint, args.out)

        train_module = build_train_module()
        step = load_legacy_checkpoint(train_module, args.checkpoint)
        dist.barrier()

        model_sd = gather_model_state(train_module)
        if rank == 0:
            emb = model_sd.get("embeddings.weight")
            if emb is None:
                raise RuntimeError("missing embeddings.weight after gather")
            emb_shape = tuple(emb.shape)
            log.info("gathered embeddings.weight shape=%s", emb_shape)
            if emb_shape != (EMBEDDING_SIZE, 1024):
                raise RuntimeError(
                    f"gathered embeddings.weight shape {emb_shape} != ({EMBEDDING_SIZE}, 1024)"
                )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "model": model_sd}, args.out)
            log.info("wrote %s (%d tensors, step=%s)", args.out, len(model_sd), step)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
