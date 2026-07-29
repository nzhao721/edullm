#!/usr/bin/env python3
"""Build exact OLMo-ladder 370M TrainConfig and launch training on one GPU.

Architecture, LR, global batch, schedule, and tokenizer match
``experiments/baseline/farmshare/train_olmo_ladder_370m.py`` / allenai OLMo-ladder
``src/ladder/ladder.py``. Intended for a single B200 with the already-tokenized
RefHQ RegMix 5.5B corpus (see ``experiments/token-selection/reference/train_olmo3_370m_refhq.py``).

W&B is hard-disabled. No AI2 weka evals.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from math import cos, pi
from pathlib import Path
from typing import Optional

# Hard-disable W&B before importing olmo (which may read these at import time).
os.environ["WANDB_DISABLED"] = "1"
os.environ["WANDB_MODE"] = "disabled"
for _var in (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_NAME",
    "WANDB_GROUP",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_DIR",
    "WANDB_CACHE_DIR",
    "WANDB_ENABLE",
):
    os.environ.pop(_var, None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# PyTorch >=2.6 defaults torch.load(weights_only=True). OLMo 0.6 checkpoints
# pickle pathlib paths (incl. Python 3.13 pathlib._local.PosixPath), which fails
# the safe unpickler. These are our own local training artifacts — allow full load.
_orig_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted  # type: ignore[assignment]
try:
    import pathlib as _pathlib

    _safe = [_pathlib.Path, _pathlib.PosixPath, _pathlib.WindowsPath]
    try:
        from pathlib import _local as _pathlib_local  # type: ignore[attr-defined]

        _safe.extend(
            [
                getattr(_pathlib_local, "Path", None),
                getattr(_pathlib_local, "PosixPath", None),
                getattr(_pathlib_local, "WindowsPath", None),
            ]
        )
    except Exception:
        pass
    torch.serialization.add_safe_globals([x for x in _safe if x is not None])
except Exception:
    pass

from olmo import (
    ActivationType,
    DDPConfig,
    InitFnType,
    LayerNormType,
    ModelConfig,
    OptimizerConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TokenizerConfig,
    TrainConfig,
)
from olmo.config import (
    ActivationCheckpointingStrategy,
    DataConfig,
    DistributedStrategy,
    FSDPConfig,
    FSDPPrecision,
    FSDPWrapStrategy,
    InstanceFilterConfig,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
)
from olmo.optim import Scheduler, build_scheduler as _olmo_build_scheduler
from olmo.torch_util import get_global_rank, get_local_rank
from olmo.util import add_cached_path_clients, find_latest_checkpoint, prepare_cli_environment
from torch.distributed.fsdp import ShardingStrategy

log = logging.getLogger("train_olmo_ladder_370m")

# Exact values from allenai/OLMo-ladder src/ladder/ladder.py
MODEL_SIZE = 371_262_464
SEQ_LEN = 4096

# ai2-olmo 0.6.0 SchedulerConfig has no t_decay; ladder.py sets
# t_decay = 0.1 * length_tokens / tokens_per_step (final 10% cosine anneal).
_LADDER_T_DECAY: Optional[int] = None


@dataclass
class CosWithWarmupAndDecay(Scheduler):
    """OLMo-ladder LR schedule: warmup → constant peak → cosine over final t_decay steps."""

    warmup_steps: int
    t_decay: int
    alpha_f: float = 0.1

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        decay_start = max(self.warmup_steps, max_steps - max(self.t_decay, 0))
        if step < decay_start:
            return initial_lr
        if step >= max_steps or max_steps <= decay_start:
            return eta_min
        progress = (step - decay_start) / (max_steps - decay_start)
        return eta_min + (initial_lr - eta_min) * (1 + cos(pi * progress)) / 2


def build_scheduler(cfg: TrainConfig, sched_cfg: Optional[SchedulerConfig] = None) -> Scheduler:
    """Prefer ladder WSD schedule when _LADDER_T_DECAY is set; else ai2-olmo default."""
    if _LADDER_T_DECAY is None:
        return _olmo_build_scheduler(cfg, sched_cfg)
    sched_cfg = sched_cfg if sched_cfg is not None else cfg.scheduler
    return CosWithWarmupAndDecay(
        grad_clip_warmup_steps=(
            None if sched_cfg.grad_clip_warmup_steps is None else int(sched_cfg.grad_clip_warmup_steps)
        ),
        grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
        warmup_min_lr=sched_cfg.warmup_min_lr,
        warmup_steps=int(sched_cfg.t_warmup),
        t_decay=int(_LADDER_T_DECAY),
        alpha_f=sched_cfg.alpha_f,
    )


def ladder_global_batch_size(batch_size_divisor: int = 32) -> int:
    gbs = 160 * (MODEL_SIZE / 108_000_000) ** (2 / 3)
    gbs /= 2  # seq 4096
    gbs /= batch_size_divisor
    gbs = round(gbs)
    gbs *= batch_size_divisor
    return int(gbs)


def ladder_lr() -> float:
    lr = 0.0047 * (MODEL_SIZE / 108_000_000) ** (-1 / 3)
    lr /= 4  # seq 4096
    return float(lr)


def resolve_flash_attention() -> bool:
    """Prefer flash-attn; fall back to PyTorch SDPA if the extension is missing.

    Architecture/hparams are unchanged either way — only the attention kernel path.
    """
    want = os.environ.get("OLMO_FLASH_ATTENTION", "1") == "1"
    if not want:
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception as e:
        log.warning("OLMO_FLASH_ATTENTION=1 but flash_attn unavailable (%s); using PyTorch SDPA", e)
        return False


def build_model_config() -> ModelConfig:
    flash = resolve_flash_attention()
    return ModelConfig(
        d_model=1024,
        n_heads=16,
        n_layers=16,
        mlp_ratio=8,
        weight_tying=False,
        alibi=False,
        rope=True,
        rope_theta=500_000,
        flash_attention=flash,
        attention_dropout=0.0,
        attention_layer_norm=True,
        include_bias=False,
        layer_norm_type=LayerNormType.rms,
        layer_norm_with_affine=True,
        layer_norm_eps=1e-6,
        bias_for_layer_norm=False,
        attention_layer_norm_with_affine=True,
        activation_type=ActivationType.swiglu,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        max_sequence_length=SEQ_LEN,
        vocab_size=100_278,
        embedding_size=100_352,
        eos_token_id=100_257,
        pad_token_id=100_277,
        init_device="cuda",
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_cutoff_factor=3,
        norm_after=True,
        precision="amp_bf16",
    )


def activation_checkpointing_from_env() -> Optional[ActivationCheckpointingStrategy]:
    ac_env = os.environ.get("OLMO_ACTIVATION_CHECKPOINTING", "0").strip().lower()
    if ac_env in ("", "0", "none", "false", "off"):
        return None
    return ActivationCheckpointingStrategy(ac_env)


def skip_unsharded_save_steps() -> set[int]:
    """Steps where interval saves are skipped so we jump to the final checkpoint.

    Default skips 7000 on the RefHQ 7011-step run (save every 500 → otherwise
    7000 would be written 11 steps before the mandatory final save at 7011).
    """
    raw = os.environ.get("OLMO_SKIP_UNSHARDED_SAVE_STEPS", "7000")
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def install_checkpoint_skip_patch() -> None:
    """Skip selected unsharded interval saves; final end-of-run save still runs."""
    from olmo.config import CheckpointType
    from olmo.train import Trainer

    skip = skip_unsharded_save_steps()
    if not skip:
        return
    orig = Trainer.save_checkpoint

    def save_checkpoint(self, checkpoint_type=CheckpointType.sharded):  # type: ignore[no-untyped-def]
        if checkpoint_type == CheckpointType.unsharded and int(self.global_step) in skip:
            log.info(
                "Skipping unsharded checkpoint at step %s (will save final at train end)",
                self.global_step,
            )
            return None, None
        return orig(self, checkpoint_type)

    Trainer.save_checkpoint = save_checkpoint  # type: ignore[method-assign]
    log.info("Unsharded checkpoint skip steps: %s", sorted(skip))


def install_memory_efficient_loss_patch() -> None:
    """Avoid a full logits[..., :-1].contiguous() clone (peaks +~25GiB at mbs=32).

    Same CE + Z-loss math under reduction='sum' (the train path): score contiguous
    chunk views instead of cloning the entire shifted logits tensor at once.
    """
    from olmo.train import Trainer

    _orig = Trainer.model_forward

    def model_forward(self, batch, loss_reduction="mean", compute_z_loss=False):  # type: ignore[no-untyped-def]
        if loss_reduction == "none":
            return _orig(self, batch, loss_reduction=loss_reduction, compute_z_loss=compute_z_loss)

        logits = self.dist_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        ).logits
        labels = self.get_labels(batch)
        batch_size = int(logits.size(0))
        chunk = max(1, min(8, batch_size))
        ce_acc = None
        z_acc = None
        n_tokens = 0
        for start in range(0, batch_size, chunk):
            end = min(batch_size, start + chunk)
            chunk_logits = logits[start:end, :-1, :].contiguous().view(-1, logits.size(-1))
            chunk_labels = labels[start:end].reshape(-1)
            ce_b, z_b = self.loss_fn(
                chunk_logits,
                chunk_labels,
                ignore_index=-100,
                reduction="sum",
                compute_z_loss=compute_z_loss,
            )
            ce_acc = ce_b if ce_acc is None else (ce_acc + ce_b)
            if z_b is not None:
                z_acc = z_b if z_acc is None else (z_acc + z_b)
            n_tokens += int((chunk_labels != -100).sum().item())
            del chunk_logits, chunk_labels

        if loss_reduction == "mean":
            denom = max(n_tokens, 1)
            ce_acc = ce_acc / denom
            if z_acc is not None:
                z_acc = z_acc / denom

        return ce_acc, z_acc, logits

    Trainer.model_forward = model_forward  # type: ignore[method-assign]
    log.info("Installed memory-efficient chunked loss (no full logits contiguous clone)")


def build_config(args: argparse.Namespace) -> TrainConfig:
    paths = [ln.strip() for ln in Path(args.paths_file).read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No training paths in {args.paths_file}")

    # No validation / evals for the RefHQ B200 run.
    if args.val_paths_file:
        log.warning("--val-paths-file=%s ignored (evals disabled)", args.val_paths_file)

    gbs = ladder_global_batch_size(args.batch_size_divisor)
    mbs = args.device_batch_size
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if gbs % (mbs * world_size) != 0:
        raise SystemExit(
            f"global_batch_size {gbs} not divisible by microbatch {mbs} * world_size {world_size}"
        )

    length_tokens = int(args.length_tokens)
    tokens_per_step = gbs * SEQ_LEN
    total_steps = length_tokens // tokens_per_step
    lr = ladder_lr()
    t_warmup = round(MODEL_SIZE / tokens_per_step)
    # Exact OLMo-ladder: final 10% of steps are cosine decay (t_decay).
    t_decay = round(0.1 * length_tokens / tokens_per_step)

    save_folder = args.save_folder
    load_path = args.load_path
    if args.fresh:
        load_path = None
    elif load_path is None:
        load_path = find_latest_checkpoint(save_folder)

    meta = {
        "model": "370M",
        "non_embedding_params": MODEL_SIZE,
        "length_tokens": length_tokens,
        "global_batch_size_sequences": gbs,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "device_microbatch": mbs,
        "world_size": world_size,
        "grad_accum": gbs // (mbs * world_size),
        "lr": lr,
        "t_warmup": t_warmup,
        "t_decay": t_decay,
        "scheduler": "cos_with_warmup_and_decay",
        "save_interval_unsharded": args.save_interval,
        "skip_unsharded_save_steps": sorted(
            {
                int(x)
                for x in os.environ.get("OLMO_SKIP_UNSHARDED_SAVE_STEPS", "7000").split(",")
                if x.strip()
            }
        ),
        "seed": args.seed,
        "paths": len(paths),
        "val_paths": 0,
        "eval_interval": None,
        "flash_attention": resolve_flash_attention(),
        "activation_checkpointing": os.environ.get("OLMO_ACTIVATION_CHECKPOINTING", "0"),
        "fused_loss": os.environ.get("OLMO_FUSED_LOSS", "1") == "1",
        "dataset": "s3://edullm-datasets/refhq/refhq-regmix-5p5b-v1/",
    }
    Path(args.progress_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.progress_dir) / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (Path(args.progress_dir) / "total_steps.txt").write_text(str(total_steps) + "\n")

    return TrainConfig(
        run_name=args.name,
        seed=args.seed,
        wandb=None,
        model=build_model_config(),
        ddp=DDPConfig(),
        fsdp=FSDPConfig(
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.mixed,
        ),
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            learning_rate=lr,
            weight_decay=0.1,
            eps=1e-8,
            decay_norm_and_bias=True,
            decay_embeddings=False,
            betas=(0.9, 0.95),
            metrics_log_interval=10,
        ),
        # OLMo-ladder schedule: warmup → constant peak → cosine over final t_decay steps.
        # SchedulerConfig in ai2-olmo 0.6.0 has no t_decay; CosWithWarmupAndDecay is injected
        # via monkeypatched olmo.optim.build_scheduler (see main()).
        scheduler=SchedulerConfig(
            name=SchedulerType.cosine_with_warmup,
            alpha_f=0.1,
            warmup_min_lr=0.0,
            t_warmup=t_warmup,
            t_max=None,
        ),
        max_duration=f"{length_tokens}T",
        global_train_batch_size=gbs,
        tokenizer=TokenizerConfig(identifier="allenai/dolma2-tokenizer"),
        save_folder=save_folder,
        remote_save_folder=None,
        save_overwrite=True,
        # Smoke / short probes: skip the ephemeral step-0 unsharded save that
        # otherwise triggers an immediate restore under try_load_latest_save.
        no_pre_train_checkpoint=os.environ.get("OLMO_NO_PRE_TRAIN_CHECKPOINT", "0") == "1",
        save_interval_unsharded=args.save_interval,
        save_num_unsharded_checkpoints_to_keep=-1,
        save_interval=None,
        load_path=load_path,
        try_load_latest_save=bool(load_path) and not args.fresh,
        eval_on_load=False,
        sharded_checkpointer=ShardedCheckpointerType.olmo_core,
        device_train_microbatch_size=mbs,
        precision="amp_bf16",
        distributed_strategy=DistributedStrategy.ddp,
        # Avoid materializing full float32 logits on large microbatches.
        fused_loss=os.environ.get("OLMO_FUSED_LOSS", "1") == "1",
        gen1_gc_interval=2,
        max_grad_norm=1.0,
        speed_monitor=SpeedMonitorConfig(window_size=1),
        # No evals: empty evaluators + unreachable interval (trainer still calls
        # eval() once at stop_at, which is a no-op over an empty list).
        eval_interval=10**9,
        device_eval_batch_size=mbs,
        eval_subset_num_batches=0,
        evaluators=[],
        activation_checkpointing=activation_checkpointing_from_env(),
        data=DataConfig(
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            instance_filter=InstanceFilterConfig(),
            paths=paths,
            memmap_dtype="uint32",
            seed=args.data_seed,
        ),
        auxiliary_loss_multiplier=1e-5,
        softmax_auxiliary_loss=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--paths-file", required=True)
    ap.add_argument("--val-paths-file", default=None, help="Held-out memmap paths for LM validation loss")
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--length-tokens", type=int, required=True)
    # 1x B200: mbs=24 → grad_accum = 192/24 = 8 (same global batch as ladder).
    # Smoke on B200: mbs=32 OOMs; mbs=24 ~115k tok/s with flash_attn.
    ap.add_argument("--device-batch-size", type=int, default=24)
    ap.add_argument("--batch-size-divisor", type=int, default=32)
    # Every 500 steps; step 7000 is skipped via OLMO_SKIP_UNSHARDED_SAVE_STEPS
    # so the run goes 6500 → final 7011.
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--load-path", type=str, default=None)
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any checkpoints under --save-folder and start from scratch",
    )
    args = ap.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    # Throughput knobs that do not change ladder math.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    progress_path = Path(args.progress_dir) / "progress.log"
    loss_curve_path = Path(args.progress_dir) / "train_loss.jsonl"
    total_steps = (
        int((Path(args.progress_dir) / "total_steps.txt").read_text())
        if (Path(args.progress_dir) / "total_steps.txt").exists()
        else None
    )

    cfg = build_config(args)
    if total_steps is None:
        total_steps = int((Path(args.progress_dir) / "total_steps.txt").read_text())

    # Inject ladder t_decay schedule into ai2-olmo's scheduler builder BEFORE
    # importing ladder train.py (it does `from olmo.optim import build_scheduler`).
    global _LADDER_T_DECAY
    meta_path = Path(args.progress_dir) / "run_meta.json"
    _LADDER_T_DECAY = int(json.loads(meta_path.read_text())["t_decay"])
    import olmo.optim as olmo_optim

    olmo_optim.build_scheduler = build_scheduler  # type: ignore[assignment]
    log.info(
        "Using CosWithWarmupAndDecay: t_warmup=%s t_decay=%s alpha_f=0.1 flash_attn=%s "
        "activation_ckpt=%s fused_loss=%s world_size=%s mbs=%s save_interval=%s",
        cfg.scheduler.t_warmup,
        _LADDER_T_DECAY,
        cfg.model.flash_attention,
        cfg.activation_checkpointing,
        cfg.fused_loss,
        dist.get_world_size(),
        args.device_batch_size,
        args.save_interval,
    )

    class ProgressHandler(logging.Handler):
        """Ephemeral live train-loss curve (local only; not required after the run)."""

        def __init__(self) -> None:
            super().__init__()
            self._last_step: Optional[int] = None

        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "step=" not in msg and "Step " not in msg and "CrossEntropyLoss" not in msg:
                return
            try:
                import re

                step = None
                loss = None
                for part in msg.replace(",", " ").split():
                    if part.startswith("step="):
                        step = int(float(part.split("=", 1)[1]))
                m_loss = re.search(r"(?:train/)?CrossEntropyLoss[=: ]+([0-9.eE+-]+)", msg)
                if m_loss:
                    loss = float(m_loss.group(1))
                if loss is None:
                    for part in msg.replace(",", " ").split():
                        if part.startswith("loss="):
                            loss = float(part.split("=", 1)[1])
                if step is None:
                    return
                line = f"{step}/{total_steps}"
                if loss is not None:
                    line += f" loss={loss:.6f}"
                line += "\n"
                with open(progress_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                status = {
                    "step": step,
                    "total_steps": total_steps,
                    "train_loss": loss,
                    "pct": round(100.0 * step / max(total_steps, 1), 4),
                }
                (Path(args.progress_dir) / "progress.json").write_text(json.dumps(status) + "\n")
                # Append-only curve for live plotting; safe to delete after the run.
                if loss is not None and step != self._last_step:
                    with open(loss_curve_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "train_loss": loss}) + "\n")
                    self._last_step = step
            except Exception:
                return

    if get_global_rank() == 0:
        progress_path.write_text(f"0/{total_steps} starting\n")
        loss_curve_path.write_text("")  # truncate prior curve for this process
        handler = ProgressHandler()
        logging.getLogger().addHandler(handler)
        logging.getLogger("trainer").addHandler(handler)
        logging.getLogger("train").addHandler(handler)

    # Import ladder train main AFTER scheduler patch.
    ladder_root = Path(os.environ.get("OLMO_LADDER_ROOT", ""))
    if ladder_root:
        sys.path.insert(0, str(ladder_root / "src" / "ladder"))
    from train import main as train_main  # type: ignore
    import train as ladder_train  # type: ignore

    ladder_train.build_scheduler = build_scheduler  # belt-and-suspenders
    install_checkpoint_skip_patch()
    if os.environ.get("OLMO_CHUNKED_LOSS", "1") == "1":
        install_memory_efficient_loss_patch()

    train_main(cfg)


if __name__ == "__main__":
    main()
