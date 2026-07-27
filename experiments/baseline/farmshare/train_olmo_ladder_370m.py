#!/usr/bin/env python3
"""Build exact OLMo-ladder 370M TrainConfig and launch training (no AI2 weka evals)."""
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

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    WandbConfig,
)
from olmo.config import (
    ActivationCheckpointingStrategy,
    DataConfig,
    DistributedStrategy,
    EvaluatorConfig,
    EvaluatorType,
    FSDPConfig,
    FSDPPrecision,
    FSDPWrapStrategy,
    InstanceFilterConfig,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
)
from olmo.optim import Scheduler, build_scheduler as _olmo_build_scheduler
from olmo.torch_util import get_local_rank
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


def build_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=1024,
        n_heads=16,
        n_layers=16,
        mlp_ratio=8,
        weight_tying=False,
        alibi=False,
        rope=True,
        rope_theta=500_000,
        flash_attention=os.environ.get("OLMO_FLASH_ATTENTION", "1") == "1",
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
    # Default off: whole-layer AC cuts throughput (~20k→18k tok/s) and is unnecessary
    # once flash-attn + fused CE keep mbs=8 under L40S memory.
    ac_env = os.environ.get("OLMO_ACTIVATION_CHECKPOINTING", "0").strip().lower()
    if ac_env in ("", "0", "none", "false", "off"):
        return None
    return ActivationCheckpointingStrategy(ac_env)


def build_config(args: argparse.Namespace) -> TrainConfig:
    paths = [ln.strip() for ln in Path(args.paths_file).read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No training paths in {args.paths_file}")

    val_paths: list[str] = []
    if args.val_paths_file:
        val_paths = [ln.strip() for ln in Path(args.val_paths_file).read_text().splitlines() if ln.strip()]
        if not val_paths:
            raise SystemExit(f"No validation paths in {args.val_paths_file}")
        overlap = set(paths) & set(val_paths)
        if overlap:
            raise SystemExit(f"Train/val path overlap ({len(overlap)} files); fix the split")

    gbs = ladder_global_batch_size(args.batch_size_divisor)
    mbs = args.device_batch_size
    if gbs % mbs != 0:
        raise SystemExit(f"global_batch_size {gbs} not divisible by microbatch {mbs}")

    length_tokens = int(args.length_tokens)
    tokens_per_step = gbs * SEQ_LEN
    total_steps = length_tokens // tokens_per_step
    lr = ladder_lr()
    t_warmup = round(MODEL_SIZE / tokens_per_step)
    # Exact OLMo-ladder: final 10% of steps are cosine decay (t_decay).
    # ai2-olmo 0.6.0 has no SchedulerConfig.t_decay; we inject CosWithWarmupAndDecay.
    t_decay = round(0.1 * length_tokens / tokens_per_step)

    save_folder = args.save_folder
    load_path = args.load_path
    if load_path is None:
        load_path = find_latest_checkpoint(save_folder)

    evaluators = []
    if val_paths:
        evaluators.append(
            EvaluatorConfig(
                label="val",
                type=EvaluatorType.lm,
                device_eval_batch_size=mbs,
                # Cap eval cost: ~32 microbatches/rank ≈ a few million tokens globally.
                subset_num_batches=args.eval_subset_batches,
                data=DataConfig(
                    num_workers=2,
                    drop_last=True,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True,
                    paths=val_paths,
                    memmap_dtype="uint32",
                    seed=(args.data_seed if args.data_seed is not None else args.seed) + 1,
                ),
            )
        )

    meta = {
        "model": "370M",
        "non_embedding_params": MODEL_SIZE,
        "length_tokens": length_tokens,
        "global_batch_size_sequences": gbs,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "device_microbatch": mbs,
        "lr": lr,
        "t_warmup": t_warmup,
        "t_decay": t_decay,
        "scheduler": "cos_with_warmup_and_decay",
        "save_interval_unsharded": args.save_interval,
        "seed": args.seed,
        "paths": len(paths),
        "val_paths": len(val_paths),
        "eval_interval": args.eval_interval if val_paths else None,
        "eval_subset_batches": args.eval_subset_batches if val_paths else None,
        "flash_attention": os.environ.get("OLMO_FLASH_ATTENTION", "1") == "1",
        "activation_checkpointing": os.environ.get("OLMO_ACTIVATION_CHECKPOINTING", "0"),
        "fused_loss": os.environ.get("OLMO_FUSED_LOSS", "1") == "1",
    }
    Path(args.progress_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.progress_dir) / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (Path(args.progress_dir) / "total_steps.txt").write_text(str(total_steps) + "\n")

    wandb_cfg = None
    # FarmShare path: W&B is opt-in only (WANDB_ENABLE=1 + WANDB_API_KEY).
    if os.environ.get("WANDB_ENABLE", "").lower() in {"1", "true", "yes"}:
        if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_DISABLED", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            wandb_cfg = WandbConfig(
                project=os.environ.get("WANDB_PROJECT", "edullm-olmo-ladder"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                group=os.environ.get("WANDB_GROUP", args.name),
                name=os.environ.get("WANDB_NAME", args.name),
                tags=["olmo-ladder", "370m", "30b", "farmshare", "val-holdout"],
                log_artifacts=False,
                rank_zero_only=True,
                log_interval=int(os.environ.get("WANDB_LOG_INTERVAL", "1")),
            )
            meta["wandb"] = {
                "project": wandb_cfg.project,
                "entity": wandb_cfg.entity,
                "group": wandb_cfg.group,
                "name": wandb_cfg.name,
            }
            (Path(args.progress_dir) / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            log.info("W&B enabled: project=%s entity=%s name=%s", wandb_cfg.project, wandb_cfg.entity, wandb_cfg.name)
        else:
            log.warning("WANDB_ENABLE set but API key missing/disabled; training without W&B")
    else:
        log.info("W&B disabled (FarmShare-only run)")

    return TrainConfig(
        run_name=args.name,
        seed=args.seed,
        wandb=wandb_cfg,
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
        save_interval_unsharded=args.save_interval,
        save_num_unsharded_checkpoints_to_keep=-1,
        save_interval=None,
        load_path=load_path,
        try_load_latest_save=True,
        eval_on_load=bool(val_paths),
        sharded_checkpointer=ShardedCheckpointerType.olmo_core,
        device_train_microbatch_size=mbs,
        precision="amp_bf16",
        distributed_strategy=DistributedStrategy.ddp,
        # Avoid materializing full float32 logits (mbs*seq*vocab*4 ≈ 6.1GiB at mbs=4).
        fused_loss=os.environ.get("OLMO_FUSED_LOSS", "1") == "1",
        gen1_gc_interval=2,
        max_grad_norm=1.0,
        speed_monitor=SpeedMonitorConfig(window_size=1),
        eval_interval=args.eval_interval if val_paths else 10**9,
        device_eval_batch_size=mbs,
        eval_subset_num_batches=args.eval_subset_batches,
        evaluators=evaluators,
        activation_checkpointing=activation_checkpointing_from_env(),
        data=DataConfig(
            num_workers=8,
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
    ap.add_argument("--device-batch-size", type=int, default=8)
    ap.add_argument("--batch-size-divisor", type=int, default=32)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--eval-interval", type=int, default=500, help="Steps between val loss evaluations")
    ap.add_argument(
        "--eval-subset-batches",
        type=int,
        default=32,
        help="Max eval microbatches per rank (keeps val cheap for a loss curve)",
    )
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--load-path", type=str, default=None)
    args = ap.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    # Progress file hook via logging filter on rank 0
    progress_path = Path(args.progress_dir) / "progress.log"
    total_steps = int((Path(args.progress_dir) / "total_steps.txt").read_text()) if (Path(args.progress_dir) / "total_steps.txt").exists() else None

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
        "Using CosWithWarmupAndDecay: t_warmup=%s t_decay=%s alpha_f=0.1 flash_attn=%s activation_ckpt=%s fused_loss=%s",
        cfg.scheduler.t_warmup,
        _LADDER_T_DECAY,
        cfg.model.flash_attention,
        cfg.activation_checkpointing,
        cfg.fused_loss,
    )

    class ProgressHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            # OLMo trainer logs often include step= and loss=
            if "step=" not in msg and "Step " not in msg:
                return
            try:
                step = None
                loss = None
                for part in msg.replace(",", " ").split():
                    if part.startswith("step="):
                        step = int(float(part.split("=", 1)[1]))
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
                # also maintain a single-line status file
                status = {
                    "step": step,
                    "total_steps": total_steps,
                    "loss": loss,
                    "pct": round(100.0 * step / max(total_steps, 1), 4),
                }
                (Path(args.progress_dir) / "progress.json").write_text(json.dumps(status) + "\n")
            except Exception:
                return

    if get_local_rank() == 0 or True:
        # only rank 0 should write; check after dist init
        from olmo.torch_util import get_global_rank

        if get_global_rank() == 0:
            progress_path.write_text(f"0/{total_steps} starting\n")
            logging.getLogger().addHandler(ProgressHandler())
            logging.getLogger("trainer").addHandler(ProgressHandler())
            logging.getLogger("train").addHandler(ProgressHandler())

    # Import ladder train main AFTER scheduler patch.
    ladder_root = Path(os.environ.get("OLMO_LADDER_ROOT", ""))
    if ladder_root:
        sys.path.insert(0, str(ladder_root / "src" / "ladder"))
    from train import main as train_main  # type: ignore
    import train as ladder_train  # type: ignore

    ladder_train.build_scheduler = build_scheduler  # belt-and-suspenders

    train_main(cfg)


if __name__ == "__main__":
    main()
