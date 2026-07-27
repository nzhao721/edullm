#!/usr/bin/env python3
"""Train OLMo-2-370M on RefHQ RegMix 5.5B (reference arm).

Architecture matches the finished GPU7 REL+EMA job
``rel-ema-5b-scratch-v1`` / ``OLMo-2-370M-scratch`` (NOT olmo3_370M / SWA):

  * ``TransformerConfig.olmo2_370M`` — d_model=1024, n_layers=16, n_heads=16,
    reordered_norm, gated SiLU FFN (hidden via llama_like ×1.5 → 4096),
    QK-RMSNorm, RoPE θ=500000, full attention (no sliding window)
  * vocab_size=100352 (dolma2), ~371M non-embedding / ~474M total
  * sequence_length=2048
  * global_batch_size=4_194_304, rank_microbatch_size=65_536 (32 seq; grad_accum=64)
  * ``CosWithWarmup``, peak LR 4e-4, warmup 24 (same as that REL train block)
  * ``compile_model=True`` + ``torch.set_float32_matmul_precision("high")``
  * attn backend ``torch``

Dataset is **only** RefHQ (``s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/``).
No evals / W&B. Do not confuse with ``olmo3-370m/run-10b-equal``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import torch

# Hard-disable W&B before importing olmo_core callbacks.
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

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger("train_olmo2_370m_refhq")

SEQ_LEN = 2048
# Match REL+EMA GPU7 job (run_5b.yaml).
DEFAULT_GLOBAL_BATCH_TOKENS = 4_194_304
DEFAULT_RANK_MICROBATCH_TOKENS = 65_536  # 32 × 2048
DEFAULT_LR = 4.0e-4
DEFAULT_WARMUP_STEPS = 24
DEFAULT_SEED = 6198
DEFAULT_TOKEN_BUDGET = 5_514_030_574
MODEL_SIZE_FOR_LR = 371_262_464
CONFIG_NAME = "OLMo-2-370M-scratch"


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = DEFAULT_SEED
    load_path: Optional[str] = None


def resolve_attn_backend() -> AttentionBackendName:
    prefer = os.environ.get("OLMO_ATTN_BACKEND", "torch").strip().lower()
    if prefer in ("torch", "sdpa", "eager"):
        return AttentionBackendName.torch
    if prefer in ("flash_2", "flash", "flash2"):
        try:
            import flash_attn  # noqa: F401

            return AttentionBackendName.flash_2
        except Exception as e:
            log.warning("OLMO_ATTN_BACKEND=%s but flash_attn unavailable (%s); using torch", prefer, e)
            return AttentionBackendName.torch
    try:
        return AttentionBackendName(prefer)
    except Exception:
        log.warning("Unknown OLMO_ATTN_BACKEND=%s; using torch", prefer)
        return AttentionBackendName.torch


def build_olmo2_370m() -> TransformerConfig:
    """REL+EMA reference architecture: full-attn olmo2_370M (no SWA)."""
    return TransformerConfig.olmo2_370M(
        vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
        attn_backend=resolve_attn_backend(),
    )


def read_paths(paths_file: Path) -> List[str]:
    paths = [ln.strip() for ln in paths_file.read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No training paths in {paths_file}")
    return paths


def build_config(opts: argparse.Namespace) -> ExperimentConfig:
    tokenizer = TokenizerConfig.dolma2()
    model_config = build_olmo2_370m()

    paths = read_paths(Path(opts.paths_file))
    dataset_config = NumpyFSLDatasetConfig(
        paths=paths,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer,
        work_dir=opts.work_dir,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
        num_workers=opts.num_workers,
    )

    lr = opts.lr if opts.lr is not None else DEFAULT_LR

    try:
        scheduler = CosWithWarmup(warmup_steps=opts.warmup_steps, alpha_f=opts.alpha_f)
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=opts.warmup_steps)
        if hasattr(scheduler, "alpha_f"):
            scheduler.alpha_f = opts.alpha_f

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=opts.compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )

    tokens_per_step = opts.global_batch_size
    total_steps = opts.token_budget // tokens_per_step
    save_interval = opts.save_interval
    # Permanent ladder: every save_interval through the last multiple strictly
    # before the final step; post_train writes the true end. Keep all.
    # No ephemeral rotation — every interval save is permanent.
    last_interval_step = (total_steps // save_interval) * save_interval
    permanent_save_steps = list(range(save_interval, last_interval_step, save_interval))

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(opts.token_budget),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,
                fixed_steps=permanent_save_steps,
                ephemeral_save_interval=None,
                pre_train_checkpoint=False,
                save_async=False,
                max_checkpoints=None,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    if get_rank() == 0:
        progress = Path(opts.progress_dir)
        progress.mkdir(parents=True, exist_ok=True)
        meta = {
            "architecture": "olmo_core.TransformerConfig.olmo2_370M",
            "config_name": CONFIG_NAME,
            "block": "reordered_norm",
            "mlp": "silu_ffn_hidden_4096",
            "sliding_window": False,
            "qk_norm": True,
            "rope_theta": 500_000,
            "d_model": 1024,
            "n_layers": 16,
            "n_heads": 16,
            "framework": "olmo_core (edu-llm/OLMo-core)",
            "scheduler": "CosWithWarmup",
            "warmup_steps": opts.warmup_steps,
            "alpha_f": opts.alpha_f,
            "lr": lr,
            "global_batch_tokens": opts.global_batch_size,
            "rank_microbatch_tokens": opts.rank_microbatch_size,
            "device_microbatch_sequences": opts.rank_microbatch_size // opts.sequence_length,
            "sequence_length": opts.sequence_length,
            "token_budget": opts.token_budget,
            "total_steps": total_steps,
            "save_interval": save_interval,
            "permanent_save_steps": permanent_save_steps,
            "final_checkpoint": "post_train",
            "max_checkpoints": None,
            "compile_model": opts.compile_model,
            "attn_backend": str(resolve_attn_backend()),
            "seed": opts.seed,
            "dataset": "s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/",
            "reference_job": "rel-ema-5b-scratch-v1 (arch/batch/seq/lr; RefHQ data)",
            "paths": len(paths),
            "evals": False,
            "model_size_non_embedding": MODEL_SIZE_FOR_LR,
        }
        (progress / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress / "total_steps.txt").write_text(str(total_steps) + "\n")
        log.info(
            "RefHQ reference: olmo2_370M (%s) CosWithWarmup warmup=%d alpha_f=%s lr=%.6g "
            "steps=%d mbs_seqs=%d seq=%d compile=%s",
            CONFIG_NAME,
            opts.warmup_steps,
            opts.alpha_f,
            lr,
            total_steps,
            opts.rank_microbatch_size // opts.sequence_length,
            opts.sequence_length,
            opts.compile_model,
        )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=opts.seed,
        load_path=opts.load_path,
    )


def main(opts: argparse.Namespace) -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    prepare_training_environment()
    try:
        cfg = build_config(opts)
        seed_all(cfg.init_seed)

        model = cfg.model.build(init_device="cuda")
        train_module = cfg.train_module.build(model)
        dataset = cfg.dataset.build()
        data_loader = cfg.data_loader.build(
            dataset, dp_process_group=train_module.dp_process_group
        )
        trainer = cfg.trainer.build(train_module, data_loader)

        if "config_saver" in trainer.callbacks:
            try:
                cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = (
                    cfg.as_config_dict()
                )
            except Exception:
                pass

        if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and cfg.load_path:
            log.info("No checkpoint in save folder; loading from %s", cfg.load_path)
            trainer.load_checkpoint(cfg.load_path, load_trainer_state=False)

        trainer.fit()
    finally:
        teardown_training_environment()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="Run name")
    ap.add_argument("--paths-file", required=True, help="RefHQ train memmap paths list")
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--work-dir", default=None, help="olmo_core dataset work dir (default: progress-dir)")
    ap.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    ap.add_argument("--sequence-length", type=int, default=SEQ_LEN)
    ap.add_argument(
        "--global-batch-size",
        type=int,
        default=DEFAULT_GLOBAL_BATCH_TOKENS,
        help="Global batch size IN TOKENS (4194304)",
    )
    ap.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=DEFAULT_RANK_MICROBATCH_TOKENS,
        help="Per-rank microbatch IN TOKENS (65536 = 32×2048; grad_accum=64)",
    )
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--alpha-f", type=float, default=0.1)
    ap.add_argument("--save-interval", type=int, default=125, help="Permanent checkpoint every N steps")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--load-path", type=str, default=None)
    ap.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile (default: on)",
    )
    ap.add_argument("--dry-run", action="store_true")
    opts = ap.parse_args(argv)
    opts.work_dir = opts.work_dir or opts.progress_dir
    if opts.global_batch_size % opts.sequence_length != 0:
        ap.error("--global-batch-size must be a multiple of --sequence-length")
    if opts.rank_microbatch_size % opts.sequence_length != 0:
        ap.error("--rank-microbatch-size must be a multiple of --sequence-length")
    if opts.global_batch_size % opts.rank_microbatch_size != 0:
        ap.error("--global-batch-size must be divisible by --rank-microbatch-size")
    return opts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.dry_run:
        prepare_training_environment()
        try:
            cfg = build_config(args)
            if get_rank() == 0:
                print(cfg)
                print("lr", args.lr)
                print("steps", args.token_budget // args.global_batch_size)
        finally:
            teardown_training_environment()
        sys.exit(0)
    main(args)
