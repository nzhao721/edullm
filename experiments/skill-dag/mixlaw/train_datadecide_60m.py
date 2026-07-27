#!/usr/bin/env python3
"""Train one mixture with the exact DataDecide 60M configuration on a single GPU.

Model geometry, batch size, learning rate and schedule are taken verbatim from
DataDecide (allenai/DataDecide-*-60M ``config.json`` plus Appendix Table 2):
``d_model=384``, 16 layers, 12 heads, ``mlp_ratio=8``, ``seq_len=2048``, global
batch 96 sequences, LR 5.8e-3, untied LM head, ``norm_after=False``,
``rope_theta=10000``, RMSNorm, SwiGLU. The one deliberate deviation is the
embedding matrix: the RegMix corpus is already tokenized with dolma2, so the
vocabulary is dolma2's 100,352 rows instead of DataDecide's 50,304. Every shape
that defines the 60M *body* is untouched (see ``--help`` on ``--weight-tying``
for the one knob that changes this trade-off).

Evaluation is **task loss**, the OLMo-ladder metric: bits-per-byte of the gold
continuation on the OLMES 5-shot RC suite (Bhagia et al., arXiv:2412.04403).
There is no LM validation loss and no held-out corpus split — the full 10B RegMix
corpus is available for training slices.

Two evaluation cadences, because a full pass over all 20 ladder bpb labels costs
more forward tokens than a short probe run costs training tokens:
  * during training, ``CURVE_TASK_LOSS_LABELS`` on a capped number of batches,
    which produces the loss curve the step law is fitted to;
  * after training, the full suite once, via ``eval_task_loss.py``.

One process owns one GPU: at d_model=384 a single run cannot fill a B200, so
throughput comes from running many mixtures concurrently rather than sharding one.
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

# Hard-disable W&B before importing olmo, which reads these at import time.
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
    DataConfig,
    DistributedStrategy,
    EvaluatorConfig,
    EvaluatorType,
    InstanceFilterConfig,
    ShardedCheckpointerType,
    SpeedMonitorConfig,
)
from olmo.optim import Scheduler, build_scheduler as _olmo_build_scheduler
from olmo.torch_util import get_global_rank, get_local_rank
from olmo.util import add_cached_path_clients, find_latest_checkpoint, prepare_cli_environment

from mixlaw_common import (
    CURVE_TASK_LOSS_LABELS,
    D_MODEL,
    DATADECIDE_MODEL_SIZE,
    EMBEDDING_SIZE,
    EOS_TOKEN_ID,
    GLOBAL_BATCH_SEQS,
    LADDER_TASK_LOSS_LABELS,
    LEARNING_RATE,
    MLP_RATIO,
    N_HEADS,
    N_LAYERS,
    PAD_TOKEN_ID,
    SEQ_LEN,
    TOKENIZER_ID,
    TOKENS_PER_STEP,
    VOCAB_SIZE,
    ladder_warmup_steps,
    normalize_eval_key,
)

log = logging.getLogger("train_datadecide_60m")

_LADDER_T_DECAY: Optional[int] = None


@dataclass
class CosWithWarmupAndDecay(Scheduler):
    """OLMo-ladder LR schedule: warmup, constant peak, cosine over the final t_decay steps."""

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


def build_model_config(weight_tying: bool) -> ModelConfig:
    """Exact DataDecide 60M, with the embedding widened to the dolma2 vocabulary."""
    return ModelConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        mlp_ratio=MLP_RATIO,
        weight_tying=weight_tying,
        alibi=False,
        alibi_bias_max=8.0,
        rope=True,
        rope_full_precision=True,
        rope_theta=10_000,
        flash_attention=os.environ.get("OLMO_FLASH_ATTENTION", "0") == "1",
        attention_dropout=0.0,
        attention_layer_norm=False,
        attention_layer_norm_with_affine=False,
        include_bias=False,
        layer_norm_type=LayerNormType.rms,
        layer_norm_with_affine=True,
        layer_norm_eps=1e-6,
        bias_for_layer_norm=False,
        activation_type=ActivationType.swiglu,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        max_sequence_length=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        init_device="cuda",
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_cutoff_factor=3,
        norm_after=False,
        precision="amp_bf16",
    )


def resolve_warmup(total_steps: int, mode: str, fraction: float) -> int:
    """Warmup steps.

    The ladder heuristic warms up over roughly one model-size worth of tokens, which
    is 290 steps at this batch size. At the default budget (5,806 steps) that is 5%
    of the run, so ``capped`` returns the exact ladder value and the cap never binds.
    The cap only matters if the budget is cut: on a 1451-step run the ladder value
    would be a third of training, which would distort the loss curve the step law is
    fitted to.
    """
    ladder = ladder_warmup_steps()
    if mode == "ladder":
        return min(ladder, max(1, total_steps - 1))
    if mode == "capped":
        return max(1, min(ladder, round(fraction * total_steps)))
    if mode == "fraction":
        return max(1, round(fraction * total_steps))
    raise SystemExit(f"unknown --warmup-mode {mode}")


def build_config(args: argparse.Namespace) -> TrainConfig:
    global _LADDER_T_DECAY

    paths = [ln.strip() for ln in Path(args.paths_file).read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No training paths in {args.paths_file}")

    mbs = args.device_batch_size
    if GLOBAL_BATCH_SEQS % mbs != 0:
        raise SystemExit(f"global batch {GLOBAL_BATCH_SEQS} not divisible by microbatch {mbs}")

    # One epoch over exactly the tokens that were materialized for this mixture.
    length_tokens = args.length_tokens
    if length_tokens is None:
        length_tokens = sum(Path(p).stat().st_size // 4 for p in paths)
    total_steps = length_tokens // TOKENS_PER_STEP
    if total_steps < 1:
        raise SystemExit(f"{length_tokens} tokens is less than one step")

    t_warmup = resolve_warmup(total_steps, args.warmup_mode, args.warmup_fraction)
    t_decay = max(1, round(args.decay_fraction * total_steps))
    _LADDER_T_DECAY = t_decay

    if args.skip_eval:
        curve_labels = []
    elif args.task_loss_labels is not None:
        curve_labels = list(args.task_loss_labels)
    elif args.full_task_suite_in_run:
        curve_labels = list(LADDER_TASK_LOSS_LABELS)
    else:
        curve_labels = list(CURVE_TASK_LOSS_LABELS)
    unknown = [lbl for lbl in curve_labels if lbl not in LADDER_TASK_LOSS_LABELS]
    if unknown:
        raise SystemExit(f"not OLMo-ladder task-loss labels: {unknown}")

    evaluators = [
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            subset_num_batches=args.eval_subset_batches,
        )
        for label in curve_labels
    ]

    meta = {
        "model": "DataDecide-60M",
        "datadecide_model_size": DATADECIDE_MODEL_SIZE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "mlp_ratio": MLP_RATIO,
        "seq_len": SEQ_LEN,
        "weight_tying": args.weight_tying,
        "vocab_size": VOCAB_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "tokenizer": TOKENIZER_ID,
        "global_batch_seqs": GLOBAL_BATCH_SEQS,
        "tokens_per_step": TOKENS_PER_STEP,
        "device_microbatch": mbs,
        "grad_accum_steps": GLOBAL_BATCH_SEQS // mbs,
        "length_tokens": length_tokens,
        "total_steps": total_steps,
        "lr": LEARNING_RATE,
        "t_warmup": t_warmup,
        "t_decay": t_decay,
        "warmup_mode": args.warmup_mode,
        "scheduler": "cos_with_warmup_and_decay",
        "eval_metric": "task_loss_bpb",
        "eval_interval": args.eval_interval,
        "eval_subset_batches": args.eval_subset_batches,
        "curve_task_labels": curve_labels,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "train_paths": len(paths),
        "flash_attention": os.environ.get("OLMO_FLASH_ATTENTION", "0") == "1",
        "wandb": None,
    }
    progress = Path(args.progress_dir)
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (progress / "total_steps.txt").write_text(f"{total_steps}\n", encoding="utf-8")

    load_path = args.load_path or find_latest_checkpoint(args.save_folder)

    return TrainConfig(
        run_name=args.name,
        seed=args.seed,
        wandb=None,
        model=build_model_config(args.weight_tying),
        ddp=DDPConfig(),
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            learning_rate=LEARNING_RATE,
            weight_decay=0.1,
            eps=1e-8,
            decay_norm_and_bias=True,
            decay_embeddings=False,
            betas=(0.9, 0.95),
            metrics_log_interval=10,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.cosine_with_warmup,
            alpha_f=0.1,
            warmup_min_lr=0.0,
            t_warmup=t_warmup,
            t_max=total_steps,
        ),
        max_duration=f"{length_tokens}T",
        global_train_batch_size=GLOBAL_BATCH_SEQS,
        tokenizer=TokenizerConfig(identifier=TOKENIZER_ID),
        save_folder=args.save_folder,
        remote_save_folder=None,
        save_overwrite=True,
        save_interval_unsharded=args.save_interval or total_steps,
        save_num_unsharded_checkpoints_to_keep=args.keep_checkpoints,
        save_interval=None,
        load_path=load_path,
        try_load_latest_save=True,
        eval_on_load=False,
        sharded_checkpointer=ShardedCheckpointerType.olmo_core,
        device_train_microbatch_size=mbs,
        precision="amp_bf16",
        # World size is 1 per mixture, so DDP is a no-op wrapper and FSDP would
        # only add collectives.
        distributed_strategy=DistributedStrategy.ddp,
        fused_loss=None,
        gen1_gc_interval=2,
        max_grad_norm=1.0,
        speed_monitor=SpeedMonitorConfig(window_size=1),
        eval_interval=args.eval_interval,
        device_eval_batch_size=args.device_eval_batch_size,
        eval_subset_num_batches=args.eval_subset_batches,
        evaluators=evaluators,
        data=DataConfig(
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            instance_filter=InstanceFilterConfig(),
            paths=paths,
            memmap_dtype="uint32",
            seed=args.data_seed if args.data_seed is not None else args.seed,
        ),
        auxiliary_loss_multiplier=1e-5,
        softmax_auxiliary_loss=True,
    )


class TaskLossHandler(logging.Handler):
    """Scrape step / train loss / task-loss bpb out of the trainer log.

    OLMo emits downstream metrics as ``eval/downstream_bpb/<label>_bpb=<value>``; for
    a ``bpb`` metric type that value *is* the task loss. Appending them to a JSONL
    keeps the whole curve available for the step-law fit without depending on W&B,
    which is disabled here because 24 concurrent runs on a shared cluster should not
    need an external service to be reachable.
    """

    def __init__(self, progress_dir: Path, total_steps: int) -> None:
        super().__init__()
        self.progress_dir = progress_dir
        self.total_steps = total_steps
        self.curve_path = progress_dir / "task_loss.jsonl"
        self.progress_path = progress_dir / "progress.json"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if "step=" not in msg and "eval/" not in msg:
                return

            step: Optional[int] = None
            train_loss: Optional[float] = None
            task_losses: dict[str, float] = {}
            for token in msg.replace(",", " ").split():
                if "=" not in token:
                    continue
                key, _, raw = token.partition("=")
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if key == "step":
                    step = int(value)
                elif key in ("train/CrossEntropyLoss", "loss"):
                    train_loss = value
                else:
                    label = normalize_eval_key(key)
                    if label is not None:
                        task_losses[label] = value

            if step is None:
                return
            if task_losses:
                with self.curve_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": step, "task_loss_bpb": task_losses}) + "\n")
            self.progress_path.write_text(
                json.dumps(
                    {
                        "step": step,
                        "total_steps": self.total_steps,
                        "pct": round(100.0 * step / max(self.total_steps, 1), 3),
                        "train_loss": train_loss,
                        "task_loss_bpb": task_losses or None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:  # a logging handler must never break training
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="Run name, e.g. mix07")
    ap.add_argument("--paths-file", required=True, help="paths_train.txt for this mixture")
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=None,
        help="Defaults to the exact token count of the mixture slices (one epoch)",
    )
    ap.add_argument(
        "--device-batch-size",
        type=int,
        default=32,
        help="Per-GPU microbatch; must divide global batch 96. On B200+dolma2, 32 is safe; 48+ OOMs",
    )
    ap.add_argument("--device-eval-batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument(
        "--eval-interval",
        type=int,
        default=120,
        help="Steps between task-loss evaluations (~12 curve points on a 1451-step run)",
    )
    ap.add_argument(
        "--eval-subset-batches",
        type=int,
        default=4,
        help="Max eval batches per label per evaluation; keeps the in-run curve cheap",
    )
    ap.add_argument(
        "--task-loss-labels",
        nargs="*",
        default=None,
        help=f"Override the in-run curve labels (subset of {len(LADDER_TASK_LOSS_LABELS)} ladder bpb labels)",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Disable all in-run downstream evaluators (for throughput smoke tests)",
    )
    ap.add_argument(
        "--full-task-suite-in-run",
        action="store_true",
        help="Evaluate all 20 ladder bpb labels during training instead of the cheap curve subset",
    )
    ap.add_argument("--warmup-mode", choices=("capped", "ladder", "fraction"), default="capped")
    ap.add_argument("--warmup-fraction", type=float, default=0.1)
    ap.add_argument("--decay-fraction", type=float, default=0.1, help="OLMo-ladder uses 0.1")
    ap.add_argument(
        "--weight-tying",
        action="store_true",
        help=(
            "Tie the embedding and LM head. DataDecide sets this False; tying removes "
            "38.5M dolma2-vocabulary parameters that DataDecide's 50k vocabulary never had"
        ),
    )
    ap.add_argument("--save-interval", type=int, default=None, help="Default: only a final checkpoint")
    ap.add_argument("--keep-checkpoints", type=int, default=1)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--data-seed", type=int, default=None)
    ap.add_argument("--load-path", default=None)
    args = ap.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as exc:
        print(f"failed to set multiprocessing start method: {exc}")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    cfg = build_config(args)

    # Inject the ladder WSD schedule before importing ladder's train.py, which
    # does `from olmo.optim import build_scheduler` at module scope.
    import olmo.optim as olmo_optim

    olmo_optim.build_scheduler = build_scheduler  # type: ignore[assignment]

    progress_dir = Path(args.progress_dir)
    total_steps = int((progress_dir / "total_steps.txt").read_text())
    log.info(
        "DataDecide-60M %s | steps=%d lr=%.4g t_warmup=%d t_decay=%d labels=%d flash=%s",
        args.name,
        total_steps,
        LEARNING_RATE,
        cfg.scheduler.t_warmup,
        _LADDER_T_DECAY,
        len(cfg.evaluators),
        cfg.model.flash_attention,
    )

    if get_global_rank() == 0:
        handler = TaskLossHandler(progress_dir, total_steps)
        for name in ("", "trainer", "train", "olmo.train"):
            logging.getLogger(name).addHandler(handler)

    ladder_root = Path(os.environ.get("OLMO_LADDER_ROOT", ""))
    if str(ladder_root):
        sys.path.insert(0, str(ladder_root / "src" / "ladder"))
    import train as ladder_train  # type: ignore

    ladder_train.build_scheduler = build_scheduler  # belt and suspenders
    ladder_train.main(cfg)


if __name__ == "__main__":
    main()
