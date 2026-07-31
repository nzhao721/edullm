#!/usr/bin/env python3
"""Train one mixture with the exact DataDecide 60M configuration on a single GPU.

Model geometry, batch size, learning rate and schedule are taken verbatim from
DataDecide (allenai/DataDecide-*-60M ``config.json`` plus Appendix Table 2):
``d_model=384``, 16 layers, 12 heads, ``mlp_ratio=8``, ``seq_len=2048``, global
batch 96 sequences, LR 5.8e-3, untied LM head, ``norm_after=False``,
``rope_theta=10000``, RMSNorm, SwiGLU. The one deliberate deviation is the
embedding matrix: the corpus is already tokenized with dolma2, so the
vocabulary is dolma2's 100,352 rows instead of DataDecide's 50,304. Every shape
that defines the 60M *body* is untouched (see ``--help`` on ``--weight-tying``
for the one knob that changes this trade-off).

Training data is a domain-stratified stream over a **working pool staged from
published+validated** ``s3://edullm-data/pretrain/olmo-127b`` (``dataset_paths`` /
``resolve_latest``). ``--pool-dir`` must carry ``edullm_data_source.json`` from
``stage_working_pool_from_edullm_data.py`` — orphan FarmShare/laptop pools and
legacy ``s3://edullm-datasets/`` are refused. Scratch under ``--save-folder`` is
ephemeral; durable checkpoints require ``--remote-save-folder`` / ``RESULTS_S3``
(unless ``--allow-local-only``).

**W&B.** When ``--wandb-mode online|offline`` and ``WANDB_API_KEY`` are set
(FarmShare: ``wandb-session.env``), OLMo logs train + in-run task-loss metrics
to project ``mixlaw`` and uploads checkpoint artifacts. S3 durable sinks stay
required for ephemeral scratch unless ``--allow-local-only``.

Evaluation is **task loss**, the OLMo-ladder metric: bits-per-byte of the gold
continuation on the OLMES 5-shot RC suite (Bhagia et al., arXiv:2412.04403).
There is no LM validation loss and no held-out corpus split — the full budgeted
token stream is available for training.

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
from typing import Any, Optional

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
    DEFAULT_RESULTS_S3,
    DOMAINS,
    EDULLM_DATA_DATASET_ID,
    POOL_PROVENANCE_NAME,
    patch_torch_load_for_olmo_checkpoints,
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
from mixlaw_wandb import (
    add_wandb_args,
    finish_wandb,
    init_wandb,
    wandb_enabled,
    wandb_log,
    wandb_log_checkpoint,
    wandb_log_eval,
    wandb_upload_existing,
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


def load_pool_provenance(pool_dir: Path, dataset_id: str) -> dict:
    """Require a pool staged from published edullm-data (not orphan scratch/local)."""
    path = pool_dir / POOL_PROVENANCE_NAME
    if not path.is_file():
        raise SystemExit(
            f"missing {path}: refuse orphan local/scratch pool without provenance. "
            f"Stage from edullm-data first "
            f"(stage_working_pool_from_edullm_data.py --dataset-id {dataset_id})"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    got = payload.get("dataset_id")
    if got != dataset_id:
        raise SystemExit(
            f"pool provenance dataset_id={got!r} does not match --dataset-id {dataset_id!r}"
        )
    if payload.get("data_bucket") not in (None, "edullm-data"):
        raise SystemExit(
            f"pool provenance data_bucket={payload.get('data_bucket')!r} is not edullm-data"
        )
    uri = str(payload.get("edullm_data_uri") or payload.get("uri") or "")
    if "edullm-datasets" in uri:
        raise SystemExit(
            f"pool provenance uri={uri!r} still points at legacy edullm-datasets; re-stage"
        )
    return payload


def resolve_dataset_version(dataset_id: str, pinned: Optional[str], provenance: dict) -> str:
    """Prefer an explicit pin, then pool provenance, then live catalog resolve_latest."""
    if pinned:
        return pinned
    from_prov = provenance.get("dataset_version")
    if from_prov:
        return str(from_prov)
    try:
        from edullm_data.read import resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:
        raise SystemExit(
            "edullm-data is required to resolve dataset versions when the pool "
            "provenance has no dataset_version"
        ) from exc
    ver = resolve_latest(dataset_id, s3=Boto3S3.default())
    if not ver:
        raise SystemExit(f"no published versions for {dataset_id}")
    return ver


def resolve_remote_save_folder(args: argparse.Namespace) -> Optional[str]:
    """Durable checkpoint sink (S3). Scratch under --save-folder is ephemeral."""
    remote = args.remote_save_folder or os.environ.get("REMOTE_SAVE_FOLDER")
    if remote:
        return remote.rstrip("/")
    results = os.environ.get("RESULTS_S3", "").strip()
    if results:
        return f"{results.rstrip('/')}/{args.name}/checkpoints"
    allow_local = os.environ.get("ALLOW_LOCAL_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if allow_local or args.allow_local_only:
        return None
    raise SystemExit(
        "durable saves required on ephemeral scratch: pass --remote-save-folder "
        f"(e.g. {DEFAULT_RESULTS_S3}/<mix>/checkpoints), set RESULTS_S3, "
        "or explicitly ALLOW_LOCAL_ONLY=1 / --allow-local-only for smoke tests"
    )


def build_config(
    args: argparse.Namespace,
    *,
    provenance: dict,
    dataset_version: str,
    remote_save_folder: Optional[str],
) -> TrainConfig:
    global _LADDER_T_DECAY

    from recipe_data import load_mix_weights

    weights, mix_meta = load_mix_weights(Path(args.mix_weights_json))
    length_tokens = int(args.length_tokens or mix_meta.get("length_tokens") or mix_meta["budget_tokens"])
    stream_seed = int(mix_meta.get("stream_seed", mix_meta.get("recipe_seed", args.seed)))
    paths = ["<domain_stream>"]

    mbs = args.device_batch_size
    if GLOBAL_BATCH_SEQS % mbs != 0:
        raise SystemExit(f"global batch {GLOBAL_BATCH_SEQS} not divisible by microbatch {mbs}")

    length_tokens = int(length_tokens)
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
        "data_mode": "domain_stratified_stream",
        "dataset_id": args.dataset_id,
        "dataset_version": dataset_version,
        "data_bucket": "edullm-data",
        "edullm_data_uri": f"s3://edullm-data/{args.dataset_id}/{dataset_version}/",
        "pool_dir": str(args.pool_dir),
        "pool_provenance": provenance,
        "mix_weights_json": str(args.mix_weights_json),
        "stream_seed": stream_seed,
        "domain_weights": weights,
        "domains": list(DOMAINS),
        "save_folder": str(args.save_folder),
        "remote_save_folder": remote_save_folder,
        "ephemeral_scratch": True,
        "flash_attention": os.environ.get("OLMO_FLASH_ATTENTION", "0") == "1",
        "fused_loss": os.environ.get("OLMO_FUSED_LOSS", "0") == "1",
        "wandb": None,
    }
    progress = Path(args.progress_dir)
    progress.mkdir(parents=True, exist_ok=True)

    # SmolLM-style: one explicit wandb.init in main(); keep TrainConfig.wandb=None
    # to avoid a second OLMo-managed run.
    if wandb_enabled(args, is_main=True):
        meta["wandb"] = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group or args.name,
            "name": args.wandb_run_name or args.name,
            "mode": args.wandb_mode,
            "enabled": True,
        }
    else:
        meta["wandb"] = {"mode": args.wandb_mode, "enabled": False}

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
        remote_save_folder=remote_save_folder,
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
        fused_loss=os.environ.get("OLMO_FUSED_LOSS", "0") == "1",
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
    keeps the whole curve available for the step-law fit. When a W&B run handle is
    attached, the same points are mirrored under ``eval/bpb/<label>``.
    """

    def __init__(
        self,
        progress_dir: Path,
        total_steps: int,
        *,
        wb_run: Any = None,
    ) -> None:
        super().__init__()
        self.progress_dir = progress_dir
        self.total_steps = total_steps
        self.curve_path = progress_dir / "task_loss.jsonl"
        self.progress_path = progress_dir / "progress.json"
        self.wb_run = wb_run

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
                if self.wb_run is not None:
                    wandb_log_eval(
                        self.wb_run,
                        {"task_loss_bpb": task_losses},
                        step=step,
                    )
            if train_loss is not None and self.wb_run is not None:
                wandb_log(self.wb_run, {"train/loss": train_loss}, step=step)
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
    patch_torch_load_for_olmo_checkpoints()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="Run name, e.g. mix07")
    ap.add_argument(
        "--mix-weights-json",
        required=True,
        help="Recipe sidecar from prepare_mixlaw_pilot_data.py (domain-stratified stream)",
    )
    ap.add_argument(
        "--pool-dir",
        required=True,
        help=(
            "Working pool root staged from edullm-data "
            f"(must contain {POOL_PROVENANCE_NAME})"
        ),
    )
    ap.add_argument(
        "--dataset-id",
        default=EDULLM_DATA_DATASET_ID,
        help=f"Published edullm-data dataset id (default {EDULLM_DATA_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin edullm-data version; default uses pool provenance or resolve_latest",
    )
    ap.add_argument(
        "--save-folder",
        required=True,
        help="Local/scratch checkpoint dir (ephemeral). Durable copy via --remote-save-folder.",
    )
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument(
        "--remote-save-folder",
        default=None,
        help=(
            "S3 prefix for durable checkpoints (OLMo remote_save_folder). "
            f"Default: $RESULTS_S3/<name>/checkpoints or {DEFAULT_RESULTS_S3}/<name>/checkpoints "
            "when RESULTS_S3 is set by the launcher."
        ),
    )
    ap.add_argument(
        "--allow-local-only",
        action="store_true",
        help="Permit training without a durable S3 sink (smoke tests only)",
    )
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=None,
        help="Defaults to mix_weights.json budget_tokens (tpp=5 → ~285M / 1451 steps)",
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
    add_wandb_args(ap)
    args = ap.parse_args()

    # Clear hard-disable leftovers if a wrapper exported WANDB_DISABLED=1 by mistake
    # while also requesting online mode via CLI.
    if args.wandb_mode != "disabled":
        os.environ.pop("WANDB_DISABLED", None)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    pool_dir = Path(args.pool_dir)
    provenance = load_pool_provenance(pool_dir, args.dataset_id)
    dataset_version = resolve_dataset_version(args.dataset_id, args.dataset_version, provenance)
    remote_save_folder = resolve_remote_save_folder(args)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as exc:
        print(f"failed to set multiprocessing start method: {exc}")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    from recipe_data import load_mix_weights
    from olmo_domain_stream_patch import apply_olmo_domain_stream_patch

    weights, mix_meta = load_mix_weights(Path(args.mix_weights_json))
    length_tokens = int(args.length_tokens or mix_meta.get("length_tokens") or mix_meta["budget_tokens"])
    stream_seed = int(mix_meta.get("stream_seed", mix_meta.get("recipe_seed", args.seed)))
    apply_olmo_domain_stream_patch(
        args.pool_dir,
        weights,
        length_tokens=length_tokens,
        seed=stream_seed,
    )

    cfg = build_config(
        args,
        provenance=provenance,
        dataset_version=dataset_version,
        remote_save_folder=remote_save_folder,
    )

    # Inject the ladder WSD schedule before importing ladder's train.py, which
    # does `from olmo.optim import build_scheduler` at module scope.
    import olmo.optim as olmo_optim

    olmo_optim.build_scheduler = build_scheduler  # type: ignore[assignment]

    progress_dir = Path(args.progress_dir)
    total_steps = int((progress_dir / "total_steps.txt").read_text())
    log.info(
        "DataDecide-60M %s | steps=%d lr=%.4g t_warmup=%d t_decay=%d labels=%d flash=%s wandb=%s",
        args.name,
        total_steps,
        LEARNING_RATE,
        cfg.scheduler.t_warmup,
        _LADDER_T_DECAY,
        len(cfg.evaluators),
        cfg.model.flash_attention,
        args.wandb_mode,
    )

    wb_run = None
    if get_global_rank() == 0:
        meta = json.loads((progress_dir / "run_meta.json").read_text(encoding="utf-8"))
        # Explicit SDK run (SmolLM-style); TrainConfig.wandb stays None to avoid a second init.
        wb_run = init_wandb(
            args,
            meta,
            id_dir=progress_dir,
            is_main=True,
            tags=["mixlaw", "datadecide-60m", args.name],
            alert_title="mixlaw datadecide-60m started",
        )
        if wb_run is not None and args.wandb_upload_existing:
            wandb_upload_existing(
                wb_run,
                checkpoints_root=Path(args.save_folder),
                progress_dir=progress_dir,
                tokens_per_step=TOKENS_PER_STEP,
            )
        handler = TaskLossHandler(progress_dir, total_steps, wb_run=wb_run)
        for name in ("", "trainer", "train", "olmo.train"):
            logging.getLogger(name).addHandler(handler)

    ladder_root = Path(os.environ.get("OLMO_LADDER_ROOT", ""))
    if str(ladder_root):
        sys.path.insert(0, str(ladder_root / "src" / "ladder"))
    import train as ladder_train  # type: ignore

    ladder_train.build_scheduler = build_scheduler  # belt and suspenders
    try:
        ladder_train.main(cfg)
    finally:
        if get_global_rank() == 0 and wb_run is not None:
            # Upload final unsharded checkpoint + progress curve if present.
            save_root = Path(args.save_folder)
            latest = None
            for cand in sorted(save_root.glob("step*-unsharded"), key=lambda p: p.name):
                latest = cand
            if latest is not None:
                try:
                    step = int(latest.name.split("-")[0].replace("step", ""))
                except ValueError:
                    step = total_steps
                wandb_log_checkpoint(
                    wb_run,
                    latest,
                    step=step,
                    tokens_seen=step * TOKENS_PER_STEP,
                )
            wandb_upload_existing(wb_run, progress_dir=progress_dir)
            finish_wandb(wb_run)


if __name__ == "__main__":
    main()
