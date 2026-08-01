#!/usr/bin/env python3
"""BLADE pretraining on RegMix 10B (token-selection arm).

Architecture / train stack match RefHQ CE (``reference/train_olmo3_370m_refhq.py``):

  * ``TransformerConfig.olmo2_370M`` (full attn, no SWA)
  * ``TransformerTrainModule`` — HSDP bf16, SkipStepAdamW, CosWithWarmup,
    ``compile_model=True``, ``z_loss_multiplier=1e-5``
  * sequence=2048, global_batch=4_194_304, rank_microbatch=65_536
  * peak LR ``4e-4``, warmup 24, ``alpha_f=0.1``

Locked BLADE schedule (do not change without a new ``run_id``):

  * ``total_steps = 2360`` (``9900000000 // 4_194_304``; one-epoch matrix budget)
  * ``tau=375``, ``K=75``, ``gamma=0.6``, ``lambda_pen=1.0``
  * ``blade_start=500`` — steps 0..499 proxy-only full CE
  * syncs at **exactly** 500, 875, 1250, 1625, 2000 then hold ref through end
  * selection score ``L_ref − L_proxy``, keep top-γ after blade_start
  * train: ``pretrain/regmix-10b``; val/HQ for K updates: ``pretrain/refhq-regmix-5p5b``
    (both from ``s3://edullm-data/`` via ``resolve_latest`` / ``dataset_paths``;
    stage with ``prepare_blade_data.py`` into job scratch)

Ephemeral-machine contract:

  * Scratch starts empty and may be wiped after the job — stage data each run.
  * Do **not** assume FarmShare/laptop corpora, old run dirs, or local venvs/ckpts.
  * Never read ``s3://edullm-datasets/``.
  * Permanent artifacts (checkpoints, progress, task_loss JSON) write under
    job-scoped ``--save-folder`` / ``--progress-dir`` and upload to W&B.

Permanent checkpoints store **proxy (+optim) and dynamic reference** (post-K at sync
steps; ``null`` before first sync). Resume loads both — **never** re-sync
``ref ← proxy`` mid-episode. Cross-job resume restores ``--wandb-resume-artifact``.

On every permanent save, rank 0 may trigger async ``task_loss_bpb`` eval on **proxy**
weights (``token_selection.olmo_ext.task_loss_hook``; needs eval script on PATH /
``TASK_LOSS_EVAL_SCRIPT``, or disable with ``TASK_LOSS_EVAL=0``).

Launch via ``torchrun`` (1..N GPUs) after ``prepare_blade_data.py`` stages shards.
Production online W&B checkpoint uploads are fail-closed on every permanent save.

W&B: project ``token-selection``, group ``blade`` (SmolLM2-style soft-skip without API key).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# Shared package lives next to this arm dir.
_TS_ROOT = Path(__file__).resolve().parents[1]
if str(_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TS_ROOT))

# SmolLM2-style W&B: do not hard-disable; soft-skip without API key.
from token_selection.olmo_ext.wandb_logging import (  # noqa: E402
    add_wandb_argparse_options,
    apply_wandb_env_defaults,
    ensure_wandb_not_hard_disabled,
    finish_wandb,
    init_wandb_from_args,
    is_production_run,
    namespace_path_config,
    require_wandb_for_production,
    wandb_log_directory_artifact,
    wandb_log_checkpoint,
    wandb_log_train,
    wandb_mode_from_args,
    wandb_upload_existing,
    WandbEvalPoller,
)

ensure_wandb_not_hard_disabled()

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from olmo_core.config import DType
from olmo_core.data.utils import get_labels
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from token_selection.olmo_ext.checkpoint_ladder import (
    DEFAULT_CHECKPOINT_INTERVAL,
    permanent_checkpoint_steps,
)

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
except Exception:  # pragma: no cover
    StateDictOptions = None  # type: ignore
    get_model_state_dict = None  # type: ignore
    get_optimizer_state_dict = None  # type: ignore
    set_model_state_dict = None  # type: ignore
    set_optimizer_state_dict = None  # type: ignore

log = logging.getLogger("train_blade_olmo2_370m")

SEQ_LEN = 2048
EMBEDDING_SIZE = 100_352
GLOBAL_BATCH_TOKENS = 4_194_304
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4
LABEL_IGNORE_INDEX = -100
CHECKPOINT_FORMAT = "blade_proxy_ref_v1"

# Locked schedule — change requires a new run_id.
BLADE_START = 500
BLADE_SYNC_STEPS: Tuple[int, ...] = (500, 875, 1250, 1625, 2000)
DEFAULT_TAU = 375
DEFAULT_K = 75
DEFAULT_GAMMA = 0.6
DEFAULT_LAMBDA_PEN = 1.0
DEFAULT_RUN_ID = "blade-regmix10b-v2"


def next_blade_sync_step(after_completed: int) -> int:
    """Next sync whose completed-step index is strictly greater than ``after_completed``."""
    for s in BLADE_SYNC_STEPS:
        if s > after_completed:
            return s
    return 10**9


# ---------------------------------------------------------------------------
# Bookkeeping stub so TrainModule.optim_step / metrics work without Trainer.fit
# ---------------------------------------------------------------------------


@dataclass
class _Bookkeeping:
    """Minimal Trainer duck-type for TrainModule.optim_step / record_metric."""

    global_step: int
    max_steps: int
    global_batch_size: int
    max_tokens: Optional[int] = None
    global_train_tokens_seen: int = 0
    dp_process_group: Any = None
    device: torch.device = torch.device("cuda")
    latest_metrics: Dict[str, float] = field(default_factory=dict)

    def record_metric(self, *args: Any, **kwargs: Any) -> None:
        name = kwargs.get("name")
        value = kwargs.get("value")
        if name is None and args:
            name = args[0]
        if value is None and len(args) >= 2:
            value = args[1]
        namespace = kwargs.get("namespace", "train")
        if name is None or value is None:
            return
        try:
            key = f"{namespace}/{name}" if namespace else str(name)
            self.latest_metrics[key] = float(value.item() if hasattr(value, "item") else value)
        except Exception:
            return

    def record_ce_loss(self, *args: Any, **kwargs: Any) -> None:
        value = kwargs.get("value")
        if value is None and args:
            value = args[0]
        if value is None:
            return
        try:
            self.latest_metrics["train/ce_loss"] = float(
                value.item() if hasattr(value, "item") else value
            )
        except Exception:
            return


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class MemmapTokenDataset(Dataset):
    """Contiguous SEQ_LEN chunks over one or more packed token memmaps (``.u32le.bin``)."""

    def __init__(
        self,
        paths: List[str],
        chunk_size: int = SEQ_LEN,
        *,
        dtype: Any = np.uint32,
    ) -> None:
        self.chunk_size = int(chunk_size)
        np_dtype = np.dtype(dtype)
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=np_dtype)
            n = (len(mm) - 1) // self.chunk_size
            if n <= 0:
                continue
            self._mmaps.append(mm)
            total += n
            self._cum_chunks.append(total)
        if total <= 0:
            raise SystemExit(f"No usable chunks in {len(paths)} paths")
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._total
        prev = 0
        for mm, cum in zip(self._mmaps, self._cum_chunks):
            if idx < cum:
                local = idx - prev
                start = local * self.chunk_size
                arr = np.asarray(mm[start : start + self.chunk_size + 1], dtype=np.int64)
                return torch.from_numpy(arr[:-1].copy())
            prev = cum
        raise IndexError(idx)


def collate_input_ids(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"input_ids": torch.stack(batch, dim=0)}


LEGACY_DATA_BUCKET = "edullm-datasets"


def read_paths(path: Path) -> List[str]:
    paths = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No paths in {path}")
    for p in paths:
        norm = p.replace("\\", "/")
        if LEGACY_DATA_BUCKET in norm:
            raise SystemExit(
                f"Refusing legacy edullm-datasets path in {path}: {p}\n"
                "Stage from s3://edullm-data via prepare_blade_data.py instead."
            )
        if norm.startswith("s3://"):
            raise SystemExit(
                f"Trainer expects local memmap paths (got {p!r}). "
                "Run prepare_blade_data.py to stage shards into job scratch."
            )
    return paths


class InfiniteBatchStream:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.pin_memory = torch.cuda.is_available()
        self._epoch = 0
        self._loader: Optional[DataLoader] = None
        self._it: Optional[Iterator] = None

    def _make_loader(self) -> DataLoader:
        if self.world_size > 1:
            sampler: Any = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.seed,
                drop_last=True,
            )
            sampler.set_epoch(self._epoch)
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch * 1_000_003)
            sampler = RandomSampler(self.dataset, replacement=False, generator=g)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_input_ids,
        )

    def next_batch(self) -> Dict[str, torch.Tensor]:
        if self._it is None:
            self._loader = self._make_loader()
            self._it = iter(self._loader)
        while True:
            try:
                return next(self._it)
            except StopIteration:
                self._epoch += 1
                self._loader = self._make_loader()
                self._it = iter(self._loader)


def next_rank_input_ids(stream: InfiniteBatchStream, n_seqs: int, device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    got = 0
    while got < n_seqs:
        x = stream.next_batch()["input_ids"]
        chunks.append(x)
        got += x.size(0)
    out = torch.cat(chunks, dim=0)[:n_seqs]
    return out.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Model / train module
# ---------------------------------------------------------------------------


def resolve_attn_backend() -> AttentionBackendName:
    prefer = os.environ.get("OLMO_ATTN_BACKEND", "auto").strip().lower()
    if prefer in ("torch", "sdpa", "eager"):
        return AttentionBackendName.torch
    want_flash = prefer in ("auto", "flash_2", "flash", "flash2")
    if want_flash:
        try:
            import flash_attn  # noqa: F401

            backend = AttentionBackendName.flash_2
            backend.get_class().assert_supported()
            log.info("attn_backend=flash_2 (flash_attn available)")
            return backend
        except Exception as e:
            if prefer != "auto":
                log.warning(
                    "OLMO_ATTN_BACKEND=%s but flash_attn unavailable (%s); using torch",
                    prefer,
                    e,
                )
            else:
                log.info("flash_attn unavailable (%s); attn_backend=torch (SDPA)", e)
    return AttentionBackendName.torch


def build_olmo2_config(*, fused_ce: bool) -> TransformerConfig:
    cfg = TransformerConfig.olmo2_370M(
        vocab_size=EMBEDDING_SIZE,
        attn_backend=resolve_attn_backend(),
    )
    if fused_ce:
        try:
            cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
            log.info("lm_head.loss_implementation=fused_linear (liger)")
        except Exception as e:
            log.warning("Could not set fused_linear (%s); using default CE", e)
    return cfg


def patch_liger_fused_ce_compat() -> bool:
    try:
        import importlib

        import liger_kernel  # noqa: F401

        cel = importlib.import_module("olmo_core.nn.functional.cross_entropy_loss")
    except Exception as e:
        log.warning("fused CE compat patch skipped (import): %s", e)
        return False

    apply_fn = getattr(cel, "_fused_linear_cross_entropy_loss", None)
    if apply_fn is None:
        log.warning("fused CE apply fn missing; cannot patch")
        return False

    if getattr(cel, "_edullm_fused_ce_patched", False):
        return True

    @torch._dynamo.disable()  # type: ignore[misc]
    def _fused_linear_cross_entropy_loss_compat(
        _input,
        weight,
        labels,
        *,
        bias=None,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
        ce_weight=None,
        label_smoothing: float = 0.0,
        softcap=None,
        accum_dtype=None,
    ):
        lse_scale = z_loss_multiplier if compute_z_loss else 0.0
        out = apply_fn(
            _input,
            weight,
            labels,
            bias,
            ce_weight,
            ignore_index,
            lse_scale,
            label_smoothing,
            reduction,
            softcap,
            compute_z_loss,
            accum_dtype,
        )
        if not isinstance(out, tuple):
            raise RuntimeError(f"unexpected fused CE return type: {type(out)}")
        ce_loss = out[0]
        z_loss = out[1] if len(out) > 1 else None
        if compute_z_loss:
            return ce_loss, z_loss
        return ce_loss, None

    cel.fused_linear_cross_entropy_loss = _fused_linear_cross_entropy_loss_compat  # type: ignore[attr-defined]
    cel._edullm_fused_ce_patched = True  # type: ignore[attr-defined]
    try:
        import olmo_core.nn.lm_head as lm_head

        lm_head.fused_linear_cross_entropy_loss = _fused_linear_cross_entropy_loss_compat  # type: ignore[attr-defined]
    except Exception as e:
        log.warning("could not patch lm_head fused CE binding (%s)", e)
    log.info("patched olmo_core fused_linear_cross_entropy_loss for liger>=0.8")
    return True


def try_enable_fused_ce() -> bool:
    try:
        import liger_kernel  # noqa: F401
    except Exception:
        log.warning("liger-kernel not installed; proxy/ref CE uses default LM-head path")
        return False
    if not patch_liger_fused_ce_compat():
        log.warning("liger present but fused CE compat patch failed; leaving default CE")
        return False
    log.info("liger-kernel available — enabling fused_linear CE")
    return True


def build_proxy_train_module(
    *,
    lr: float,
    lr_warmup_steps: int,
    alpha_f: float,
    compile_model: bool,
    rank_microbatch_tokens: int,
) -> TransformerTrainModule:
    # Keep fused CE off until focused production-LM-head value/gradient parity
    # is established; availability of liger alone is not evidence of parity.
    fused = False
    model_cfg = build_olmo2_config(fused_ce=fused)
    try:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps, alpha_f=alpha_f)
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps)
        if hasattr(scheduler, "alpha_f"):
            scheduler.alpha_f = alpha_f

    tm_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_tokens,
        max_sequence_length=SEQ_LEN,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=compile_model,
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
    train_module = tm_cfg.build(model)
    log.info(
        "Built proxy TransformerTrainModule (HSDP bf16, SkipStepAdamW, compile=%s, fused_ce=%s)",
        compile_model,
        fused,
    )
    return train_module


def cuda_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_reference_model(*, fused_ce: bool) -> nn.Module:
    """Dense reference copy (bf16 autocast + tiny microbatches in K-updates)."""
    cuda_gc()
    cfg = build_olmo2_config(fused_ce=fused_ce)
    model = cfg.build(init_device="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cuda_gc()
    return model


def mean_ce_loss(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    labels = get_labels({"input_ids": input_ids}, label_ignore_index=LABEL_IGNORE_INDEX)
    n = (labels != LABEL_IGNORE_INDEX).sum().clamp(min=1)
    with _autocast_ctx(input_ids.device):
        out = model(
            input_ids,
            labels=labels,
            ignore_index=LABEL_IGNORE_INDEX,
            loss_reduction="sum",
            loss_div_factor=n,
            return_logits=False,
        )
    if hasattr(out, "loss"):
        return out.loss
    if isinstance(out, tuple):
        return out[1]
    raise RuntimeError(f"Unexpected model output type: {type(out)}")


def per_token_ce(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    labels = get_labels({"input_ids": input_ids}, label_ignore_index=LABEL_IGNORE_INDEX)
    with _autocast_ctx(input_ids.device):
        out = model(
            input_ids,
            labels=labels,
            ignore_index=LABEL_IGNORE_INDEX,
            loss_reduction="none",
            return_logits=False,
        )
    if hasattr(out, "ce_loss"):
        return out.ce_loss
    raise RuntimeError(f"per-token CE unavailable from output type {type(out)}")


def top_gamma_label_mask(
    proxy_ce: torch.Tensor,
    ref_ce: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Keep top-γ tokens by BLADE score ``Δ = L_ref − L_proxy`` (paper Eq. 5)."""
    valid = labels != LABEL_IGNORE_INDEX
    score = ref_ce - proxy_ce
    flat = score[valid]
    if flat.numel() == 0:
        return valid
    k = max(1, int(math.ceil(gamma * flat.numel())))
    if k >= flat.numel():
        return valid
    thresh = torch.topk(flat, k, largest=True).values[-1]
    return valid & (score >= thresh)


# ---------------------------------------------------------------------------
# Reference sync / K updates
# ---------------------------------------------------------------------------


def sync_reference_from_proxy(reference: nn.Module, train_module: TransformerTrainModule) -> None:
    """Copy full proxy weights into the dense reference model."""
    cuda_gc()
    if get_model_state_dict is not None and StateDictOptions is not None:
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        sd = get_model_state_dict(train_module.model, options=opts)
        try:
            reference.load_state_dict(sd, strict=True)
        except Exception:
            stripped = {
                (k.split(".", 1)[-1] if k.startswith("module.") else k): v for k, v in sd.items()
            }
            reference.load_state_dict(stripped, strict=False)
        del sd
    else:
        sd = train_module.state_dict(optim=False)["model"]
        reference.load_state_dict(sd, strict=False)
        del sd
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    cuda_gc()


def update_reference_k_steps(
    reference: nn.Module,
    ref_opt: torch.optim.Optimizer,
    train_stream: InfiniteBatchStream,
    ref_stream: InfiniteBatchStream,
    device: torch.device,
    *,
    k_steps: int,
    lambda_pen: float,
    max_grad_norm: float,
    seqs_per_rank: int,
    micro_seqs: int,
    log_every: int = 25,
) -> float:
    """Paper: ``K`` steps of ``L_val + λ L_train`` on the reference."""
    cuda_gc()
    reference.train()
    for p in reference.parameters():
        p.requires_grad_(True)

    last_loss = 0.0
    micro_seqs = max(1, int(micro_seqs))
    n_micro = max(1, math.ceil(seqs_per_rank / micro_seqs))
    if get_rank() == 0:
        log.info(
            "  ref K-updates: K=%d micro_seqs=%d n_micro=%d (token-matched to GBS/rank)",
            k_steps,
            micro_seqs,
            n_micro,
        )
    for k in range(k_steps):
        ref_opt.zero_grad(set_to_none=True)
        micro_losses: List[float] = []
        micro_lval: List[float] = []
        micro_ltrain: List[float] = []
        tokens_done = 0
        for _ in range(n_micro):
            remain = seqs_per_rank - tokens_done
            this_m = min(micro_seqs, remain)
            if this_m <= 0:
                break
            train_ids = next_rank_input_ids(train_stream, this_m, device)
            val_ids = next_rank_input_ids(ref_stream, this_m, device)
            l_train = mean_ce_loss(reference, train_ids)
            l_val = mean_ce_loss(reference, val_ids)
            weight = float(this_m) / float(seqs_per_rank)
            loss = (l_val + lambda_pen * l_train) * weight
            loss.backward()
            micro_losses.append(float((l_val + lambda_pen * l_train).detach().item()))
            micro_lval.append(float(l_val.detach().item()))
            micro_ltrain.append(float(l_train.detach().item()))
            tokens_done += this_m
            del train_ids, val_ids, l_train, l_val, loss
        if is_distributed() and get_world_size() > 1:
            for p in reference.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), max_grad_norm)
        ref_opt.step()
        last_loss = float(sum(micro_losses) / max(1, len(micro_losses)))
        if get_rank() == 0 and ((k + 1) % log_every == 0 or k == 0 or (k + 1) == k_steps):
            log.info(
                "  ref update k=%d/%d loss=%.4f L_val=%.4f L_train=%.4f",
                k + 1,
                k_steps,
                last_loss,
                float(sum(micro_lval) / max(1, len(micro_lval))),
                float(sum(micro_ltrain) / max(1, len(micro_ltrain))),
            )

    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    cuda_gc()
    return last_loss


def build_blade_labels(
    train_module: TransformerTrainModule,
    reference: nn.Module,
    input_ids: torch.Tensor,
    gamma: float,
    micro_seqs: int,
) -> Tuple[torch.Tensor, float]:
    """Top-γ mask by ``L_ref − L_proxy``; return masked labels + select fraction."""
    B = input_ids.size(0)
    labels_full = get_labels({"input_ids": input_ids}, label_ignore_index=LABEL_IGNORE_INDEX)
    select = torch.zeros_like(labels_full, dtype=torch.bool)
    for start in range(0, B, micro_seqs):
        sl = slice(start, min(B, start + micro_seqs))
        ids = input_ids[sl]
        lab = labels_full[sl]
        with torch.no_grad():
            out_p = train_module.model_forward(
                ids,
                labels=lab,
                ignore_index=LABEL_IGNORE_INDEX,
                loss_reduction="none",
                return_logits=False,
            )
            if hasattr(out_p, "ce_loss"):
                proxy_ce = out_p.ce_loss
            else:
                proxy_ce = per_token_ce(train_module.model, ids)
            ref_ce = per_token_ce(reference, ids)
            select[sl] = top_gamma_label_mask(proxy_ce, ref_ce, lab, gamma)
    masked = labels_full.masked_fill(~select, LABEL_IGNORE_INDEX)
    valid = labels_full != LABEL_IGNORE_INDEX
    kept = (masked != LABEL_IGNORE_INDEX) & valid
    frac = float(kept.sum().item() / max(1, valid.sum().item()))
    return masked, frac


# ---------------------------------------------------------------------------
# Checkpointing (proxy + dynamic reference)
# ---------------------------------------------------------------------------


def _cpu_plain_tensor(t: Any) -> torch.Tensor:
    if torch.is_tensor(t) and type(t).__name__ == "Tensor":
        return t.detach().cpu()
    full = getattr(t, "full_tensor", None)
    if callable(full):
        try:
            return full().detach().cpu()
        except Exception:
            pass
    local = getattr(t, "to_local", None)
    if callable(local):
        try:
            return local().detach().cpu()
        except Exception:
            pass
    if torch.is_tensor(t):
        return t.detach().cpu()
    raise TypeError(f"cannot convert {type(t)} to CPU tensor")


def _plainify_state_tree(obj: Any) -> Any:
    if torch.is_tensor(obj) or type(obj).__name__ == "DTensor":
        return _cpu_plain_tensor(obj)
    if isinstance(obj, dict):
        return {k: _plainify_state_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_plainify_state_tree(v) for v in obj]
        return type(obj)(seq) if not isinstance(obj, list) else seq
    return obj


def gather_train_module_state_dict(train_module: TransformerTrainModule) -> dict[str, Any]:
    """All ranks must call. Full unsharded CPU proxy state on every rank."""
    if get_model_state_dict is None or StateDictOptions is None:
        return _plainify_state_tree(train_module.state_dict_to_save())

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(train_module.model, options=opts)
    optim_sd: Any = None
    if get_optimizer_state_dict is not None:
        try:
            optim_sd = get_optimizer_state_dict(
                train_module.model, train_module.optim, options=opts
            )
        except Exception as e:
            log.warning("full optimizer state gather failed (%s); saving model only", e)
    return {
        "model": _plainify_state_tree(model_sd),
        "optim": _plainify_state_tree(optim_sd) if optim_sd is not None else None,
    }


def gather_reference_state_dict(reference: Optional[nn.Module]) -> Optional[dict[str, Any]]:
    """Dense reference → CPU state dict (or None before first sync)."""
    if reference is None:
        return None
    return {"model": _plainify_state_tree(reference.state_dict())}


ARM = "blade"


def _broadcast_export_ok(export_ok: bool) -> bool:
    """Share rank-0 export success with every rank; return the consensus value."""
    if not is_distributed() or get_world_size() <= 1:
        return export_ok
    flag = torch.tensor(
        [1 if export_ok else 0],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def save_checkpoint(
    path: Path,
    step: int,
    train_module: TransformerTrainModule,
    reference: Optional[nn.Module],
    args: argparse.Namespace,
    meta: dict,
    *,
    task_loss_dir: Optional[Path] = None,
    task_loss_enabled: bool = True,
    progress_dir: Optional[Path] = None,
) -> None:
    """All ranks gather; rank 0 materializes, evaluates, and uploads to W&B."""
    train_module_sd = gather_train_module_state_dict(train_module)
    ref_sd = gather_reference_state_dict(reference)
    if get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "train_module": train_module_sd,
            "reference": ref_sd,
            "args": vars(args),
            "meta": meta,
            "architecture": "olmo_core.TransformerConfig.olmo2_370M",
            "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (RefHQ-matched)",
            "method": "BLADE",
            "blade_sync_steps": list(BLADE_SYNC_STEPS),
            "blade_start": BLADE_START,
            "selection_score": "L_ref - L_proxy",
            "checkpoint_format": CHECKPOINT_FORMAT,
            "has_reference": ref_sd is not None,
        }
        tmp = path / "state.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(path / "state.pt")
        (path / "step.txt").write_text(str(step) + "\n")
        n_model = len(train_module_sd.get("model") or {})
        log.info(
            "Saved BLADE checkpoint → %s (step=%s, proxy_tensors=%d, has_ref=%s, has_optim=%s)",
            path,
            step,
            n_model,
            ref_sd is not None,
            train_module_sd.get("optim") is not None,
        )
        from token_selection.olmo_ext.permanent_checkpoint import (
            copy_fingerprint_into_checkpoint,
        )

        copy_fingerprint_into_checkpoint(
            Path(args.save_folder) / "run_fingerprint.json", path
        )
    if get_rank() == 0:
        from token_selection.olmo_ext.permanent_checkpoint import (
            finalize_permanent_checkpoint,
        )

        finalize_permanent_checkpoint(
            arm=ARM,
            checkpoint_dir=path,
            step=step,
            run_name=str(args.name),
            task_loss_dir=task_loss_dir or Path(args.progress_dir) / "task_loss_results",
            task_loss_enabled=bool(task_loss_enabled),
            task_loss_eval_script=getattr(args, "task_loss_eval_script", None),
            progress_dir=progress_dir,
            fingerprint_path=Path(args.save_folder) / "run_fingerprint.json",
            wandb_run=getattr(args, "_wandb_run", None),
            wandb_mode=wandb_mode_from_args(args),
            production=bool(getattr(args, "_production", False)),
        )


def final_durable_export(
    *,
    save_folder: Path,
    progress_dir: Path,
    task_loss_dir: Optional[Path],
    run: Any | None,
    run_name: str,
    production: bool,
    mode: str,
) -> None:
    """Upload final progress/eval trees to W&B; checkpoints upload per save."""
    from token_selection.olmo_ext.wandb_logging import production_online

    export_ok = True
    if get_rank() == 0:
        strict = production_online(production=production, mode=mode)
        try:
            wandb_log_directory_artifact(
                run,
                progress_dir,
                name=f"{run_name}-progress",
                artifact_type="metrics",
                strict=strict,
            )
            if task_loss_dir is not None:
                wandb_log_directory_artifact(
                    run,
                    task_loss_dir,
                    name=f"{run_name}-task-loss",
                    artifact_type="eval",
                    strict=strict,
                )
        except Exception:
            export_ok = False
            if strict:
                raise
    if not _broadcast_export_ok(export_ok):
        raise SystemExit(f"W&B artifact upload failed for {save_folder}")


def maybe_restore_checkpoint_from_wandb(
    save_folder: Path,
    *,
    load_path: Optional[str],
) -> None:
    """Restore the requested W&B model artifact into runtime scratch."""
    del load_path
    if find_latest_checkpoint(save_folder) is not None:
        log.info("Local checkpoints already under %s (skip W&B restore)", save_folder)
        return
    ref = os.environ.get("WANDB_RESUME_ARTIFACT", "").strip()
    if not ref:
        raise RuntimeError("WANDB_RESUME_ARTIFACT is required for cross-job resume")
    from token_selection.olmo_ext.wandb_logging import restore_checkpoint_artifact

    restore_checkpoint_artifact(ref, save_folder)


def load_checkpoint(
    path: Path,
    train_module: TransformerTrainModule,
    *,
    fused_ref: bool,
) -> Tuple[int, Optional[nn.Module], Optional[torch.optim.Optimizer]]:
    """Load proxy (+optim) and dynamic reference. Does **not** re-sync ref←proxy."""
    ckpt = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    tm_sd = ckpt["train_module"]
    fmt = ckpt.get("checkpoint_format")
    if (
        fmt in (CHECKPOINT_FORMAT, "full_state_dict_v1")
        and isinstance(tm_sd, dict)
        and "model" in tm_sd
        and set_model_state_dict is not None
        and StateDictOptions is not None
    ):
        opts = StateDictOptions(full_state_dict=True, strict=True)
        set_model_state_dict(train_module.model, tm_sd["model"], options=opts)
        if tm_sd.get("optim") is not None and set_optimizer_state_dict is not None:
            try:
                set_optimizer_state_dict(
                    train_module.model,
                    train_module.optim,
                    tm_sd["optim"],
                    options=opts,
                )
            except Exception as e:
                log.warning("optimizer restore failed (%s); continuing with model weights", e)
        else:
            try:
                train_module.load_state_dict({"model": tm_sd["model"], "optim": tm_sd.get("optim")})
            except Exception:
                pass
    else:
        train_module.load_state_dict(tm_sd)
    _move_optim_state_to_param_device(train_module.optim)

    reference: Optional[nn.Module] = None
    ref_opt: Optional[torch.optim.Optimizer] = None
    ref_payload = ckpt.get("reference")
    if isinstance(ref_payload, dict) and ref_payload.get("model") is not None:
        reference = build_reference_model(fused_ce=fused_ref)
        try:
            reference.load_state_dict(ref_payload["model"], strict=True)
        except Exception:
            stripped = {
                (k.split(".", 1)[-1] if k.startswith("module.") else k): v
                for k, v in ref_payload["model"].items()
            }
            reference.load_state_dict(stripped, strict=False)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        # Optimizer for future K updates (fresh AdamW; K state is not resumed — syncs are rare).
        ref_opt = torch.optim.AdamW(
            reference.parameters(),
            lr=float(PEAK_LR),
            betas=(0.9, 0.95),
            weight_decay=0.1,
            foreach=False,
        )
        if get_rank() == 0:
            log.info("Restored dynamic reference from checkpoint (no re-sync)")
    elif get_rank() == 0:
        log.info("Checkpoint has no reference (warmup / pre-sync); will allocate at next sync")

    step = int(ckpt["step"])
    if get_rank() == 0:
        log.info(
            "Resumed from %s at step=%s format=%s has_ref=%s",
            path,
            step,
            fmt or "legacy",
            reference is not None,
        )
    return step, reference, ref_opt


def _move_optim_state_to_param_device(optim: torch.optim.Optimizer) -> None:
    from olmo_core.distributed.utils import get_local_tensor as _glt

    moved = 0
    for group in optim.param_groups:
        for p in group["params"]:
            state = optim.state.get(p)
            if not state:
                continue
            try:
                device = _glt(p).device
            except Exception:
                device = p.device
            for k, v in list(state.items()):
                if torch.is_tensor(v) and v.device != device:
                    state[k] = v.to(device=device)
                    moved += 1
    if moved and get_rank() == 0:
        log.info("Moved %d optimizer state tensor(s) onto param devices after resume", moved)


def find_latest_checkpoint(save_folder: Path) -> Optional[Path]:
    if not save_folder.is_dir():
        return None
    cands = sorted(
        [
            p
            for p in save_folder.iterdir()
            if p.is_dir() and p.name.startswith("step") and (p / "state.pt").is_file()
        ],
        key=lambda p: int(p.name.replace("step", "").split("-")[0]),
    )
    return cands[-1] if cands else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="Run id (e.g. blade-regmix10b-v2)")
    ap.add_argument(
        "--train-paths-file",
        required=True,
        help="Local memmap path list from prepare_blade_data.py (edullm-data staged shards)",
    )
    ap.add_argument(
        "--ref-paths-file",
        required=True,
        help="Local RefHQ memmap path list from prepare_blade_data.py",
    )
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument(
        "--length-tokens",
        type=int,
        required=True,
        help="Proxy train token budget (default via prepare: 9900000000 -> 2360 steps)",
    )
    ap.add_argument(
        "--token-dtype",
        type=str,
        default="uint32",
        help="Numpy dtype for staged .u32le.bin shards (from dataset_paths dtype)",
    )
    ap.add_argument(
        "--device-batch-size",
        type=int,
        default=MICROBATCH_TOKENS // SEQ_LEN,
        help="Sequences per proxy microbatch (default 32 = 65536 tokens)",
    )
    ap.add_argument(
        "--ref-device-batch-size",
        type=int,
        default=16,
        help="Sequences per reference microbatch during K-updates/scoring",
    )
    ap.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Permanent ladder interval (default 125)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=6198, help="Init / data seed (RefHQ-aligned default)")
    ap.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Local checkpoint dir to resume",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=0.1)
    ap.add_argument("--tau", type=int, default=DEFAULT_TAU)
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--lambda-pen", type=float, default=DEFAULT_LAMBDA_PEN)
    ap.add_argument("--blade-start", type=int, default=BLADE_START)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument(
        "--task-loss-dir",
        type=str,
        default=None,
        help="Directory for step{N}_task_loss.json "
        "(default: <progress-dir>/task_loss_results — job-scoped scratch)",
    )
    ap.add_argument(
        "--task-loss-eval-script",
        type=str,
        default=None,
        help="Path to eval_task_loss_olmo_core.py (optional; or set TASK_LOSS_EVAL_SCRIPT). "
        "Missing script skips eval without failing training.",
    )
    ap.add_argument(
        "--no-task-loss-eval",
        action="store_true",
        help="Do not spawn task_loss eval on permanent checkpoints "
        "(also gated by TASK_LOSS_EVAL=0 when eval is otherwise enabled)",
    )
    add_wandb_argparse_options(ap, default_run_name=DEFAULT_RUN_ID)
    return ap.parse_args()


def _validate_locked_schedule(args: argparse.Namespace) -> None:
    """Hard-fail on any deviation from the locked BLADE schedule (new run_id required)."""
    mismatches: List[str] = []
    if int(args.blade_start) != BLADE_START:
        mismatches.append(f"--blade-start={args.blade_start} (locked {BLADE_START})")
    if int(args.tau) != DEFAULT_TAU:
        mismatches.append(f"--tau={args.tau} (locked {DEFAULT_TAU})")
    if int(args.K) != DEFAULT_K:
        mismatches.append(f"--K={args.K} (locked {DEFAULT_K})")
    if abs(float(args.gamma) - DEFAULT_GAMMA) > 1e-9:
        mismatches.append(f"--gamma={args.gamma} (locked {DEFAULT_GAMMA})")
    if abs(float(args.lambda_pen) - DEFAULT_LAMBDA_PEN) > 1e-9:
        mismatches.append(f"--lambda-pen={args.lambda_pen} (locked {DEFAULT_LAMBDA_PEN})")
    if mismatches:
        raise SystemExit(
            "Locked BLADE schedule mismatch (do not change without a new run_id):\n  - "
            + "\n  - ".join(mismatches)
            + f"\n  sync steps remain {list(BLADE_SYNC_STEPS)}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    prepare_training_environment()
    try:
        _run(args)
    finally:
        teardown_training_environment()


def _run(args: argparse.Namespace) -> None:
    _validate_locked_schedule(args)
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device("cuda")
    seed_all(args.seed + rank)

    mbs = int(args.device_batch_size)
    ref_mbs = max(1, int(args.ref_device_batch_size))
    rank_micro_tokens = mbs * SEQ_LEN
    if GLOBAL_BATCH_TOKENS % (world_size * rank_micro_tokens) != 0:
        raise SystemExit(
            f"global_batch_tokens {GLOBAL_BATCH_TOKENS} not divisible by "
            f"world_size*rank_micro ({world_size}*{rank_micro_tokens}). "
            "Adjust --device-batch-size or world size so they divide evenly."
        )
    seqs_per_rank = GLOBAL_BATCH_TOKENS // (SEQ_LEN * world_size)
    tokens_per_step = GLOBAL_BATCH_TOKENS
    total_steps = int(args.length_tokens) // tokens_per_step
    production = is_production_run(
        max_tokens=int(args.length_tokens), total_steps=total_steps
    )
    args._production = production
    blade_start = int(args.blade_start)
    if blade_start >= total_steps:
        raise SystemExit(f"blade_start={blade_start} must be < total_steps={total_steps}")
    for s in BLADE_SYNC_STEPS:
        if s > total_steps:
            raise SystemExit(f"sync step {s} exceeds total_steps={total_steps}")

    ckpt_interval = int(args.checkpoint_interval)
    ladder = permanent_checkpoint_steps(total_steps, ckpt_interval)
    ladder_set: Set[int] = set(ladder)
    for s in BLADE_SYNC_STEPS:
        if s not in ladder_set:
            raise SystemExit(
                f"sync step {s} is not on the permanent checkpoint ladder {ladder[:5]}…; "
                f"interval={ckpt_interval} total={total_steps}"
            )

    lr = float(PEAK_LR)
    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if args.task_loss_dir:
        task_loss_dir = Path(args.task_loss_dir)
    else:
        # Job-scoped under progress (not repo-tree); uploaded with progress via W&B.
        task_loss_dir = progress_dir / "task_loss_results"
    task_loss_enabled = not bool(args.no_task_loss_eval)

    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        if task_loss_enabled:
            task_loss_dir.mkdir(parents=True, exist_ok=True)

    apply_wandb_env_defaults(project="token-selection", run_name=args.name, group=ARM)
    ensure_wandb_not_hard_disabled()
    wb_run = None
    eval_poller: Optional[WandbEvalPoller] = None
    if rank == 0:
        wb_run = init_wandb_from_args(
            args,
            run_name=args.name,
            config=namespace_path_config(args),
            group=ARM,
            tags=(ARM, "blade"),
            dir=progress_dir / "wandb",
            id_path=progress_dir / "wandb_run_id.txt",
            is_main=True,
            alert_title=f"token-selection {ARM} started",
        )
        args._wandb_run = wb_run
        require_wandb_for_production(
            wb_run, production=production, mode=wandb_mode_from_args(args)
        )
        eval_poller = WandbEvalPoller(str(task_loss_dir), wb_run)
        if wb_run is not None and bool(getattr(args, "wandb_upload_existing", False)):
            wandb_upload_existing(
                wb_run,
                checkpoint_dir=save_folder,
                task_loss_dir=str(task_loss_dir),
                progress_dir=progress_dir,
                tokens_per_step=GLOBAL_BATCH_TOKENS,
            )
    else:
        args._wandb_run = None

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "run_id": args.name,
        "method": "BLADE",
        "proxy_train_stack": "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
        "matched_reference": "experiments/token-selection/reference/train_olmo3_370m_refhq.py",
        "selection_score": "L_ref - L_proxy",
        "gamma": float(args.gamma),
        "lambda_pen": float(args.lambda_pen),
        "tau": int(args.tau),
        "K": int(args.K),
        "blade_start": blade_start,
        "blade_sync_steps": list(BLADE_SYNC_STEPS),
        "length_tokens": int(args.length_tokens),
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": rank_micro_tokens,
        "sequence_length": SEQ_LEN,
        "vocab_size": EMBEDDING_SIZE,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "permanent_checkpoint_steps": ladder,
        "checkpoint_interval": ckpt_interval,
        "max_checkpoints": None,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "device_microbatch_seqs": mbs,
        "ref_device_batch_size": ref_mbs,
        "seqs_per_rank": seqs_per_rank,
        "world_size": world_size,
        "lr": lr,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_alpha_f": float(args.lr_alpha_f),
        "z_loss_multiplier": 1e-5,
        "max_grad_norm": float(args.max_grad_norm),
        "compile": bool(args.compile),
        "attn_backend": str(resolve_attn_backend()),
        "s3_prefix": "token-sel/blade",
        "artifact_store": "wandb",
        "train_dataset": "pretrain/regmix-10b",
        "ref_dataset": "pretrain/refhq-regmix-5p5b",
        "train_dataset_uri": "s3://edullm-data/pretrain/regmix-10b/",
        "ref_dataset_uri": "s3://edullm-data/pretrain/refhq-regmix-5p5b/",
        "token_dtype": str(args.token_dtype),
        "seed": args.seed,
        "task_loss_dir": str(task_loss_dir),
        "task_loss_on_save": task_loss_enabled,
        "ephemeral_scratch": True,
    }
    from token_selection.scripts.experiment_contract import sha256_file

    run_identity = {
        "arm": ARM,
        "run_id": args.name,
        "method": "BLADE",
        "seed": int(args.seed),
        "model": "olmo2_370M",
        "sequence_length": SEQ_LEN,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "max_tokens": int(args.length_tokens),
        "total_steps": total_steps,
        "keep_fraction": float(args.gamma),
        "blade_start": blade_start,
        "blade_sync_steps": list(BLADE_SYNC_STEPS),
        "tau": int(args.tau),
        "K": int(args.K),
        "lambda_pen": float(args.lambda_pen),
        "selection_score": "L_ref-L_proxy",
        "train_paths_sha256": sha256_file(Path(args.train_paths_file)),
        "reference_paths_sha256": sha256_file(Path(args.ref_paths_file)),
        "reference_stream_seed_offset": 101,
        "reference_train_stream_seed_offset": 17,
        "lr": lr,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_alpha_f": float(args.lr_alpha_f),
        "fused_ce": False,
        "task_loss_definition": "olmo-ladder-20-label-macro-bpb",
    }
    if rank == 0 and args.fresh:
        from token_selection.olmo_ext.permanent_checkpoint import write_run_fingerprint

        write_run_fingerprint(save_folder, run_identity)
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "run_plan.txt").write_text(
            "\n".join(
                [
                    "architecture=olmo_core.TransformerConfig.olmo2_370M",
                    "method=BLADE",
                    f"run_id={args.name}",
                    f"world_size={world_size}",
                    f"total_steps={total_steps}",
                    f"blade_start={blade_start}  # warmup steps 0..{blade_start - 1}",
                    f"blade_sync_steps={list(BLADE_SYNC_STEPS)}",
                    f"tau={args.tau}  K={args.K}  gamma={args.gamma}  lambda={args.lambda_pen}",
                    "selection_score=L_ref - L_proxy",
                    f"permanent_checkpoints={ladder}",
                    f"checkpoint_format={CHECKPOINT_FORMAT}  # proxy+optim + reference",
                    f"global_batch_tokens={GLOBAL_BATCH_TOKENS}  mbs_seqs={mbs}  "
                    f"ref_mbs_seqs={ref_mbs}  seqs_per_rank={seqs_per_rank}",
                    f"lr={lr}  cos_warmup={args.lr_warmup_steps}  alpha_f={args.lr_alpha_f}",
                    f"compile={args.compile}",
                    "train=pretrain/regmix-10b (s3://edullm-data/)",
                    "refhq=pretrain/refhq-regmix-5p5b (s3://edullm-data/)",
                    f"token_dtype={args.token_dtype}",
                    f"task_loss_dir={task_loss_dir}",
                    "artifact_store=wandb  # production online uploads fail closed",
                    "ephemeral_scratch=True  # S3 only stages inputs at run start",
                    "",
                ]
            )
        )
        log.info(
            "Plan: BLADE olmo2_370M world=%d total=%d blade_start=%d syncs=%s "
            "K=%d γ=%.2f ladder_n=%d lr=%.3e",
            world_size,
            total_steps,
            blade_start,
            list(BLADE_SYNC_STEPS),
            args.K,
            args.gamma,
            len(ladder),
            lr,
        )

    train_paths = read_paths(Path(args.train_paths_file))
    ref_paths = read_paths(Path(args.ref_paths_file))
    token_dtype = np.dtype(str(args.token_dtype))
    train_ds = MemmapTokenDataset(train_paths, SEQ_LEN, dtype=token_dtype)
    ref_ds = MemmapTokenDataset(ref_paths, SEQ_LEN, dtype=token_dtype)
    workers = args.num_workers if world_size == 1 else max(1, args.num_workers // world_size)
    train_stream = InfiniteBatchStream(
        train_ds, mbs, workers, args.seed, rank=rank, world_size=world_size
    )
    ref_train_stream = InfiniteBatchStream(
        train_ds, mbs, max(1, workers // 2), args.seed + 17, rank=rank, world_size=world_size
    )
    ref_hq_stream = InfiniteBatchStream(
        ref_ds, mbs, max(1, workers // 2), args.seed + 101, rank=rank, world_size=world_size
    )

    train_module = build_proxy_train_module(
        lr=lr,
        lr_warmup_steps=int(args.lr_warmup_steps),
        alpha_f=float(args.lr_alpha_f),
        compile_model=bool(args.compile),
        rank_microbatch_tokens=rank_micro_tokens,
    )
    books = _Bookkeeping(
        global_step=0,
        max_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        device=device,
    )
    train_module._attach_trainer(books)  # type: ignore[arg-type]

    fused_ref = False
    reference: Optional[nn.Module] = None
    ref_opt: Optional[torch.optim.Optimizer] = None

    start_step = 0
    if args.fresh:
        if rank == 0:
            log.info("--fresh: starting from scratch (ignoring local checkpoints)")
    else:
        if args.wandb_resume_artifact:
            os.environ["WANDB_RESUME_ARTIFACT"] = str(args.wandb_resume_artifact)
            fetch_ok = True
            if rank == 0:
                try:
                    maybe_restore_checkpoint_from_wandb(
                        save_folder, load_path=args.load_path
                    )
                except Exception as exc:  # noqa: BLE001
                    fetch_ok = False
                    log.error("--wandb-resume-artifact failed: %s", exc)
            if not _broadcast_export_ok(fetch_ok):
                raise SystemExit("W&B checkpoint restore failed; verify the artifact ref")
            if is_distributed():
                dist.barrier()
        load_dir: Optional[Path] = None
        if args.load_path:
            load_dir = Path(args.load_path)
            if not (load_dir / "state.pt").is_file():
                raise SystemExit(
                    f"--load-path {load_dir} has no state.pt. "
                    "Pass --wandb-resume-artifact or stage the step locally first."
                )
        else:
            load_dir = find_latest_checkpoint(save_folder)
            if load_dir is None and rank == 0:
                log.info(
                    "No local checkpoint under %s; starting at step 0. "
                    "For cross-job resume use --wandb-resume-artifact.",
                    save_folder,
                )
        if load_dir is not None:
            from token_selection.olmo_ext.permanent_checkpoint import (
                assert_resume_fingerprint,
                write_run_fingerprint,
            )

            assert_resume_fingerprint(load_dir, run_identity)
            if rank == 0:
                write_run_fingerprint(save_folder, run_identity)
            start_step, reference, ref_opt = load_checkpoint(
                load_dir, train_module, fused_ref=fused_ref
            )
        elif rank == 0:
            from token_selection.olmo_ext.permanent_checkpoint import (
                write_run_fingerprint,
            )

            write_run_fingerprint(save_folder, run_identity)

    # Methodology-safe resume: next sync is the next *future* scheduled sync only.
    # Never inject an unscheduled re-sync between sync points.
    next_sync = next_blade_sync_step(start_step)
    if rank == 0:
        log.info(
            "BLADE sync schedule=%s next_sync=%s (blade_start=%d start_step=%d has_ref=%s)",
            list(BLADE_SYNC_STEPS),
            next_sync if next_sync < 10**9 else None,
            blade_start,
            start_step,
            reference is not None,
        )

    # Step-0 init snapshot (proxy only; reference null).
    if start_step == 0 and 0 in ladder_set:
        if is_distributed():
            dist.barrier()
        save_checkpoint(
            save_folder / "step0",
            0,
            train_module,
            reference,
            args,
            meta,
            task_loss_dir=task_loss_dir,
            task_loss_enabled=task_loss_enabled,
            progress_dir=progress_dir,
        )
        if is_distributed():
            dist.barrier()

    t0 = time.time()
    window_t0 = t0
    window_step0 = start_step
    loss_path = progress_dir / "train_loss.jsonl"

    if is_distributed():
        dist.barrier()

    # ``step`` = completed optimizer steps; about_to = step+1 is the update we run.
    for step in range(start_step, total_steps):
        about_to = step + 1
        books.global_step = step
        books.global_train_tokens_seen = step * tokens_per_step
        in_blade = about_to >= blade_start

        if about_to in BLADE_SYNC_STEPS:
            if reference is None:
                if rank == 0:
                    log.info(
                        "Allocating reference model at sync step %d (ref_mbs=%d)",
                        about_to,
                        ref_mbs,
                    )
                train_module.zero_grads()
                cuda_gc()
                reference = build_reference_model(fused_ce=fused_ref)
                ref_opt = torch.optim.AdamW(
                    reference.parameters(),
                    lr=lr,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                    foreach=False,
                )
                cuda_gc()
            assert reference is not None and ref_opt is not None
            if rank == 0:
                log.info("=== BLADE sync at step %d: reference ← proxy, K=%d ===", about_to, args.K)
            train_module.zero_grads()
            cuda_gc()
            sync_reference_from_proxy(reference, train_module)
            for g in ref_opt.param_groups:
                g["lr"] = lr
            update_reference_k_steps(
                reference,
                ref_opt,
                ref_train_stream,
                ref_hq_stream,
                device,
                k_steps=int(args.K),
                lambda_pen=float(args.lambda_pen),
                max_grad_norm=float(args.max_grad_norm),
                seqs_per_rank=seqs_per_rank,
                micro_seqs=ref_mbs,
            )
            next_sync = next_blade_sync_step(about_to)
            if rank == 0:
                log.info(
                    "Next BLADE sync scheduled at step %s",
                    next_sync if next_sync < 10**9 else None,
                )
            if is_distributed():
                dist.barrier()

        input_ids = next_rank_input_ids(train_stream, seqs_per_rank, device)
        batch: Dict[str, torch.Tensor] = {"input_ids": input_ids}
        select_frac = 1.0
        if in_blade:
            if reference is None:
                raise RuntimeError(
                    f"BLADE selection at step {about_to} requires a reference; "
                    "checkpoint may be corrupt or resume skipped a sync without saving ref"
                )
            was_training = train_module.model.training
            train_module.model.eval()
            labels, select_frac = build_blade_labels(
                train_module, reference, input_ids, float(args.gamma), ref_mbs
            )
            if was_training:
                train_module.model.train()
            batch["labels"] = labels

        train_module.zero_grads()
        train_module.train_batch(batch)
        train_module.optim_step()

        global_step = about_to
        if global_step % args.log_interval == 0 or global_step == 1:
            now = time.time()
            elapsed = now - t0
            done = max(1, global_step - start_step)
            tok_s_avg = done * tokens_per_step / max(elapsed, 1e-6)
            w_steps = max(1, global_step - window_step0)
            w_elapsed = max(now - window_t0, 1e-6)
            tok_s = w_steps * tokens_per_step / w_elapsed
            window_t0 = now
            window_step0 = global_step
            phase = "blade" if in_blade else "warmup"
            if rank == 0:
                log.info(
                    "step=%d/%d phase=%s select_frac=%.3f tok/s=%.0f (avg=%.0f) world=%d",
                    global_step,
                    total_steps,
                    phase,
                    select_frac,
                    tok_s,
                    tok_s_avg,
                    world_size,
                )
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "phase": phase,
                                "select_frac": select_frac,
                                "tok_per_s": tok_s,
                                "tok_per_s_avg": tok_s_avg,
                            }
                        )
                        + "\n"
                    )
                (progress_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "phase": phase,
                            "select_frac": select_frac,
                            "world_size": world_size,
                            "tok_per_s": tok_s,
                            "tok_per_s_avg": tok_s_avg,
                            "pct": round(100.0 * global_step / total_steps, 4),
                            "has_reference": reference is not None,
                        }
                    )
                    + "\n"
                )
                train_loss = books.latest_metrics.get("train/ce_loss") or books.latest_metrics.get(
                    "train/loss"
                )
                wandb_log_train(
                    wb_run,
                    step=global_step,
                    train_loss=train_loss,
                    tokens_seen=global_step * tokens_per_step,
                    tok_per_s=tok_s,
                    tok_per_s_avg=tok_s_avg,
                    extra={
                        **{k: v for k, v in books.latest_metrics.items() if k.startswith("train/")},
                        "train/select_frac": select_frac,
                    },
                )
                if eval_poller is not None:
                    eval_poller.poll()

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            # At sync steps this save is post-K (sync ran at start of this iteration).
            save_checkpoint(
                save_folder / f"step{global_step}",
                global_step,
                train_module,
                reference,
                args,
                meta,
                task_loss_dir=task_loss_dir,
                task_loss_enabled=task_loss_enabled,
                progress_dir=progress_dir,
            )
            if rank == 0 and eval_poller is not None:
                eval_poller.poll()
            if is_distributed():
                dist.barrier()

    if rank == 0:
        if eval_poller is not None:
            eval_poller.poll()
        log.info("Training complete at step=%d world_size=%d", total_steps, world_size)
    final_durable_export(
        save_folder=save_folder,
        progress_dir=progress_dir,
        task_loss_dir=task_loss_dir if task_loss_enabled else None,
        run=wb_run,
        run_name=str(args.name),
        production=production,
        mode=wandb_mode_from_args(args),
    )
    if rank == 0:
        finish_wandb(wb_run)


if __name__ == "__main__":
    main()
