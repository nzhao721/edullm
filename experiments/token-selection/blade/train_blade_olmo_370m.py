#!/usr/bin/env python3
"""BLADE pretraining with GPU7-matched olmo_core proxy training.

Proxy warmup + BLADE selected steps use the same train stack as the GPU7
RefHQ CE run (``train_olmo3_370m_refhq.py`` / REL+EMA shape):

  * ``TransformerConfig.olmo2_370M`` (full attn, no SWA)
  * ``TransformerTrainModule`` — HSDP bf16, SkipStepAdamW, CosWithWarmup,
    ``compile_model=True``, ``z_loss_multiplier=1e-5``
  * fused LM-head CE when liger-kernel is available (no full logits)
  * sequence=2048, global_batch=4_194_304, rank_microbatch=65_536

BLADE additions (unchanged intent):
  * proxy-only warmup for ``total − n_blocks·τ`` steps
  * BLADE syncs at fixed steps (default 750, 1150, 1550, 1950): reference ← proxy,
    then ``K`` reference updates; between syncs top-``γ`` excess-loss label mask (γ=0.6)

Datasets:
  * train / proxy: RegMix 10B
  * reference HQ / val: RefHQ 5.5B

Launch via ``torchrun`` + your own launcher script. Does **not** call AWS.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
except Exception:  # pragma: no cover
    StateDictOptions = None  # type: ignore
    get_model_state_dict = None  # type: ignore
    set_model_state_dict = None  # type: ignore

log = logging.getLogger("train_blade_olmo2_370m")

SEQ_LEN = 2048
EMBEDDING_SIZE = 100_352
GLOBAL_BATCH_TOKENS = 4_194_304
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4
LABEL_IGNORE_INDEX = -100

# Fixed BLADE reference-sync steps (not every τ). τ still sets warmup length via
# blade_steps = n_blade_blocks * τ. Mid-BLADE resume syncs immediately when the
# resume step is not itself a scheduled sync (reference weights are not checkpointed).
BLADE_SYNC_STEPS: Tuple[int, ...] = (750, 1150, 1550, 1950)


def next_blade_sync_step(after_step: int, *, inclusive: bool = False) -> int:
    """Return the next scheduled sync step, or a large sentinel if none remain."""
    for s in BLADE_SYNC_STEPS:
        if inclusive and s >= after_step:
            return s
        if not inclusive and s > after_step:
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

    def record_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_ce_loss(self, *args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class MemmapTokenDataset(Dataset):
    """Contiguous SEQ_LEN chunks over one or more uint32 token memmaps."""

    def __init__(self, paths: List[str], chunk_size: int = SEQ_LEN) -> None:
        self.chunk_size = int(chunk_size)
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=np.uint32)
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


def read_paths(path: Path) -> List[str]:
    paths = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No paths in {path}")
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
    """Concatenate microbatches until we have ``n_seqs`` sequences for this rank."""
    chunks: List[torch.Tensor] = []
    got = 0
    while got < n_seqs:
        x = stream.next_batch()["input_ids"]
        chunks.append(x)
        got += x.size(0)
    out = torch.cat(chunks, dim=0)[:n_seqs]
    return out.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Model / train module (GPU7 RefHQ stack)
# ---------------------------------------------------------------------------


def resolve_attn_backend() -> AttentionBackendName:
    """Prefer flash_attn when available; else PyTorch SDPA (often Flash under the hood).

    Env ``OLMO_ATTN_BACKEND``: ``auto`` (default), ``flash_2``/``flash``, or ``torch``.
    Kernel choice does not change the training objective — only the attn implementation.
    """
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
                log.warning("OLMO_ATTN_BACKEND=%s but flash_attn unavailable (%s); using torch", prefer, e)
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
    """Make olmo_core fused CE work with liger-kernel>=0.8 (4-tuple returns).

    liger 0.8+ returns ``(loss, z_loss, token_acc, predicted_tokens)`` while older
    olmo_core unpacks 3 values. Also zero ``lse_square_scale`` when
    ``compute_z_loss=False`` so fused CE matches plain CE (no silent z-term).
    """
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

    # Idempotent.
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
        # Accuracy: do not fold z-loss into CE when compute_z_loss is False.
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
    # lm_head typically binds the symbol at import time — patch both.
    try:
        import olmo_core.nn.lm_head as lm_head

        lm_head.fused_linear_cross_entropy_loss = _fused_linear_cross_entropy_loss_compat  # type: ignore[attr-defined]
    except Exception as e:
        log.warning("could not patch lm_head fused CE binding (%s)", e)
    log.info("patched olmo_core fused_linear_cross_entropy_loss for liger>=0.8")
    return True


def try_enable_fused_ce() -> bool:
    """Enable olmo_core fused LM-head CE via liger-kernel (same CE, no full logits)."""
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
    fused = try_enable_fused_ce()
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
    """Dense reference copy. Kept small: bf16 autocast + tiny microbatches in K-updates."""
    cuda_gc()
    cfg = build_olmo2_config(fused_ce=fused_ce)
    model = cfg.build(init_device="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cuda_gc()
    return model


def mean_ce_loss(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Mean token CE via LM-head (fused when configured)."""
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
    """Per-position CE aligned with ``get_labels`` layout, shape [B, S]."""
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
    """Bool mask over label positions: keep top-γ excess-loss tokens."""
    valid = labels != LABEL_IGNORE_INDEX
    excess = proxy_ce - ref_ce
    flat = excess[valid]
    if flat.numel() == 0:
        return valid
    k = max(1, int(math.ceil(gamma * flat.numel())))
    if k >= flat.numel():
        return valid
    thresh = torch.topk(flat, k, largest=True).values[-1]
    return valid & (excess >= thresh)


# ---------------------------------------------------------------------------
# Reference sync / K updates
# ---------------------------------------------------------------------------


def sync_reference_from_proxy(reference: nn.Module, train_module: TransformerTrainModule) -> None:
    """Copy full proxy weights into the dense reference model."""
    cuda_gc()
    if get_model_state_dict is not None and StateDictOptions is not None:
        # cpu_offload avoids a second full GPU copy while gathering HSDP shards.
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
    """Paper Eq. 4: ``K`` steps of ``L_val + λ L_train`` on the reference.

    Uses tiny microbatches so dense-ref activations/logits fit beside the HSDP proxy.
    """
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
            # Last microbatch may be partial so we still cover seqs_per_rank.
            remain = seqs_per_rank - tokens_done
            this_m = min(micro_seqs, remain)
            if this_m <= 0:
                break
            train_ids = next_rank_input_ids(train_stream, this_m, device)
            val_ids = next_rank_input_ids(ref_stream, this_m, device)
            l_train = mean_ce_loss(reference, train_ids)
            l_val = mean_ce_loss(reference, val_ids)
            # Weight by microbatch size so partial last step is unbiased.
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
    """Compute excess-loss top-γ mask and return masked labels + select fraction."""
    B = input_ids.size(0)
    labels_full = get_labels({"input_ids": input_ids}, label_ignore_index=LABEL_IGNORE_INDEX)
    select = torch.zeros_like(labels_full, dtype=torch.bool)
    # Score in microbatches to limit activation memory.
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
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    step: int,
    train_module: TransformerTrainModule,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "train_module": train_module.state_dict_to_save(),
        "args": vars(args),
        "meta": meta,
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (GPU7-matched)",
    }
    tmp = path / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(path / "state.pt")
    (path / "step.txt").write_text(str(step) + "\n")
    log.info("Saved checkpoint → %s (step=%s)", path, step)


def load_checkpoint(path: Path, train_module: TransformerTrainModule) -> int:
    ckpt = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    train_module.load_state_dict(ckpt["train_module"])
    # torch.load(..., map_location="cpu") can leave Adam moment/step tensors on CPU;
    # SkipStepAdamW foreach kernels then mix CPU step_sizes with CUDA step_factor.
    _move_optim_state_to_param_device(train_module.optim)
    step = int(ckpt["step"])
    log.info("Resumed from %s at step=%s", path, step)
    return step


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
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-paths-file", required=True)
    ap.add_argument("--ref-paths-file", required=True)
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--length-tokens", type=int, required=True)
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
        help="Sequences per reference microbatch during K-updates/scoring (default 16; "
        "max stable beside HSDP proxy on 2x B200; 24 OOMs)",
    )
    ap.add_argument("--save-interval", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--load-path", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=0.1)
    ap.add_argument("--tau", type=int, default=375)
    ap.add_argument("--K", type=int, default=75)
    ap.add_argument("--n-blade-blocks", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--lambda-pen", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    prepare_training_environment()
    try:
        _run(args)
    finally:
        teardown_training_environment()


def _run(args: argparse.Namespace) -> None:
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
            f"world_size*rank_micro ({world_size}*{rank_micro_tokens})"
        )
    seqs_per_rank = GLOBAL_BATCH_TOKENS // (SEQ_LEN * world_size)
    tokens_per_step = GLOBAL_BATCH_TOKENS
    total_steps = int(args.length_tokens) // tokens_per_step
    blade_steps = int(args.n_blade_blocks) * int(args.tau)
    if blade_steps >= total_steps:
        raise SystemExit(f"blade_steps={blade_steps} must be < total_steps={total_steps}")
    warmup_steps = total_steps - blade_steps
    lr = float(PEAK_LR)

    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "matched_teammate_run": "rel-ema-5b-scratch-v1 / GPU7 RefHQ CE stack",
        "method": "BLADE",
        "proxy_train_stack": "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
        "proxy_loss": "LM-head CE (+ label_mask top-γ during BLADE)",
        "reference_loss": "L_val + λ L_train (paper)",
        "ref_device_batch_size": ref_mbs,
        "length_tokens": int(args.length_tokens),
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "sequence_length": SEQ_LEN,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "blade_steps": blade_steps,
        "n_blade_blocks": args.n_blade_blocks,
        "tau": args.tau,
        "K": args.K,
        "gamma": args.gamma,
        "lambda_pen": args.lambda_pen,
        "device_microbatch_seqs": mbs,
        "seqs_per_rank": seqs_per_rank,
        "world_size": world_size,
        "lr": lr,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "compile": bool(args.compile),
        "attn_backend": str(resolve_attn_backend()),
        "train_dataset": "s3://edullm-dataset-regmix/regmix-10b/",
        "ref_dataset": "s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/",
        "seed": args.seed,
    }
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "run_plan.txt").write_text(
            "\n".join(
                [
                    "architecture=olmo_core.TransformerConfig.olmo2_370M",
                    "proxy_stack=TransformerTrainModule/HSDP/SkipStepAdamW (GPU7-matched)",
                    f"world_size={world_size}",
                    f"sequence_length={SEQ_LEN}",
                    f"total_steps={total_steps}",
                    f"warmup_steps={warmup_steps}",
                    f"blade_steps={blade_steps}  # {args.n_blade_blocks} × tau={args.tau}",
                    f"blade_sync_steps={list(BLADE_SYNC_STEPS)}",
                    f"K={args.K}  gamma={args.gamma}  lambda={args.lambda_pen}",
                    f"global_batch_tokens={GLOBAL_BATCH_TOKENS}  mbs_seqs={mbs}  "
                    f"ref_mbs_seqs={ref_mbs}  seqs_per_rank={seqs_per_rank}",
                    f"lr={lr}  cos_warmup={args.lr_warmup_steps}  alpha_f={args.lr_alpha_f}",
                    f"compile={args.compile}",
                    "train=s3://edullm-dataset-regmix/regmix-10b/",
                    "refhq=s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/",
                    "",
                ]
            )
        )
        log.info(
            "Plan: olmo2_370M GPU7-stack world=%d total=%d warmup=%d blade=%d "
            "K=%d γ=%.2f mbs=%d ref_mbs=%d seqs/rank=%d lr=%.3e",
            world_size,
            total_steps,
            warmup_steps,
            blade_steps,
            args.K,
            args.gamma,
            mbs,
            ref_mbs,
            seqs_per_rank,
            lr,
        )

    train_paths = read_paths(Path(args.train_paths_file))
    ref_paths = read_paths(Path(args.ref_paths_file))
    train_ds = MemmapTokenDataset(train_paths, SEQ_LEN)
    ref_ds = MemmapTokenDataset(ref_paths, SEQ_LEN)
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
    # Attach bookkeeping so optim_step / SkipStepAdamW LR schedule / metrics work.
    books = _Bookkeeping(
        global_step=0,
        max_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        device=device,
    )
    train_module._attach_trainer(books)  # type: ignore[arg-type]

    start_step = 0
    if args.fresh:
        if rank == 0:
            log.info("--fresh: starting from scratch")
    else:
        load_dir = Path(args.load_path) if args.load_path else find_latest_checkpoint(save_folder)
        if load_dir is not None:
            start_step = load_checkpoint(load_dir, train_module)

    reference: Optional[nn.Module] = None
    ref_opt: Optional[torch.optim.Optimizer] = None
    fused_ref = try_enable_fused_ce()

    blade_start = warmup_steps
    # Scheduled syncs: BLADE_SYNC_STEPS. Fresh runs still sync once at blade_start
    # (first entry to BLADE) if that is before the first scheduled step.
    # Mid-BLADE resume: sync at start_step when it is scheduled, otherwise sync
    # immediately so token selection has a live reference (not checkpointed).
    if start_step < blade_start:
        next_sync_step = blade_start
    elif start_step in BLADE_SYNC_STEPS or start_step == blade_start:
        next_sync_step = start_step
    else:
        # Between scheduled syncs (or past them): one-shot resume sync at start_step.
        next_sync_step = start_step
    if rank == 0:
        log.info(
            "BLADE sync schedule=%s next_sync_step=%s (blade_start=%d start_step=%d)",
            list(BLADE_SYNC_STEPS),
            next_sync_step if next_sync_step < 10**9 else None,
            blade_start,
            start_step,
        )
    t0 = time.time()
    window_t0 = t0
    window_step0 = start_step
    loss_path = progress_dir / "train_loss.jsonl"

    if is_distributed():
        dist.barrier()

    for step in range(start_step, total_steps):
        books.global_step = step
        books.global_train_tokens_seen = step * tokens_per_step
        in_blade = step >= blade_start

        # Lazy-init reference at first BLADE sync (keeps warmup memory like GPU7 CE).
        if in_blade and reference is None:
            if rank == 0:
                log.info(
                    "Allocating reference model at blade_start=%d (ref_mbs=%d)",
                    blade_start,
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

        if in_blade and step == next_sync_step:
            assert reference is not None and ref_opt is not None
            if rank == 0:
                log.info("=== BLADE sync at step %d: reference ← proxy, K=%d ===", step, args.K)
            train_module.zero_grads()
            cuda_gc()
            sync_reference_from_proxy(reference, train_module)
            # Match proxy LR at this step for the K reference updates.
            for g in ref_opt.param_groups:
                g["lr"] = lr  # SkipStep schedule lives on proxy; ref uses peak LR
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
            # After blade_start (if unscheduled) or any sync, advance to next fixed step.
            next_sync_step = next_blade_sync_step(step, inclusive=False)
            if rank == 0:
                log.info(
                    "Next BLADE sync scheduled at step %s",
                    next_sync_step if next_sync_step < 10**9 else None,
                )
            if is_distributed():
                dist.barrier()

        input_ids = next_rank_input_ids(train_stream, seqs_per_rank, device)
        batch: Dict[str, torch.Tensor] = {"input_ids": input_ids}
        select_frac = 1.0
        if in_blade:
            assert reference is not None
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

        global_step = step + 1
        if global_step % args.log_interval == 0 or global_step == 1:
            now = time.time()
            elapsed = now - t0
            done = max(1, global_step - start_step)
            tok_s_avg = done * tokens_per_step / max(elapsed, 1e-6)
            # Recent window (since last log) — excludes compile / sync stalls.
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
                        }
                    )
                    + "\n"
                )

        if global_step % args.save_interval == 0 or global_step == total_steps:
            if is_distributed():
                dist.barrier()
            if rank == 0:
                save_checkpoint(
                    save_folder / f"step{global_step}",
                    global_step,
                    train_module,
                    args,
                    meta,
                )
            if is_distributed():
                dist.barrier()

    if rank == 0:
        log.info("Training complete at step=%d world_size=%d", total_steps, world_size)


if __name__ == "__main__":
    main()
