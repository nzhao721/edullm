#!/usr/bin/env python3
"""Plain CE control pretraining on RegMix 10B (token-selection baseline).

Matches BLADE proxy **warmup** architecture + wall-clock train stack
(``train_blade_olmo_370m.py``), without BLADE token selection / reference:

  * ``TransformerConfig.olmo2_370M`` (full attn, no SWA)
  * ``TransformerTrainModule`` — HSDP bf16, SkipStepAdamW, CosWithWarmup,
    ``compile_model=True``, ``z_loss_multiplier=1e-5``
  * fused LM-head CE when liger-kernel is available (no full logits)
  * sequence=2048, global_batch=4_194_304, rank_microbatch=65_536
  * train corpus: RegMix 10B (same as BLADE proxy)
  * persistent checkpoints every 250 steps (all kept)
  * ephemeral checkpoints every 50 steps (latest only; overwritten)
  * multi-GPU via ``torchrun`` + HSDP

Launch via ``torchrun`` + your own launcher script. Does **not** call AWS.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

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
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from olmo_core.config import DType
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

log = logging.getLogger("train_ce_regmix_olmo2_370m")

SEQ_LEN = 2048
EMBEDDING_SIZE = 100_352
GLOBAL_BATCH_TOKENS = 4_194_304
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4


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
# Data (same memmap path as BLADE proxy)
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
# Model / train module (BLADE warmup / GPU7 stack)
# ---------------------------------------------------------------------------


def resolve_attn_backend() -> AttentionBackendName:
    """Prefer flash_attn when available; else PyTorch SDPA.

    Env ``OLMO_ATTN_BACKEND``: ``auto`` (default), ``flash_2``/``flash``, or ``torch``.
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
    """Enable olmo_core fused LM-head CE via liger-kernel (same CE, no full logits)."""
    try:
        import liger_kernel  # noqa: F401
    except Exception:
        log.warning("liger-kernel not installed; CE uses default LM-head path")
        return False
    if not patch_liger_fused_ce_compat():
        log.warning("liger present but fused CE compat patch failed; leaving default CE")
        return False
    log.info("liger-kernel available — enabling fused_linear CE")
    return True


def build_train_module(
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
        "Built TransformerTrainModule (HSDP bf16, SkipStepAdamW, compile=%s, fused_ce=%s)",
        compile_model,
        fused,
    )
    return train_module


# ---------------------------------------------------------------------------
# Checkpointing
#   persistent: step{N}/ every --save-interval (kept forever)
#   ephemeral:  ephemeral/step{N}/ every --ephemeral-save-interval (latest only)
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    step: int,
    train_module: TransformerTrainModule,
    args: argparse.Namespace,
    meta: dict,
    *,
    ephemeral: bool = False,
) -> None:
    """All ranks must call ``state_dict_to_save`` (HSDP gather); rank 0 writes."""
    state_tm = train_module.state_dict_to_save()
    if get_rank() != 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "train_module": state_tm,
        "args": vars(args),
        "meta": meta,
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (BLADE-warmup-matched)",
        "method": "plain_ce",
        "ephemeral": bool(ephemeral),
    }
    tmp = path / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(path / "state.pt")
    (path / "step.txt").write_text(str(step) + "\n")
    kind = "ephemeral" if ephemeral else "persistent"
    log.info("Saved %s checkpoint → %s (step=%s)", kind, path, step)


def prune_old_ephemeral(ephemeral_root: Path, keep_step: int) -> None:
    """Delete prior ephemeral step dirs; keep only ``keep_step``."""
    if not ephemeral_root.is_dir():
        return
    for p in list(ephemeral_root.iterdir()):
        if not p.is_dir() or not p.name.startswith("step"):
            continue
        try:
            step = int(p.name.replace("step", "").split("-")[0])
        except ValueError:
            continue
        if step != keep_step:
            shutil.rmtree(p, ignore_errors=True)
            log.info("Pruned ephemeral checkpoint %s", p)


def load_checkpoint(path: Path, train_module: TransformerTrainModule) -> int:
    ckpt = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    train_module.load_state_dict(ckpt["train_module"])
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


def _checkpoint_step(path: Path) -> int:
    return int(path.name.replace("step", "").split("-")[0])


def find_latest_checkpoint(save_folder: Path) -> Optional[Path]:
    """Prefer newest among persistent ``step*`` and ephemeral ``ephemeral/step*``."""
    cands: List[Path] = []
    if save_folder.is_dir():
        cands.extend(
            p
            for p in save_folder.iterdir()
            if p.is_dir() and p.name.startswith("step") and (p / "state.pt").is_file()
        )
    eph = save_folder / "ephemeral"
    if eph.is_dir():
        cands.extend(
            p
            for p in eph.iterdir()
            if p.is_dir() and p.name.startswith("step") and (p / "state.pt").is_file()
        )
    if not cands:
        return None
    return max(cands, key=_checkpoint_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-paths-file", required=True)
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--length-tokens", type=int, required=True)
    ap.add_argument(
        "--device-batch-size",
        type=int,
        default=MICROBATCH_TOKENS // SEQ_LEN,
        help="Sequences per microbatch (default 32 = 65536 tokens)",
    )
    ap.add_argument("--save-interval", type=int, default=250, help="Persistent checkpoint every N steps")
    ap.add_argument(
        "--ephemeral-save-interval",
        type=int,
        default=50,
        help="Non-persistent checkpoint every N steps (keeps only the latest under ephemeral/)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=6198)
    ap.add_argument("--load-path", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=0.1)
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
    rank_micro_tokens = mbs * SEQ_LEN
    if GLOBAL_BATCH_TOKENS % (world_size * rank_micro_tokens) != 0:
        raise SystemExit(
            f"global_batch_tokens {GLOBAL_BATCH_TOKENS} not divisible by "
            f"world_size*rank_micro ({world_size}*{rank_micro_tokens})"
        )
    seqs_per_rank = GLOBAL_BATCH_TOKENS // (SEQ_LEN * world_size)
    tokens_per_step = GLOBAL_BATCH_TOKENS
    total_steps = int(args.length_tokens) // tokens_per_step
    lr = float(PEAK_LR)

    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "matched_teammate_run": "rel-ema-5b-scratch-v1 / BLADE warmup stack",
        "method": "plain_ce",
        "control_for": "BLADE / token-selection experiments",
        "train_stack": "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
        "loss": "LM-head CE (fused_linear when liger available)",
        "length_tokens": int(args.length_tokens),
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "sequence_length": SEQ_LEN,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "device_microbatch_seqs": mbs,
        "seqs_per_rank": seqs_per_rank,
        "world_size": world_size,
        "lr": lr,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_alpha_f": float(args.lr_alpha_f),
        "compile": bool(args.compile),
        "attn_backend": str(resolve_attn_backend()),
        "save_interval": int(args.save_interval),
        "ephemeral_save_interval": int(args.ephemeral_save_interval),
        "checkpoints": (
            f"persistent every {int(args.save_interval)}; "
            f"ephemeral every {int(args.ephemeral_save_interval)} (latest only)"
        ),
        "train_dataset": "s3://edullm-dataset-regmix/regmix-10b/",
        "seed": args.seed,
    }
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "run_plan.txt").write_text(
            "\n".join(
                [
                    "architecture=olmo_core.TransformerConfig.olmo2_370M",
                    "method=plain_ce",
                    "train_stack=TransformerTrainModule/HSDP/SkipStepAdamW (BLADE-warmup-matched)",
                    f"world_size={world_size}",
                    f"sequence_length={SEQ_LEN}",
                    f"total_steps={total_steps}",
                    f"global_batch_tokens={GLOBAL_BATCH_TOKENS}  mbs_seqs={mbs}  "
                    f"seqs_per_rank={seqs_per_rank}",
                    f"lr={lr}  cos_warmup={args.lr_warmup_steps}  alpha_f={args.lr_alpha_f}",
                    f"compile={args.compile}",
                    f"save_interval={args.save_interval} (persistent)",
                    f"ephemeral_save_interval={args.ephemeral_save_interval} (latest only)",
                    "train=s3://edullm-dataset-regmix/regmix-10b/",
                    "",
                ]
            )
        )
        log.info(
            "Plan: plain CE olmo2_370M world=%d total=%d mbs=%d seqs/rank=%d lr=%.3e",
            world_size,
            total_steps,
            mbs,
            seqs_per_rank,
            lr,
        )

    train_paths = read_paths(Path(args.train_paths_file))
    train_ds = MemmapTokenDataset(train_paths, SEQ_LEN)
    workers = args.num_workers if world_size == 1 else max(1, args.num_workers // world_size)
    train_stream = InfiniteBatchStream(
        train_ds, mbs, workers, args.seed, rank=rank, world_size=world_size
    )

    train_module = build_train_module(
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

    start_step = 0
    if args.fresh:
        if rank == 0:
            log.info("--fresh: starting from scratch")
    else:
        load_dir = Path(args.load_path) if args.load_path else find_latest_checkpoint(save_folder)
        if load_dir is not None:
            start_step = load_checkpoint(load_dir, train_module)

    t0 = time.time()
    window_t0 = t0
    window_step0 = start_step
    loss_path = progress_dir / "train_loss.jsonl"

    if is_distributed():
        dist.barrier()

    for step in range(start_step, total_steps):
        books.global_step = step
        books.global_train_tokens_seen = step * tokens_per_step

        input_ids = next_rank_input_ids(train_stream, seqs_per_rank, device)
        batch: Dict[str, torch.Tensor] = {"input_ids": input_ids}

        train_module.zero_grads()
        train_module.train_batch(batch)
        train_module.optim_step()

        global_step = step + 1
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
            if rank == 0:
                log.info(
                    "step=%d/%d phase=ce tok/s=%.0f (avg=%.0f) world=%d",
                    global_step,
                    total_steps,
                    tok_s,
                    tok_s_avg,
                    world_size,
                )
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "phase": "ce",
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
                            "phase": "ce",
                            "world_size": world_size,
                            "tok_per_s": tok_s,
                            "tok_per_s_avg": tok_s_avg,
                            "pct": round(100.0 * global_step / total_steps, 4),
                        }
                    )
                    + "\n"
                )

        persist = global_step % args.save_interval == 0 or global_step == total_steps
        eph_iv = int(args.ephemeral_save_interval)
        ephemeral = eph_iv > 0 and global_step % eph_iv == 0 and not persist
        if persist or ephemeral:
            if is_distributed():
                dist.barrier()
            if persist:
                save_checkpoint(
                    save_folder / f"step{global_step}",
                    global_step,
                    train_module,
                    args,
                    meta,
                    ephemeral=False,
                )
            else:
                eph_root = save_folder / "ephemeral"
                save_checkpoint(
                    eph_root / f"step{global_step}",
                    global_step,
                    train_module,
                    args,
                    meta,
                    ephemeral=True,
                )
                if get_rank() == 0:
                    prune_old_ephemeral(eph_root, global_step)
            if is_distributed():
                dist.barrier()

    if rank == 0:
        log.info("Training complete at step=%d world_size=%d", total_steps, world_size)


if __name__ == "__main__":
    main()
