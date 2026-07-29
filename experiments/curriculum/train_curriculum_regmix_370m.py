#!/usr/bin/env python3
"""Curriculum / control CE pretraining on RegMix 10B (OLMo2-370M).

Fork of ``experiments/token-selection/control/train_ce_regmix_olmo_370m.py`` with:

  * Warmup + **constant** LR (``--lr-alpha-f 1.0`` default)
  * ``--pacing`` / ``--difficulty-metric`` selecting the data stream
  * Control arm: random shuffle over flat ``tokenized/`` memmaps
  * Curriculum arms: ``CurriculumChunkStream`` over the ``curriculum/`` index

Architecture / hparams match the control / RefHQ contract (olmo2_370M, GBS
4_194_304, SkipStepAdamW, z_loss 1e-5, HSDP bf16, compile). Permanent
checkpoint ladder and task_loss hooks are imported from
``token_selection.olmo_ext`` — not duplicated.

Does **not** submit AWS jobs or mutate S3 unless an optional export helper is
present and succeeds (same best-effort pattern as control).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

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

_CUR_ROOT = Path(__file__).resolve().parent
_TS_ROOT = Path(__file__).resolve().parents[1] / "token-selection"
if str(_CUR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUR_ROOT))
if str(_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TS_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
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
from token_selection.olmo_ext.task_loss_hook import trigger_task_loss_eval

from curriculum_pacing import (
    DIFFICULTY_METRICS,
    PACING_NAMES,
    TOTAL_STEPS,
    CurriculumChunkStream,
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

log = logging.getLogger("train_curriculum_regmix_370m")

SEQ_LEN = 2048
TOKENIZER_ID = "allenai/dolma2-tokenizer"
EMBEDDING_SIZE = 100_352
GLOBAL_BATCH_TOKENS = 4_194_304
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4
DEFAULT_SEED = 42
DEFAULT_LENGTH_TOKENS = 10_000_058_051  # → 2384 steps at GBS 4_194_304
CONFIG_NAME = "OLMo-2-370M-scratch"
CHECKPOINT_BUCKET = "edullm-checkpoints"
CURRICULUM_S3_ROOT = "curriculum"


def arm_s3_prefix(arm_id: str) -> str:
    arm = str(arm_id).strip().strip("/")
    if not arm:
        raise ValueError("arm_id must be non-empty")
    return f"{CURRICULUM_S3_ROOT}/{arm}"


@dataclass
class _Bookkeeping:
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


class CurriculumIndexedDataset(Dataset):
    """Random-access by ``global_chunk_idx`` using chunk_index + domain memmaps."""

    def __init__(
        self,
        curriculum_root: Path,
        *,
        chunk_size: int = SEQ_LEN,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.root = Path(curriculum_root)
        chunk_path = self.root / "chunk_index.jsonl.gz"
        if not chunk_path.is_file():
            # Allow plain jsonl for tests/fixtures.
            alt = self.root / "chunk_index.jsonl"
            if not alt.is_file():
                raise SystemExit(f"missing chunk index under {self.root}")
            chunk_path = alt
        import gzip

        open_fn = gzip.open if str(chunk_path).endswith(".gz") else open
        rows: List[dict] = []
        with open_fn(chunk_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            raise SystemExit(f"empty chunk index: {chunk_path}")
        # Index by global_chunk_idx
        self._by_gidx: Dict[int, dict] = {}
        for r in rows:
            self._by_gidx[int(r["global_chunk_idx"])] = r
        self._mmaps: Dict[str, np.memmap] = {}
        self._n = max(self._by_gidx) + 1

    def __len__(self) -> int:
        return self._n

    def _mmap(self, rel: str) -> np.memmap:
        if rel not in self._mmaps:
            path = self.root / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            self._mmaps[rel] = np.memmap(path, mode="r", dtype=np.uint32)
        return self._mmaps[rel]

    def __getitem__(self, global_chunk_idx: int) -> torch.Tensor:
        meta = self._by_gidx[int(global_chunk_idx)]
        mm = self._mmap(meta["memmap"])
        start = int(meta["token_offset"])
        arr = np.asarray(mm[start : start + self.chunk_size], dtype=np.int64)
        if len(arr) < self.chunk_size:
            raise IndexError(
                f"short chunk global_chunk_idx={global_chunk_idx} offset={start} len={len(arr)}"
            )
        return torch.from_numpy(arr.copy())


def load_ranked_chunks(curriculum_root: Path, metric: str) -> np.ndarray:
    path = Path(curriculum_root) / f"ranked_chunks_{metric}.npy"
    if not path.is_file():
        raise SystemExit(f"missing ranked chunk array for metric={metric}: {path}")
    return np.load(path)


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


class CurriculumBatchStream:
    """Pulls ``seqs_per_rank`` sequences for the current global step via pacing."""

    def __init__(
        self,
        indexed: CurriculumIndexedDataset,
        stream: CurriculumChunkStream,
        *,
        seqs_per_rank: int,
        device: torch.device,
    ) -> None:
        self.indexed = indexed
        self.stream = stream
        self.seqs_per_rank = int(seqs_per_rank)
        self.device = device

    def next_input_ids(self, step: int) -> torch.Tensor:
        idxs = self.stream.next_indices(step, self.seqs_per_rank)
        tensors = [self.indexed[i] for i in idxs]
        return torch.stack(tensors, dim=0).to(self.device, non_blocking=True)


def next_rank_input_ids(stream: InfiniteBatchStream, n_seqs: int, device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    got = 0
    while got < n_seqs:
        x = stream.next_batch()["input_ids"]
        chunks.append(x)
        got += x.size(0)
    return torch.cat(chunks, dim=0)[:n_seqs].to(device, non_blocking=True)


def resolve_attn_backend() -> AttentionBackendName:
    prefer = os.environ.get("OLMO_ATTN_BACKEND", "torch").strip().lower()
    if prefer in ("torch", "sdpa", "eager"):
        return AttentionBackendName.torch
    if prefer in ("flash_2", "flash", "flash2", "auto"):
        try:
            import flash_attn  # noqa: F401

            backend = AttentionBackendName.flash_2
            backend.get_class().assert_supported()
            log.info("attn_backend=flash_2")
            return backend
        except Exception as e:
            log.warning("flash_attn unavailable (%s); using torch", e)
            return AttentionBackendName.torch
    try:
        return AttentionBackendName(prefer)
    except Exception:
        log.warning("Unknown OLMO_ATTN_BACKEND=%s; using torch", prefer)
        return AttentionBackendName.torch


def build_olmo2_config(*, fused_ce: bool) -> TransformerConfig:
    vocab_size = TokenizerConfig.dolma2().padded_vocab_size()
    if vocab_size != EMBEDDING_SIZE:
        raise SystemExit(
            f"dolma2 padded vocab {vocab_size} != expected EMBEDDING_SIZE {EMBEDDING_SIZE}"
        )
    cfg = TransformerConfig.olmo2_370M(
        vocab_size=vocab_size,
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
        log.warning("liger-kernel not installed; CE uses default LM-head path")
        return False
    if not patch_liger_fused_ce_compat():
        log.warning("liger present but fused CE compat patch failed; leaving default CE")
        return False
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
        "Built TransformerTrainModule (HSDP bf16, SkipStepAdamW, compile=%s, fused_ce=%s, alpha_f=%s)",
        compile_model,
        fused,
        alpha_f,
    )
    return train_module


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


def save_checkpoint(
    path: Path,
    step: int,
    train_module: TransformerTrainModule,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    train_module_sd = gather_train_module_state_dict(train_module)
    if get_rank() != 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "train_module": train_module_sd,
        "args": vars(args),
        "meta": meta,
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (curriculum)",
        "method": "plain_ce" if args.pacing == "control" else f"curriculum:{args.pacing}",
        "arm": args.arm_id,
        "run_id": args.name,
        "checkpoint_format": "full_state_dict_v1",
    }
    tmp = path / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(path / "state.pt")
    (path / "step.txt").write_text(str(step) + "\n")
    log.info("Saved permanent full checkpoint → %s (step=%s)", path, step)


def load_checkpoint(path: Path, train_module: TransformerTrainModule) -> int:
    ckpt = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    tm_sd = ckpt["train_module"]
    fmt = ckpt.get("checkpoint_format")
    if (
        fmt == "full_state_dict_v1"
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
    step = int(ckpt["step"])
    log.info("Resumed from %s at step=%s format=%s", path, step, fmt or "legacy_sharded")
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
    if not save_folder.is_dir():
        return None
    cands = [
        p
        for p in save_folder.iterdir()
        if p.is_dir() and p.name.startswith("step") and (p / "state.pt").is_file()
    ]
    if not cands:
        return None
    return max(cands, key=_checkpoint_step)


def _maybe_task_loss(args: argparse.Namespace, ckpt_dir: Path, step: int) -> None:
    if get_rank() != 0:
        return
    results_dir = Path(args.task_loss_results_dir)
    trigger_task_loss_eval(
        ckpt_dir,
        run_name=f"{args.name}-step{step}",
        out_path=results_dir / f"step{step}_task_loss.json",
        eval_script=args.task_loss_eval_script,
        enabled=None if args.task_loss_on_save else False,
    )


def default_arm_id(pacing: str, metric: Optional[str]) -> str:
    if pacing == "control":
        return "control"
    metric = metric or "compression_ratio"
    pacing_map = {
        "linear_n10": "linear10",
        "expanding_25_1000": "expand",
        "warmup_1000": "warmup",
        "interleave_i10_linear": "interleave",
    }
    metric_map = {
        "compression_ratio": "cr",
        "flesch": "flesch",
        "mtld": "mtld",
        "learnability": "learn",
    }
    return f"{pacing_map[pacing]}-{metric_map[metric]}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=None, help="Run id (default: arm_id)")
    ap.add_argument(
        "--arm-id",
        default=None,
        help="Arm id for S3 layout / logging (default derived from pacing+metric)",
    )
    ap.add_argument(
        "--pacing",
        choices=list(PACING_NAMES),
        default="control",
        help="Data pacing schedule (control = flat memmap shuffle)",
    )
    ap.add_argument(
        "--difficulty-metric",
        choices=list(DIFFICULTY_METRICS),
        default=None,
        help="Required for non-control pacing",
    )
    ap.add_argument(
        "--train-paths-file",
        type=str,
        default=None,
        help="Required for control: list of flat tokenized memmap paths",
    )
    ap.add_argument(
        "--curriculum-index",
        type=str,
        default=None,
        help="Required for curriculum arms: local curriculum/ root",
    )
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--length-tokens", type=int, default=DEFAULT_LENGTH_TOKENS)
    ap.add_argument(
        "--device-batch-size",
        type=int,
        default=MICROBATCH_TOKENS // SEQ_LEN,
        help="Sequences per microbatch (default 32 = 65536 tokens)",
    )
    ap.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Permanent ladder interval (default 125)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--load-path", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument(
        "--lr-alpha-f",
        type=float,
        default=1.0,
        help="CosWithWarmup alpha_f (1.0 = constant LR after warmup)",
    )
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--task-loss-results-dir", type=str, default=None)
    ap.add_argument("--task-loss-eval-script", type=str, default=None)
    ap.add_argument(
        "--task-loss-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()
    if args.pacing != "control" and not args.difficulty_metric:
        raise SystemExit("--difficulty-metric is required when --pacing != control")
    if args.pacing == "control" and not args.train_paths_file:
        raise SystemExit("--train-paths-file is required for control pacing")
    if args.pacing != "control" and not args.curriculum_index:
        raise SystemExit("--curriculum-index is required for curriculum pacing")
    if args.arm_id is None:
        args.arm_id = default_arm_id(args.pacing, args.difficulty_metric)
    if args.name is None:
        args.name = args.arm_id
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        args.task_loss_results_dir = str(_CUR_ROOT / "task_loss_results" / args.arm_id)
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
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device("cuda")
    seed_all(args.seed + rank)

    mbs = int(args.device_batch_size)
    rank_micro_tokens = mbs * SEQ_LEN
    if GLOBAL_BATCH_TOKENS % (world_size * rank_micro_tokens) != 0:
        raise SystemExit(
            f"global_batch_tokens {GLOBAL_BATCH_TOKENS} not divisible by "
            f"world_size*rank_micro ({world_size}*{rank_micro_tokens}). "
            f"Adjust --device-batch-size or WORLD_SIZE so the product divides evenly."
        )
    seqs_per_rank = GLOBAL_BATCH_TOKENS // (SEQ_LEN * world_size)
    tokens_per_step = GLOBAL_BATCH_TOKENS
    total_steps = int(args.length_tokens) // tokens_per_step
    ladder = permanent_checkpoint_steps(total_steps, int(args.save_interval))
    ladder_set: Set[int] = set(ladder)
    lr = float(PEAK_LR)
    s3_prefix = arm_s3_prefix(args.arm_id)

    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        Path(args.task_loss_results_dir).mkdir(parents=True, exist_ok=True)

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "arm": args.arm_id,
        "pacing": args.pacing,
        "difficulty_metric": args.difficulty_metric,
        "method": "plain_ce" if args.pacing == "control" else f"curriculum:{args.pacing}",
        "run_id": args.name,
        "s3_prefix": s3_prefix,
        "s3_uri": f"s3://{CHECKPOINT_BUCKET}/{s3_prefix}/",
        "train_stack": "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
        "tokenizer": TOKENIZER_ID,
        "vocab_size": EMBEDDING_SIZE,
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
        "z_loss_multiplier": 1e-5,
        "max_grad_norm": 1.0,
        "compile": bool(args.compile),
        "attn_backend": str(resolve_attn_backend()),
        "save_interval": int(args.save_interval),
        "permanent_checkpoint_steps": ladder,
        "max_checkpoints": None,
        "ephemeral": False,
        "train_dataset": (
            "s3://edullm-datasets/regmix/regmix-10b/tokenized/"
            if args.pacing == "control"
            else "s3://edullm-datasets/regmix/regmix-10b/curriculum/"
        ),
        "seed": args.seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_results_dir": args.task_loss_results_dir,
        "ema_merge_steps": [2000, 2125, 2250, 2384],
        "ema_alpha": 0.8,
    }
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "checkpoint_ladder.json").write_text(
            json.dumps({"steps": ladder, "interval": int(args.save_interval)}, indent=2) + "\n"
        )
        log.info(
            "Plan: arm=%s pacing=%s metric=%s world=%d total=%d ladder_n=%d "
            "mbs=%d seqs/rank=%d lr=%.3e alpha_f=%s",
            args.arm_id,
            args.pacing,
            args.difficulty_metric,
            world_size,
            total_steps,
            len(ladder),
            mbs,
            seqs_per_rank,
            lr,
            args.lr_alpha_f,
        )
        if total_steps == TOTAL_STEPS and 2375 in ladder_set:
            raise SystemExit("BUG: ladder for 2384 must omit 2375")

    control_stream: Optional[InfiniteBatchStream] = None
    curr_stream: Optional[CurriculumBatchStream] = None
    if args.pacing == "control":
        train_paths = read_paths(Path(args.train_paths_file))
        train_ds = MemmapTokenDataset(train_paths, SEQ_LEN)
        workers = args.num_workers if world_size == 1 else max(1, args.num_workers // world_size)
        control_stream = InfiniteBatchStream(
            train_ds, mbs, workers, args.seed, rank=rank, world_size=world_size
        )
    else:
        assert args.curriculum_index and args.difficulty_metric
        indexed = CurriculumIndexedDataset(Path(args.curriculum_index), chunk_size=SEQ_LEN)
        ranked = load_ranked_chunks(Path(args.curriculum_index), args.difficulty_metric)
        pacing_stream = CurriculumChunkStream(
            ranked,
            pacing=args.pacing,
            difficulty_metric=args.difficulty_metric,
            total_steps=total_steps,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
        )
        curr_stream = CurriculumBatchStream(
            indexed, pacing_stream, seqs_per_rank=seqs_per_rank, device=device
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

    if start_step == 0 and 0 in ladder_set:
        if is_distributed():
            dist.barrier()
        ckpt0 = save_folder / "step0"
        save_checkpoint(ckpt0, 0, train_module, args, meta)
        _maybe_task_loss(args, ckpt0, 0)
        if is_distributed():
            dist.barrier()

    for step in range(start_step, total_steps):
        books.global_step = step
        books.global_train_tokens_seen = step * tokens_per_step

        if control_stream is not None:
            input_ids = next_rank_input_ids(control_stream, seqs_per_rank, device)
        else:
            assert curr_stream is not None
            input_ids = curr_stream.next_input_ids(step)
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
                    "step=%d/%d pacing=%s tok/s=%.0f (avg=%.0f) world=%d",
                    global_step,
                    total_steps,
                    args.pacing,
                    tok_s,
                    tok_s_avg,
                    world_size,
                )
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "pacing": args.pacing,
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
                            "pacing": args.pacing,
                            "world_size": world_size,
                            "tok_per_s": tok_s,
                            "tok_per_s_avg": tok_s_avg,
                            "pct": round(100.0 * global_step / total_steps, 4),
                        }
                    )
                    + "\n"
                )

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            ckpt_dir = save_folder / f"step{global_step}"
            save_checkpoint(ckpt_dir, global_step, train_module, args, meta)
            _maybe_task_loss(args, ckpt_dir, global_step)
            if is_distributed():
                dist.barrier()

    if rank == 0:
        log.info(
            "Training complete at step=%d world_size=%d arm=%s",
            total_steps,
            world_size,
            args.arm_id,
        )


if __name__ == "__main__":
    main()
