#!/usr/bin/env python3
"""Random 60% keep-rate control pretraining on RegMix 10B (token-selection baseline).

Architecture matches RefHQ CE exactly (``reference/train_olmo3_370m_refhq.py``):

  * ``TransformerConfig.olmo2_370M`` — d_model=1024, 16L/16H, full attn (no SWA)
  * vocab 100352, seq 2048, GBS 4_194_304, rank microbatch 65_536
  * SkipStepAdamW + CosWithWarmup LR 4e-4 warmup 24 alpha_f 0.1
  * z_loss 1e-5, compile_model=True, from scratch
  * HSDP bf16 train module; world size from torchrun (no hardcoded GPU count)

Permanent checkpoint ladder (shared helper): step 0, every 125, final;
omit last on-grid if within 125 of final (2360 → skip 2250).
Post-save hook launches full 20-label OLMo-ladder task_loss_bpb eval.

Ephemeral runtime: scratch may start empty and be wiped after the job. Data is
resolved from published+validated ``pretrain/regmix-10b`` on ``s3://edullm-data/``
via ``edullm_data.read.resolve_latest`` + ``dataset_paths``; rank 0 stages
``.u32le.bin`` shards into ``--stage-dir`` (size-checked). Token budget defaults
to one epoch under published train (``9900000000`` → 2360 steps) so training does
not wrap past the corpus to force 10B. Does **not** read
``s3://edullm-datasets/`` or assume FarmShare/laptop corpora already present.

Artifacts remain on runtime scratch and upload to W&B. Production online
checkpoint uploads are fail-closed. Resume via ``--wandb-resume-artifact`` or
explicit local ``--load-path``. Local auto-resume is off unless
``ALLOW_LOCAL_RESUME=1``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import urlparse

# Shared package lives under experiments/token-selection/
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
from token_selection.olmo_ext.train_module import TokenSelectConfig, TokenSelectTrainModule

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

log = logging.getLogger("train_ce_regmix_olmo2_370m")

SEQ_LEN = 2048
TOKENIZER_ID = "allenai/dolma2-tokenizer"
EMBEDDING_SIZE = 100_352  # TokenizerConfig.dolma2().padded_vocab_size()
GLOBAL_BATCH_TOKENS = 4_194_304
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4
DEFAULT_SEED = 6198
MASK_KEEP_RATE = 0.6
# Clean new run_id — do not append to old edullm-370M-ce-regmix10b prefix.
DEFAULT_RUN_ID = "control-regmix10b-v2"
DEFAULT_LENGTH_TOKENS = 9_900_000_000  # → 2360 steps at GBS 4_194_304 (one-epoch matrix)
DEFAULT_DATASET_ID = "pretrain/regmix-10b"
CONFIG_NAME = "OLMo-2-370M-scratch"
ARM = "control"
DATA_BUCKET = "edullm-data"
LEGACY_DATA_BUCKET = "edullm-datasets"


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


class MemmapTokenDataset(Dataset):
    """Contiguous SEQ_LEN chunks over one or more fixed-width token memmaps."""

    def __init__(
        self,
        paths: List[str],
        chunk_size: int = SEQ_LEN,
        *,
        dtype: Any = np.uint32,
        header_bytes: int = 0,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        np_dtype = np.dtype(dtype)
        offset = int(header_bytes)
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=np_dtype, offset=offset)
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


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise SystemExit(f"Expected s3:// URI from dataset_paths, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _reject_legacy_uri(uri: str) -> None:
    norm = uri.replace("\\", "/")
    if f"s3://{LEGACY_DATA_BUCKET}/" in norm or norm.startswith(f"{LEGACY_DATA_BUCKET}/"):
        raise SystemExit(
            f"Refusing legacy dataset URI under s3://{LEGACY_DATA_BUCKET}/: {uri}. "
            f"Use published s3://{DATA_BUCKET}/ via --dataset-id."
        )


def resolve_published_train(
    *,
    dataset_id: str,
    version: Optional[str],
) -> Any:
    """Resolve train shard URIs + dtype from validated edullm-data."""
    try:
        from edullm_data.read import NotValidated, ReadError, dataset_paths, resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as e:
        raise SystemExit(
            "edullm-data package required to resolve training shards. "
            'Install with: uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0" '
            f"(import failed: {e})"
        ) from e

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3, data_bucket=DATA_BUCKET)
    if not ver:
        raise SystemExit(
            f"Dataset {dataset_id!r} is not published in s3://{DATA_BUCKET}/_catalog/. "
            f"Do not use s3://{LEGACY_DATA_BUCKET}/ or pre-staged FarmShare/laptop paths."
        )
    try:
        resolved = dataset_paths(
            dataset_id,
            ver,
            split="train",
            s3=s3,
            data_bucket=DATA_BUCKET,
            require_validated=True,
        )
    except NotValidated as exc:
        raise SystemExit(
            f"{dataset_id}/{ver} has no _VALIDATED.json — refusing unvalidated data: {exc}"
        ) from exc
    except ReadError as exc:
        raise SystemExit(f"Cannot resolve {dataset_id}/{ver} split=train: {exc}") from exc
    if not resolved.paths:
        raise SystemExit(f"No train shards for {dataset_id}/{ver}")
    for uri in resolved.paths:
        _reject_legacy_uri(uri)
    return resolved


def stage_s3_uris(
    uris: List[str],
    stage_dir: Path,
    *,
    dataset_id: str,
    version: str,
) -> List[str]:
    """Download s3://edullm-data shards under stage_dir (skip when local size matches HEAD)."""
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )
    prefix = f"{dataset_id}/{version}/"
    stage_dir.mkdir(parents=True, exist_ok=True)
    local_paths: List[str] = []
    downloaded = 0
    skipped = 0
    for uri in uris:
        _reject_legacy_uri(uri)
        bucket, key = _parse_s3_uri(uri)
        if bucket != DATA_BUCKET:
            raise SystemExit(f"Only s3://{DATA_BUCKET}/ staging is allowed, got: {uri}")
        if key.startswith(prefix):
            rel = key[len(prefix) :]
        else:
            rel = Path(key).name
        dest = stage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        head = client.head_object(Bucket=bucket, Key=key)
        remote_size = int(head["ContentLength"])
        if dest.is_file() and dest.stat().st_size == remote_size:
            skipped += 1
        else:
            tmp = dest.with_suffix(dest.suffix + ".partial")
            if tmp.exists():
                tmp.unlink()
            client.download_file(bucket, key, str(tmp))
            got = tmp.stat().st_size
            if got != remote_size:
                tmp.unlink(missing_ok=True)
                raise SystemExit(
                    f"Incomplete download for {uri}: got {got}, expected {remote_size}"
                )
            tmp.replace(dest)
            downloaded += 1
        local_paths.append(str(dest.resolve()))
    log.info(
        "Staged %d shards under %s (downloaded=%d skipped=%d)",
        len(local_paths),
        stage_dir,
        downloaded,
        skipped,
    )
    return local_paths


def prepare_train_memmaps(args: argparse.Namespace) -> Tuple[List[str], Dict[str, Any]]:
    """Return local memmap paths + data provenance for run meta.

    Prefer ``--train-paths-file`` when set (prior stage from this arm). Otherwise
    resolve ``--dataset-id`` from edullm-data and stage into ``--stage-dir``.
    """
    data_meta: Dict[str, Any] = {
        "dataset_id": None,
        "dataset_version": None,
        "dataset_uri": None,
        "dtype": "uint32",
        "numpy_dtype": "<u4",
        "header_bytes": 0,
        "rows": None,
        "n_shards": None,
        "stage_dir": None,
        "train_paths_file": None,
        "source": None,
        "bucket": DATA_BUCKET,
    }
    if args.train_paths_file:
        paths = read_paths(Path(args.train_paths_file))
        for p in paths:
            norm = p.replace("\\", "/")
            if LEGACY_DATA_BUCKET in norm:
                raise SystemExit(
                    f"Refusing legacy {LEGACY_DATA_BUCKET} path in --train-paths-file: {p}"
                )
            if not Path(p).is_file():
                raise SystemExit(
                    f"Missing shard listed in --train-paths-file: {p}. "
                    "Omit --train-paths-file and pass --stage-dir to fetch from edullm-data."
                )
        data_meta["train_paths_file"] = str(Path(args.train_paths_file).resolve())
        data_meta["n_shards"] = len(paths)
        data_meta["dataset_id"] = args.dataset_id
        data_meta["dataset_uri"] = (
            f"s3://{DATA_BUCKET}/{args.dataset_id}/ (paths-file override)"
        )
        data_meta["source"] = "train_paths_file"
        return paths, data_meta

    if not args.stage_dir:
        raise SystemExit(
            "Provide --stage-dir (ephemeral scratch) to fetch edullm-data shards, "
            "or --train-paths-file from a prior stage of prepare_control_data.py / this trainer."
        )

    resolved = resolve_published_train(
        dataset_id=args.dataset_id,
        version=args.dataset_version,
    )
    stage_dir = Path(args.stage_dir)
    paths_file = stage_dir / "paths_train.txt"
    if get_rank() == 0:
        local_paths = stage_s3_uris(
            list(resolved.paths),
            stage_dir,
            dataset_id=resolved.dataset_id,
            version=resolved.version,
        )
        paths_file.write_text("\n".join(local_paths) + "\n", encoding="utf-8")
    if is_distributed():
        dist.barrier()
    paths = read_paths(paths_file)
    data_meta.update(
        {
            "source": "edullm-data",
            "bucket": DATA_BUCKET,
            "dataset_id": resolved.dataset_id,
            "dataset_version": resolved.version,
            "dataset_uri": f"s3://{DATA_BUCKET}/{resolved.dataset_id}/{resolved.version}/",
            "dtype": resolved.dtype or "uint32",
            "numpy_dtype": resolved.numpy_dtype or "<u4",
            "header_bytes": int(resolved.header_bytes or 0),
            "rows": resolved.rows,
            "n_shards": len(paths),
            "stage_dir": str(stage_dir.resolve()),
            "train_paths_file": str(paths_file.resolve()),
        }
    )
    return paths, data_meta


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
    return torch.cat(chunks, dim=0)[:n_seqs].to(device, non_blocking=True)


def resolve_attn_backend() -> AttentionBackendName:
    """Match RefHQ default (torch) unless OLMO_ATTN_BACKEND overrides."""
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
    total_steps: int,
    seed: int,
    mask_keep_rate: float,
) -> TokenSelectTrainModule:
    # Fused CE is intentionally off until a focused production-LM-head parity
    # test proves both value and gradient equivalence.
    fused = False
    model_cfg = build_olmo2_config(fused_ce=fused)
    try:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps, alpha_f=alpha_f)
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps)
        if hasattr(scheduler, "alpha_f"):
            scheduler.alpha_f = alpha_f

    ts_config = TokenSelectConfig(
        method="random",
        k=float(mask_keep_rate),
        t0_steps=0,
        total_steps=int(total_steps),
        seed=int(seed),
    )
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
    train_module = TokenSelectTrainModule(
        model=model,
        optim=tm_cfg.optim,
        rank_microbatch_size=tm_cfg.rank_microbatch_size,
        max_sequence_length=tm_cfg.max_sequence_length,
        compile_model=tm_cfg.compile_model,
        dp_config=tm_cfg.dp_config,
        z_loss_multiplier=tm_cfg.z_loss_multiplier,
        max_grad_norm=tm_cfg.max_grad_norm,
        scheduler=tm_cfg.scheduler,
        ts_config=ts_config,
    )
    log.info(
        "Built TokenSelectTrainModule (random keep=%.2f, HSDP bf16, SkipStepAdamW, "
        "compile=%s, fused_ce=%s)",
        mask_keep_rate,
        compile_model,
        fused,
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
    """All ranks gather; rank 0 materializes a complete local checkpoint."""
    train_module_sd = gather_train_module_state_dict(train_module)
    if get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "train_module": train_module_sd,
            "args": vars(args),
            "meta": meta,
            "architecture": "olmo_core.TransformerConfig.olmo2_370M",
            "config_name": CONFIG_NAME,
            "train_stack": "TokenSelectTrainModule/random/HSDP/SkipStepAdamW (RefHQ-matched)",
            "method": "random",
            "mask_keep_rate": float(getattr(args, "mask_keep_rate", MASK_KEEP_RATE)),
            "arm": ARM,
            "run_id": args.name,
            "checkpoint_format": "full_state_dict_v1",
        }
        tmp = path / "state.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(path / "state.pt")
        (path / "step.txt").write_text(str(step) + "\n")
        n_model = len(train_module_sd.get("model") or {})
        log.info(
            "Saved permanent full checkpoint → %s (step=%s, model_tensors=%d, has_optim=%s)",
            path,
            step,
            n_model,
            train_module_sd.get("optim") is not None,
        )
        from token_selection.olmo_ext.permanent_checkpoint import (
            copy_fingerprint_into_checkpoint,
        )

        copy_fingerprint_into_checkpoint(
            Path(args.save_folder) / "run_fingerprint.json", path
        )

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
    """Newest permanent ``step*`` under save_folder (same-job local only)."""
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


def resolve_resume_dir(args: argparse.Namespace, save_folder: Path) -> Optional[Path]:
    """Resolve a local checkpoint, optionally restoring it from W&B first."""
    if getattr(args, "wandb_resume_artifact", None):
        from token_selection.olmo_ext.wandb_logging import restore_checkpoint_artifact

        return restore_checkpoint_artifact(args.wandb_resume_artifact, save_folder)
    if args.load_path:
        load_dir = Path(args.load_path)
        if not (load_dir / "state.pt").is_file():
            raise SystemExit(
                f"--load-path {load_dir} has no state.pt. "
                "Restore --wandb-resume-artifact or stage a local checkpoint first."
            )
        return load_dir
    allow_local = os.environ.get("ALLOW_LOCAL_RESUME", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if allow_local:
        return find_latest_checkpoint(save_folder)
    return None


def upload_before_end(
    *,
    progress_dir: Path,
    task_loss_dir: Optional[str],
    run: Any | None,
    run_name: str,
    production: bool,
    mode: str,
) -> None:
    """Upload final progress/eval trees to W&B; checkpoints upload per save."""
    from token_selection.olmo_ext.wandb_logging import production_online

    strict = production_online(production=production, mode=mode)
    wandb_log_directory_artifact(
        run,
        progress_dir,
        name=f"{run_name}-progress",
        artifact_type="metrics",
        strict=strict,
    )
    if task_loss_dir:
        wandb_log_directory_artifact(
            run,
            task_loss_dir,
            name=f"{run_name}-task-loss",
            artifact_type="eval",
            strict=strict,
        )


def _maybe_task_loss(args: argparse.Namespace, ckpt_dir: Path, step: int) -> None:
    if get_rank() != 0:
        return
    from token_selection.olmo_ext.permanent_checkpoint import (
        finalize_permanent_checkpoint,
    )

    finalize_permanent_checkpoint(
        arm=ARM,
        checkpoint_dir=ckpt_dir,
        step=step,
        run_name=str(args.name),
        task_loss_dir=args.task_loss_results_dir,
        task_loss_enabled=bool(args.task_loss_on_save),
        task_loss_eval_script=args.task_loss_eval_script,
        progress_dir=args.progress_dir,
        fingerprint_path=Path(args.save_folder) / "run_fingerprint.json",
        wandb_run=getattr(args, "_wandb_run", None),
        wandb_mode=wandb_mode_from_args(args),
        production=bool(getattr(args, "_production", False)),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=DEFAULT_RUN_ID, help=f"Run id (default: {DEFAULT_RUN_ID})")
    ap.add_argument(
        "--train-paths-file",
        default=None,
        help="Optional local memmap path list from a prior edullm-data stage "
        "(refuses edullm-datasets paths; missing shards fail closed)",
    )
    ap.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"edullm-data dataset id (default: {DEFAULT_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin edullm-data version (default: resolve_latest)",
    )
    ap.add_argument(
        "--stage-dir",
        default=None,
        help="Ephemeral scratch for edullm-data .u32le.bin staging "
        "(required unless --train-paths-file)",
    )
    ap.add_argument("--save-folder", required=True, help="Local/scratch checkpoint root (uploaded to W&B)")
    ap.add_argument("--progress-dir", required=True, help="Local/scratch progress root (uploaded to W&B)")
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
    ap.add_argument(
        "--mask-keep-rate",
        type=float,
        default=MASK_KEEP_RATE,
        help=f"Fraction of valid tokens kept per sequence (default: {MASK_KEEP_RATE})",
    )
    ap.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Explicit local checkpoint dir to resume. "
        "Local auto-resume is off unless ALLOW_LOCAL_RESUME=1",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=0.1)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument(
        "--task-loss-results-dir",
        type=str,
        default=None,
        help="Where to write step{{N}}_task_loss.json (default: ../task_loss_results/control)",
    )
    ap.add_argument(
        "--task-loss-eval-script",
        type=str,
        default=None,
        help="Override path to eval_task_loss_olmo_core.py",
    )
    ap.add_argument(
        "--task-loss-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch full 20-label task_loss eval on each permanent checkpoint (default: on; "
        "also gated by TASK_LOSS_EVAL env)",
    )
    add_wandb_argparse_options(ap, default_run_name=DEFAULT_RUN_ID)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        args.task_loss_results_dir = str(_TS_ROOT / "task_loss_results" / ARM)
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
    production = is_production_run(
        max_tokens=int(args.length_tokens), total_steps=total_steps
    )
    args._production = production
    ladder = permanent_checkpoint_steps(total_steps, int(args.save_interval))
    ladder_set: Set[int] = set(ladder)
    lr = float(PEAK_LR)

    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        Path(args.task_loss_results_dir).mkdir(parents=True, exist_ok=True)

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
            tags=(ARM, "random_60"),
            dir=progress_dir / "wandb",
            id_path=progress_dir / "wandb_run_id.txt",
            is_main=True,
            alert_title=f"token-selection {ARM} started",
        )
        args._wandb_run = wb_run
        require_wandb_for_production(
            wb_run, production=production, mode=wandb_mode_from_args(args)
        )
        eval_poller = WandbEvalPoller(args.task_loss_results_dir, wb_run)
        if wb_run is not None and bool(getattr(args, "wandb_upload_existing", False)):
            wandb_upload_existing(
                wb_run,
                checkpoint_dir=save_folder,
                task_loss_dir=args.task_loss_results_dir,
                progress_dir=progress_dir,
                tokens_per_step=GLOBAL_BATCH_TOKENS,
            )
    else:
        args._wandb_run = None

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "arm": ARM,
        "method": "random",
        "mask_keep_rate": float(args.mask_keep_rate),
        "run_id": args.name,
        "artifact_store": "wandb",
        "matched_reference": "experiments/token-selection/reference/train_olmo3_370m_refhq.py",
        "train_stack": "TokenSelectTrainModule HSDP bf16 SkipStepAdamW compile",
        "loss": (
            f"masked CE on uniform random {args.mask_keep_rate:.0%} of valid tokens "
            "(per-sequence, seeded by step)"
        ),
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
        "ephemeral": False,  # permanent ladder (no prune); scratch itself is ephemeral
        "scratch_ephemeral": True,
        "checkpoint_artifacts": "wandb",
        "train_dataset": f"s3://{DATA_BUCKET}/{args.dataset_id}/",
        "seed": args.seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_results_dir": args.task_loss_results_dir,
    }

    train_paths, data_meta = prepare_train_memmaps(args)
    meta["train_dataset"] = data_meta.get("dataset_uri") or meta["train_dataset"]
    meta["data"] = data_meta
    run_identity = {
        "arm": ARM,
        "run_id": args.name,
        "method": "random",
        "seed": int(args.seed),
        "dataset_id": str(args.dataset_id),
        "dataset_version": str(data_meta.get("dataset_version") or ""),
        "model": "olmo2_370M",
        "sequence_length": SEQ_LEN,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "max_tokens": int(args.length_tokens),
        "total_steps": total_steps,
        "keep_fraction": float(args.mask_keep_rate),
        "lr": lr,
        "lr_warmup_steps": int(args.lr_warmup_steps),
        "lr_alpha_f": float(args.lr_alpha_f),
        "z_loss_multiplier": 1e-5,
        "fused_ce": False,
        "task_loss_definition": "olmo-ladder-20-label-macro-bpb",
    }
    if rank == 0 and args.fresh:
        from token_selection.olmo_ext.permanent_checkpoint import write_run_fingerprint

        write_run_fingerprint(save_folder, run_identity)

    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "checkpoint_ladder.json").write_text(
            json.dumps({"steps": ladder, "interval": int(args.save_interval)}, indent=2) + "\n"
        )
        (progress_dir / "run_plan.txt").write_text(
            "\n".join(
                [
                    f"run_id={args.name}",
                    "architecture=olmo_core.TransformerConfig.olmo2_370M",
                    "method=random (uniform random 60% keep per sequence)",
                    f"mask_keep_rate={args.mask_keep_rate}",
                    "artifact_store=wandb",
                    f"tokenizer={TOKENIZER_ID} vocab={EMBEDDING_SIZE}",
                    f"world_size={world_size}",
                    f"sequence_length={SEQ_LEN}",
                    f"total_steps={total_steps}",
                    f"global_batch_tokens={GLOBAL_BATCH_TOKENS}  mbs_seqs={mbs}  "
                    f"seqs_per_rank={seqs_per_rank}",
                    f"lr={lr}  cos_warmup={args.lr_warmup_steps}  alpha_f={args.lr_alpha_f}",
                    f"compile={args.compile}",
                    f"permanent_ladder={ladder}",
                    f"train={meta['train_dataset']}",
                    f"dtype={data_meta.get('numpy_dtype')} header_bytes={data_meta.get('header_bytes')}",
                    f"n_shards={data_meta.get('n_shards')}",
                    "checkpoint_artifacts=wandb (production online uploads fail closed)",
                    "scratch_ephemeral=True (data staging allowed; artifacts stay local + W&B)",
                    "",
                ]
            )
        )
        log.info(
            "Plan: random-keep control olmo2_370M run_id=%s keep=%.2f world=%d total=%d ladder_n=%d "
            "(omit near-final grid if needed) mbs=%d seqs/rank=%d lr=%.3e data=%s",
            args.name,
            float(args.mask_keep_rate),
            world_size,
            total_steps,
            len(ladder),
            mbs,
            seqs_per_rank,
            lr,
            meta["train_dataset"],
        )
        if total_steps == 2360 and 2250 in ladder_set:
            raise SystemExit("BUG: ladder for 2360 must omit 2250")

    train_ds = MemmapTokenDataset(
        train_paths,
        SEQ_LEN,
        dtype=data_meta.get("numpy_dtype") or "<u4",
        header_bytes=int(data_meta.get("header_bytes") or 0),
    )
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
        total_steps=total_steps,
        seed=int(args.seed),
        mask_keep_rate=float(args.mask_keep_rate),
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
            log.info("--fresh: starting from scratch (ignoring any local checkpoints)")
    else:
        load_dir = resolve_resume_dir(args, save_folder)
        if load_dir is not None:
            from token_selection.olmo_ext.permanent_checkpoint import (
                assert_resume_fingerprint,
                write_run_fingerprint,
            )

            assert_resume_fingerprint(load_dir, run_identity)
            if rank == 0:
                write_run_fingerprint(save_folder, run_identity)
            start_step = load_checkpoint(load_dir, train_module)
            train_module._ensure_state().step = start_step
        elif rank == 0:
            log.info(
                "No --load-path; starting from scratch "
                "(set ALLOW_LOCAL_RESUME=1 only for same-job local recovery)"
            )
            from token_selection.olmo_ext.permanent_checkpoint import (
                write_run_fingerprint,
            )

            write_run_fingerprint(save_folder, run_identity)

    t0 = time.time()
    window_t0 = t0
    window_step0 = start_step
    loss_path = progress_dir / "train_loss.jsonl"

    if is_distributed():
        dist.barrier()

    # Step-0 init snapshot (pre-train) when starting fresh.
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

        input_ids = next_rank_input_ids(train_stream, seqs_per_rank, device)
        batch: Dict[str, torch.Tensor] = {"input_ids": input_ids}

        train_module.zero_grads()
        train_module.train_batch(batch)
        train_module.optim_step()
        train_module.on_optim_step_end()

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
                    extra={k: v for k, v in books.latest_metrics.items() if k.startswith("train/")},
                )
                if eval_poller is not None:
                    eval_poller.poll()

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            ckpt_dir = save_folder / f"step{global_step}"
            save_checkpoint(ckpt_dir, global_step, train_module, args, meta)
            _maybe_task_loss(args, ckpt_dir, global_step)
            if rank == 0 and eval_poller is not None:
                eval_poller.poll()
            if is_distributed():
                dist.barrier()

    if rank == 0:
        if eval_poller is not None:
            eval_poller.poll()
        upload_before_end(
            progress_dir=progress_dir,
            task_loss_dir=args.task_loss_results_dir,
            run=wb_run,
            run_name=str(args.name),
            production=production,
            mode=wandb_mode_from_args(args),
        )
        finish_wandb(wb_run)
        log.info(
            "Training complete at step=%d world_size=%d run_id=%s artifact_store=wandb",
            total_steps,
            world_size,
            args.name,
        )


if __name__ == "__main__":
    main()
