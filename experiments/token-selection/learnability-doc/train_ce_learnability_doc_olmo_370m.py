#!/usr/bin/env python3
"""Plain CE on learnability-doc filtered RegMix (top 60% tokens by early−late).

RefHQ-matched ``olmo2_370M`` stack (same as control / RefHQ):

  * ``TransformerConfig.olmo2_370M`` (full attn, no SWA)
  * ``TransformerTrainModule`` — HSDP bf16, SkipStepAdamW, CosWithWarmup,
    ``compile_model=True``, ``z_loss_multiplier=1e-5``
  * fused LM-head CE when liger-kernel is available (no full logits)
  * sequence=2048, global_batch=4_194_304, rank_microbatch=65_536
  * train corpus: published ``pretrain/learnability-doc-top60`` from
    ``s3://edullm-data/`` (upsample to 9.9B / 2360 steps); fail-closed if missing
  * permanent checkpoints ``{0,125,…,2125,2360}`` (omit 2250); no ephemeral prune
  * immediate 20-label ``task_loss_bpb`` via shared ``task_loss_hook`` on each permanent save
  * single- or multi-GPU via ``torchrun`` + HSDP (world size from env)
  * durable checkpoint/results export to ``s3://edullm-checkpoints/token-sel/learnability-doc/``
    (upload-before-continue; ``S3_EXPORT=0`` only for intentional local smoke)

Ephemeral empty-scratch: stage shards from ``edullm-data`` into ``--stage-dir`` for
the job; do not assume pre-existing scratch corpora or local checkpoints.
Resume only via explicit ``--load-path`` (stage that tree from durable storage first).
Refuses legacy ``s3://edullm-datasets/`` paths.

W&B: project ``token-selection``, group ``learnability-doc`` (SmolLM2-style soft-skip without API key).
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

_ARM_DIR = Path(__file__).resolve().parent
_TS_ROOT = _ARM_DIR.parent
if str(_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TS_ROOT))

# SmolLM2-style W&B: do not hard-disable; soft-skip without API key.
from token_selection.olmo_ext.wandb_logging import (  # noqa: E402
    add_wandb_argparse_options,
    apply_wandb_env_defaults,
    ensure_wandb_not_hard_disabled,
    finish_wandb,
    init_wandb_from_args,
    namespace_path_config,
    wandb_log_checkpoint,
    wandb_log_train,
    wandb_upload_existing,
    WandbEvalPoller,
)

ensure_wandb_not_hard_disabled()

from token_selection.olmo_ext.checkpoint_ladder import (  # noqa: E402
    DEFAULT_CHECKPOINT_INTERVAL,
    permanent_checkpoint_steps,
)
from token_selection.olmo_ext.task_loss_hook import trigger_task_loss_eval  # noqa: E402

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

log = logging.getLogger("train_ce_learnability_doc_olmo2_370m")

SEQ_LEN = 2048
EMBEDDING_SIZE = 100_352
GLOBAL_BATCH_TOKENS = 4_194_304
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4
DEFAULT_SEED = 6198
DEFAULT_RUN_ID = "edullm-370M-learnability-doc-10b"
DEFAULT_LENGTH_TOKENS = 9_900_000_000  # → 2360 steps at GBS 4_194_304 (one-epoch matrix)
CONFIG_NAME = "OLMo-2-370M-scratch"
ARM = "learnability-doc"
# Intended published filtered corpus (top 60% tokens by early−late learnability).
# Not a substitute for pretrain/regmix-10b (control / unfiltered base).
DEFAULT_TRAIN_DATASET_ID = "pretrain/learnability-doc-top60"
DATA_BUCKET = "edullm-data"


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
# Data — edullm-data resolve + stage, then memmap
# ---------------------------------------------------------------------------


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
        self._header_bytes = int(header_bytes)
        if self._header_bytes < 0:
            raise SystemExit(f"header_bytes must be >= 0, got {self._header_bytes}")
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=dtype, offset=self._header_bytes)
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


def _stage_uri(uri: str, dest: Path) -> None:
    """Download one object to ``dest`` if missing or size-mismatched (boto3 streaming)."""
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _parse_s3_uri(uri)
    client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        raise SystemExit(f"Cannot HEAD {uri}: {exc}") from exc
    size = int(head["ContentLength"])
    if dest.is_file() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    log.info("Staging %s → %s (%d bytes)", uri, dest, size)
    client.download_file(bucket, key, str(tmp))
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Incomplete download for {uri}: got {got}, want {size}")
    tmp.replace(dest)


def resolve_and_stage_train_paths(
    *,
    dataset_id: str,
    version: Optional[str],
    stage_dir: Path,
    split: str = "train",
) -> Tuple[List[str], str, str, Any, int, Optional[int]]:
    """Resolve validated edullm-data shards and stage them under ``stage_dir``.

    Returns ``(local_paths, version, dtype_name, numpy_dtype, header_bytes, rows)``.
    """
    try:
        from edullm_data.read import NotValidated, ReadError, dataset_paths, resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:
        raise SystemExit(
            "edullm-data package is required to resolve training shards. "
            'Install with: uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"'
        ) from exc

    s3 = Boto3S3.default()
    ver = version
    if not ver:
        ver = resolve_latest(dataset_id, s3=s3, data_bucket=DATA_BUCKET)
    if not ver:
        raise SystemExit(
            f"Dataset {dataset_id!r} is not published in s3://{DATA_BUCKET}/_catalog/. "
            "Publish the offline top-60% learnability-doc filtered token corpus "
            "(filter_learnability_docs.py → build_filtered_corpus.py → edullm_data.publish) "
            f"before training. Related published base (not a substitute): pretrain/regmix-10b. "
            f"Do not use s3://edullm-datasets/ or pre-staged FarmShare/laptop paths."
        )
    try:
        resolved = dataset_paths(
            dataset_id,
            ver,
            split=split,
            s3=s3,
            data_bucket=DATA_BUCKET,
            require_validated=True,
        )
    except NotValidated as exc:
        raise SystemExit(
            f"{dataset_id}/{ver} has no _VALIDATED.json — refusing unvalidated data: {exc}"
        ) from exc
    except ReadError as exc:
        raise SystemExit(f"Cannot resolve {dataset_id}/{ver} split={split}: {exc}") from exc

    uris = list(resolved.paths)
    if not uris:
        raise SystemExit(
            f"{dataset_id}/{ver} split={split!r} resolved to zero shards under "
            f"s3://{DATA_BUCKET}/"
        )
    dtype_name = resolved.dtype or "uint32"
    np_dtype = np.dtype(resolved.numpy_dtype or dtype_name)
    header_bytes = int(getattr(resolved, "header_bytes", 0) or 0)

    root = stage_dir / dataset_id.replace("/", "__") / ver / split
    root.mkdir(parents=True, exist_ok=True)
    local_paths: List[str] = []
    for uri in uris:
        _, key = _parse_s3_uri(uri)
        dest = root / Path(key).name
        _stage_uri(uri, dest)
        local_paths.append(str(dest.resolve()))

    paths_file = root / "paths_train.txt"
    paths_file.write_text("\n".join(local_paths) + "\n", encoding="utf-8")
    meta = {
        "dataset_id": dataset_id,
        "version": ver,
        "split": split,
        "bucket": DATA_BUCKET,
        "dtype": dtype_name,
        "numpy_dtype": str(np_dtype),
        "header_bytes": header_bytes,
        "rows": resolved.rows,
        "n_shards": len(local_paths),
        "s3_uris": uris,
        "local_paths": local_paths,
        "paths_file": str(paths_file.resolve()),
    }
    (root / "stage_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info(
        "Staged %s/%s split=%s → %d shards dtype=%s rows=%s under %s",
        dataset_id,
        ver,
        split,
        len(local_paths),
        dtype_name,
        resolved.rows,
        root,
    )
    return local_paths, ver, dtype_name, np_dtype, header_bytes, resolved.rows


def resolve_train_corpus(args: argparse.Namespace) -> Tuple[List[str], Any, int, Dict[str, Any]]:
    """Return ``(local_paths, numpy_dtype, header_bytes, provenance)`` for MemmapTokenDataset."""
    if "edullm-datasets" in str(args.dataset_id).replace("\\", "/"):
        raise SystemExit(
            f"Refusing legacy edullm-datasets dataset id: {args.dataset_id!r}. "
            f"Use published edullm-data id {DEFAULT_TRAIN_DATASET_ID!r}."
        )
    if args.train_paths_file:
        # Explicit reuse of a prior stage from *this* script (paths_train.txt under stage-dir).
        paths = read_paths(Path(args.train_paths_file))
        for p in paths:
            if "edullm-datasets" in p.replace("\\", "/"):
                raise SystemExit(
                    f"Refusing legacy edullm-datasets path in --train-paths-file: {p}"
                )
            if not Path(p).is_file():
                raise SystemExit(
                    f"Missing shard listed in --train-paths-file: {p}. "
                    "Omit --train-paths-file and pass --stage-dir to fetch from edullm-data."
                )
        dtype = np.dtype(args.token_dtype)
        prov = {
            "source": "train_paths_file",
            "train_paths_file": str(Path(args.train_paths_file).resolve()),
            "dataset_id": args.dataset_id,
            "note": "Caller-supplied paths; prefer --stage-dir + dataset resolve on a clean machine",
        }
        return paths, dtype, int(args.header_bytes), prov

    if not args.stage_dir:
        raise SystemExit(
            "Pass --stage-dir (scratch location for edullm-data fetch) or "
            "--train-paths-file from a prior stage of this script."
        )

    # Rank 0 stages; all ranks read the handoff after barrier.
    stage_dir = Path(args.stage_dir)
    handoff = stage_dir / "_learnability_doc_stage_handoff.json"

    if get_rank() == 0:
        stage_dir.mkdir(parents=True, exist_ok=True)
        local_paths, ver, dtype_name, np_dtype, header_bytes, rows = resolve_and_stage_train_paths(
            dataset_id=args.dataset_id,
            version=args.dataset_version,
            stage_dir=stage_dir,
            split="train",
        )
        handoff.write_text(
            json.dumps(
                {
                    "paths": local_paths,
                    "version": ver,
                    "dtype": dtype_name,
                    "numpy_dtype": str(np_dtype),
                    "header_bytes": header_bytes,
                    "rows": rows,
                    "dataset_id": args.dataset_id,
                    "bucket": DATA_BUCKET,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if is_distributed():
        dist.barrier()

    if not handoff.is_file():
        raise SystemExit(f"Stage handoff missing after rank-0 stage: {handoff}")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    paths = list(payload["paths"])
    np_dtype = np.dtype(payload.get("numpy_dtype") or payload.get("dtype") or "uint32")
    header_bytes = int(payload.get("header_bytes") or 0)
    prov = {
        "source": "edullm-data",
        "bucket": DATA_BUCKET,
        "dataset_id": payload["dataset_id"],
        "version": payload["version"],
        "dtype": payload.get("dtype"),
        "numpy_dtype": str(np_dtype),
        "header_bytes": header_bytes,
        "rows": payload.get("rows"),
        "stage_dir": str(stage_dir.resolve()),
        "n_shards": len(paths),
    }
    return paths, np_dtype, header_bytes, prov


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
# Model / train module (RefHQ-matched olmo2_370M)
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
# Checkpointing — permanent ladder only (no ephemeral prune)
# ---------------------------------------------------------------------------


def _cpu_plain_tensor(t: Any) -> torch.Tensor:
    """Detach a (possibly DTensor) value to a plain CPU tensor."""
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
    """All ranks must call this. Returns a full (unsharded) CPU state dict.

    ``state_dict_to_save()`` keeps HSDP local shards; rank-0-only saves then
    drop half the weights at world_size=2. Use full_state_dict gather instead.
    """
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
    """All ranks must call (HSDP gather); rank 0 writes + durable-exports.

    When S3 export is enabled, export failure aborts every rank (no hang on barrier).
    """
    train_module_sd = gather_train_module_state_dict(train_module)
    export_ok = True
    if get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "step": step,
            "train_module": train_module_sd,
            "args": vars(args),
            "meta": meta,
            "architecture": "olmo_core.TransformerConfig.olmo2_370M",
            "config_name": CONFIG_NAME,
            "arm": ARM,
            "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (RefHQ-matched)",
            "method": "learnability_doc_ce",
            "ephemeral": False,
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
        from token_selection.olmo_ext.s3_export import (
            export_arm_checkpoint,
            export_arm_task_loss_dir,
            s3_export_enabled,
        )

        if not s3_export_enabled():
            log.warning(
                "S3 export disabled (S3_EXPORT=0 / SKIP_S3_UPLOAD=1); checkpoint is local-only "
                "and will be lost when ephemeral scratch is wiped"
            )
        else:
            export_ok = bool(export_arm_checkpoint(ARM, path))
            tl = getattr(args, "task_loss_results_dir", None)
            if export_ok and tl:
                export_ok = bool(export_arm_task_loss_dir(ARM, tl))
        wb_run = getattr(args, "_wandb_run", None)
        if wb_run is not None:
            tokens_seen = int(step) * int(GLOBAL_BATCH_TOKENS)
            wandb_log_checkpoint(wb_run, path, step=int(step), tokens_seen=tokens_seen)

    if is_distributed():
        flag = torch.tensor([1 if export_ok else 0], device=torch.device("cuda"))
        dist.broadcast(flag, src=0)
        export_ok = bool(flag.item())
    if not export_ok:
        raise SystemExit(
            f"Durable S3 export failed for checkpoint {path}. "
            "Fix AWS credentials / aws CLI on the train host, then retry. "
            "Set S3_EXPORT=0 only for intentional non-durable local smoke runs."
        )


def _checkpoint_step(path: Path) -> int:
    return int(path.name.replace("step", "").split("-")[0])


def find_latest_checkpoint(save_folder: Path) -> Optional[Path]:
    """Newest permanent ``step*`` under save_folder (leftover-scratch detection)."""
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
    """Fire shared 20-label task_loss_bpb hook after a permanent save (rank 0)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=DEFAULT_RUN_ID, help=f"Run id (default: {DEFAULT_RUN_ID})")
    ap.add_argument(
        "--dataset-id",
        default=DEFAULT_TRAIN_DATASET_ID,
        help=(
            f"Published edullm-data dataset id for the filtered learnability-doc corpus "
            f"(default: {DEFAULT_TRAIN_DATASET_ID})"
        ),
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin a version (e.g. v1); default resolve_latest via edullm-data catalog",
    )
    ap.add_argument(
        "--stage-dir",
        type=str,
        default=None,
        help="Scratch dir for clean-machine fetch from s3://edullm-data/ (required unless "
        "--train-paths-file reuses a prior stage from this script)",
    )
    ap.add_argument(
        "--train-paths-file",
        default=None,
        help="Optional reuse of paths_train.txt written by a prior --stage-dir fetch "
        "(must list local shards; refuses edullm-datasets URIs)",
    )
    ap.add_argument(
        "--token-dtype",
        default="uint32",
        help="Only used with --train-paths-file when stage_meta is unavailable (default uint32)",
    )
    ap.add_argument(
        "--header-bytes",
        type=int,
        default=0,
        help="Only used with --train-paths-file (edullm-data path uses resolved header_bytes)",
    )
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument(
        "--length-tokens",
        type=int,
        default=DEFAULT_LENGTH_TOKENS,
        help="Train token budget (upsample via dataloader cycling); default 9.9B / 2360 steps",
    )
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
    ap.add_argument(
        "--checkpoint-interval",
        dest="save_interval",
        type=int,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Explicit checkpoint dir to resume (stage from s3://edullm-checkpoints/ first on "
        "ephemeral scratch). Local save_folder is never auto-resumed.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Start from step 0 (default when --load-path is omitted)",
    )
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=0.1)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument(
        "--task-loss-results-dir",
        type=str,
        default=None,
        help="Where to write step{{N}}_task_loss.json (default: ../task_loss_results/learnability-doc)",
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
            tags=(ARM, "learnability_doc_ce"),
            dir=progress_dir / "wandb",
            id_path=progress_dir / "wandb_run_id.txt",
            is_main=True,
            alert_title=f"token-selection {ARM} started",
        )
        args._wandb_run = wb_run
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
        "method": "learnability_doc_ce",
        "run_id": args.name,
        "matched_reference": "experiments/token-selection/reference/train_olmo3_370m_refhq.py",
        "filter_polarity": (
            "rank by improvement_early_minus_late = "
            "-learnability_late_minus_early_avg_nll; keep top 60% tokens"
        ),
        "train_stack": "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
        "loss": "LM-head CE (fused_linear when liger available); no online token masking",
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
        "compile": bool(args.compile),
        "attn_backend": str(resolve_attn_backend()),
        "save_interval": int(args.save_interval),
        "permanent_checkpoint_steps": ladder,
        "max_checkpoints": None,
        "ephemeral": False,
        "train_dataset_id": args.dataset_id,
        "train_dataset_version": args.dataset_version,
        "train_data_bucket": DATA_BUCKET,
        "stage_dir": args.stage_dir,
        "s3_export_prefix": f"token-sel/{ARM}",
        "seed": args.seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_results_dir": args.task_loss_results_dir,
    }

    train_paths, token_dtype, header_bytes, data_prov = resolve_train_corpus(args)
    meta["train_data"] = data_prov
    meta["train_dataset"] = (
        f"s3://{DATA_BUCKET}/{data_prov.get('dataset_id', args.dataset_id)}/"
        f"{data_prov.get('version', args.dataset_version or 'latest')}/"
    )

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
                    "method=learnability_doc_ce",
                    f"arm={ARM}",
                    "train_stack=TransformerTrainModule/HSDP/SkipStepAdamW (RefHQ-matched)",
                    f"world_size={world_size}",
                    f"sequence_length={SEQ_LEN}",
                    f"total_steps={total_steps}",
                    f"global_batch_tokens={GLOBAL_BATCH_TOKENS}  mbs_seqs={mbs}  "
                    f"seqs_per_rank={seqs_per_rank}",
                    f"lr={lr}  cos_warmup={args.lr_warmup_steps}  alpha_f={args.lr_alpha_f}",
                    f"compile={args.compile}",
                    f"permanent_ladder={ladder}",
                    f"train={meta['train_dataset']}",
                    f"data_source={data_prov.get('source')}",
                    f"s3_export=s3://edullm-checkpoints/token-sel/{ARM}/",
                    "",
                ]
            )
        )
        log.info(
            "Plan: learnability-doc CE olmo2_370M run_id=%s world=%d total=%d ladder_n=%d "
            "mbs=%d seqs/rank=%d lr=%.3e data=%s",
            args.name,
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
        train_paths, SEQ_LEN, dtype=token_dtype, header_bytes=header_bytes
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
    )
    books = _Bookkeeping(
        global_step=0,
        max_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        device=device,
    )
    train_module._attach_trainer(books)  # type: ignore[arg-type]

    start_step = 0
    if args.load_path:
        if args.fresh and rank == 0:
            log.warning("--fresh ignored because --load-path=%s was set", args.load_path)
        start_step = load_checkpoint(Path(args.load_path), train_module)
    elif args.fresh:
        if rank == 0:
            log.info("--fresh: starting from scratch (ephemeral; no local auto-resume)")
    else:
        # Ephemeral contract: never silently overwrite leftover SAVE_FOLDER trees.
        leftover = find_latest_checkpoint(save_folder)
        if leftover is not None:
            raise SystemExit(
                f"Found local checkpoint {leftover} under --save-folder, but ephemeral "
                "runs do not auto-resume or clobber scratch trees (scratch may be wiped). "
                f"Pass --load-path {leftover} after confirming it was staged from durable S3, "
                "or --fresh to start over."
            )
        if rank == 0:
            log.info(
                "Starting from step 0 (ephemeral: no auto-resume from --save-folder; "
                "pass --load-path after staging a durable checkpoint if resuming)"
            )

    def _maybe_save_and_eval(step: int) -> None:
        if step not in ladder_set:
            return
        if is_distributed():
            dist.barrier()
        ckpt_dir = save_folder / f"step{step}"
        save_checkpoint(ckpt_dir, step, train_module, args, meta)
        if args.task_loss_on_save:
            _maybe_task_loss(args, ckpt_dir, step)
        if rank == 0 and eval_poller is not None:
            eval_poller.poll()
        if is_distributed():
            dist.barrier()

    t0 = time.time()
    window_t0 = t0
    window_step0 = start_step
    loss_path = progress_dir / "train_loss.jsonl"

    if is_distributed():
        dist.barrier()

    # Step-0 init snapshot (skip if resuming past 0).
    if start_step == 0:
        _maybe_save_and_eval(0)

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

        _maybe_save_and_eval(global_step)

    if rank == 0:
        if eval_poller is not None:
            eval_poller.poll()
        log.info("Training complete at step=%d world_size=%d", total_steps, world_size)
        from token_selection.olmo_ext.s3_export import (
            export_arm_task_loss_dir,
            s3_export_enabled,
            sync_to_s3,
        )
        from token_selection.olmo_ext.s3_layout import arm_uri

        if not s3_export_enabled():
            log.warning(
                "S3_EXPORT disabled at end of run; local save_folder/progress are not durable"
            )
        else:
            ckpt_ok = sync_to_s3(save_folder, arm_uri(ARM, "checkpoints"))
            prog_ok = sync_to_s3(progress_dir, arm_uri(ARM, "progress"))
            tl_ok = True
            if args.task_loss_results_dir:
                tl_ok = export_arm_task_loss_dir(ARM, args.task_loss_results_dir)
            if not (ckpt_ok and prog_ok and tl_ok):
                raise SystemExit(
                    "Final durable S3 export failed (checkpoints/progress/task_loss). "
                    "Local artifacts may be wiped with ephemeral scratch. "
                    "Fix AWS credentials / aws CLI, or set S3_EXPORT=0 for local-only smoke."
                )
            log.info("Final durable export → %s", arm_uri(ARM))
        finish_wandb(wb_run)


if __name__ == "__main__":
    main()
