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

Dataset is **only** published RefHQ from ``s3://edullm-data/`` via
``edullm_data.read.resolve_latest`` / ``dataset_paths``
(``pretrain/refhq-regmix-5p5b``). Shards may be staged to an **ephemeral**
local/scratch directory for the job; this script does not assume FarmShare
scratch, laptop-local, or legacy ``s3://edullm-datasets/`` data already present.

``--save-folder`` / ``--progress-dir`` are working dirs on scratch. Checkpoints
and progress upload to W&B artifacts. Production online checkpoint uploads are
fail-closed; ``--local-smoke`` permits a non-production run without W&B.

W&B (SmolLM2-style, project ``token-selection``): soft-enabled when
``WANDB_API_KEY`` is set; skipped otherwise. No task-loss evals on this arm.
Do not confuse with ``olmo3-370m/run-10b-equal``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, cast

import torch
import torch.distributed as dist

_ARM_DIR = Path(__file__).resolve().parent
_TS_ROOT = _ARM_DIR.parent
if str(_TS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TS_ROOT))

# SmolLM2-style W&B: do not hard-disable; soft-skip without API key.
from token_selection.olmo_ext.wandb_logging import (  # noqa: E402
    add_wandb_argparse_options,
    apply_wandb_env_defaults,
    ensure_wandb_not_hard_disabled,
    is_production_run,
    make_wandb_artifacts_callback,
    production_online,
    wandb_callback_kwargs_from_env,
    wandb_enabled,
    wandb_mode_from_args,
)

ensure_wandb_not_hard_disabled()

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
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
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from token_selection.olmo_ext.checkpoint_ladder import (
    checkpointer_kwargs_for_ladder,
    is_permanent_checkpoint_step,
)
log = logging.getLogger("train_olmo2_370m_refhq")

SEQ_LEN = 2048
# Match REL+EMA GPU7 job (run_5b.yaml).
DEFAULT_GLOBAL_BATCH_TOKENS = 4_194_304
DEFAULT_RANK_MICROBATCH_TOKENS = 65_536  # 32 × 2048
DEFAULT_LR = 4.0e-4
DEFAULT_WARMUP_STEPS = 24
DEFAULT_SEED = 6198
# Published train partition rows for pretrain/refhq-regmix-5p5b v2.
DEFAULT_TOKEN_BUDGET = 5_509_020_202
MODEL_SIZE_FOR_LR = 371_262_464
CONFIG_NAME = "OLMo-2-370M-scratch"
ARM = "reference"

# Canonical published dataset (edullm-data); never edullm-datasets.
DEFAULT_DATASET_ID = "pretrain/refhq-regmix-5p5b"
DEFAULT_DATA_BUCKET = "edullm-data"
LEGACY_DATA_BUCKET = "edullm-datasets"


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = DEFAULT_SEED
    load_path: Optional[str] = None


@dataclass(frozen=True)
class ResolvedTrainData:
    dataset_id: str
    version: str
    paths: List[str]
    dtype: str
    rows: int
    source: str  # "edullm-data" | "paths-file"


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


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def resolve_edullm_data_train(
    *,
    dataset_id: str,
    version: Optional[str],
    data_bucket: str = DEFAULT_DATA_BUCKET,
) -> ResolvedTrainData:
    """Resolve validated train shard URIs + dtype from s3://edullm-data."""
    try:
        from edullm_data.read import dataset_paths, resolve_latest
        from edullm_data.s3 import Boto3S3
    except ImportError as exc:
        raise SystemExit(
            "edullm-data package is required to resolve training paths. "
            "Install with: uv add 'edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0' "
            f"(or pip install -e <edullm-data checkout>). Import error: {exc}"
        ) from exc

    s3 = Boto3S3.default()
    ver = version or resolve_latest(dataset_id, s3=s3, data_bucket=data_bucket)
    if not ver:
        raise SystemExit(
            f"No published versions of {dataset_id!r} in s3://{data_bucket}/_catalog/. "
            "Refuse to train without a validated edullm-data dataset."
        )
    resolved = dataset_paths(
        dataset_id,
        ver,
        split="train",
        s3=s3,
        data_bucket=data_bucket,
        require_validated=True,
    )
    if not resolved.paths:
        raise SystemExit(f"{dataset_id}/{ver} train split resolved to zero paths")
    dtype = resolved.dtype or "uint32"
    rows = int(resolved.rows or 0)
    if rows <= 0:
        raise SystemExit(f"{dataset_id}/{ver} train split has no declared row count")
    log.info(
        "Resolved %s/%s train: %d shards, %d tokens, dtype=%s (bucket=%s)",
        dataset_id,
        ver,
        len(resolved.paths),
        rows,
        dtype,
        data_bucket,
    )
    return ResolvedTrainData(
        dataset_id=dataset_id,
        version=ver,
        paths=list(resolved.paths),
        dtype=dtype,
        rows=rows,
        source="edullm-data",
    )


def stage_s3_uris(
    uris: Sequence[str],
    stage_dir: Path,
    *,
    workers: int = 8,
) -> List[str]:
    """Download s3:// shard URIs under stage_dir, preserving key relative path.

    Idempotent: skips objects whose local size already matches HEAD ContentLength.
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise SystemExit(
            f"boto3 is required to stage edullm-data shards locally ({exc})"
        ) from exc

    stage_dir.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")

    planned: List[Tuple[str, str, Path]] = []
    for uri in uris:
        bucket, key = _parse_s3_uri(uri)
        # Keep tokens/... layout under stage_dir.
        rel = key
        marker = f"{DEFAULT_DATASET_ID}/"
        if marker in key:
            # strip "<dataset_id>/<version>/" → tokens/...
            after = key.split(marker, 1)[1]
            # after is "v2/tokens/..."
            parts = after.split("/", 1)
            rel = parts[1] if len(parts) == 2 else after
        dest = stage_dir / rel
        planned.append((bucket, key, dest))

    def _one(item: Tuple[str, str, Path]) -> Path:
        bucket, key, dest = item
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"HEAD s3://{bucket}/{key} failed: {exc}") from exc
        expected = int(head["ContentLength"])
        if dest.is_file() and dest.stat().st_size == expected:
            return dest
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            client.download_file(bucket, key, str(tmp))
            if tmp.stat().st_size != expected:
                raise RuntimeError(
                    f"size mismatch for s3://{bucket}/{key}: "
                    f"got {tmp.stat().st_size}, expected {expected}"
                )
            tmp.replace(dest)
        finally:
            if tmp.exists() and (not dest.exists() or dest.stat().st_size != expected):
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return dest

    local_paths: List[str] = [""] * len(planned)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_one, item): i for i, item in enumerate(planned)}
        for fut in as_completed(futs):
            i = futs[fut]
            local_paths[i] = str(fut.result().resolve())
    return local_paths


def _distcp_ready(step_dir: Path) -> bool:
    return (
        (step_dir / "model_and_optim" / ".metadata").is_file()
        or (step_dir / "model_eval.pt").is_file()
        or (step_dir / "state.pt").is_file()
    )


def _wait_distcp_ready(step_dir: Path, *, timeout_s: float = 600.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if step_dir.is_dir() and _distcp_ready(step_dir):
            return True
        time.sleep(2.0)
    return step_dir.is_dir() and _distcp_ready(step_dir)


def _broadcast_export_ok(ok: bool) -> bool:
    """Share rank-0 export success with every rank (avoids NCCL hang on abort)."""
    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flag = torch.tensor([1 if ok else 0], device=device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())
    return ok


def dtype_to_olmo(dtype_name: str) -> NumpyDatasetDType:
    name = dtype_name.strip().lower()
    # ResolvedSplit may return "uint32"; numpy_dtype may return "<u4".
    aliases = {
        "uint8": NumpyDatasetDType.uint8,
        "u1": NumpyDatasetDType.uint8,
        "|u1": NumpyDatasetDType.uint8,
        "uint16": NumpyDatasetDType.uint16,
        "u2": NumpyDatasetDType.uint16,
        "<u2": NumpyDatasetDType.uint16,
        ">u2": NumpyDatasetDType.uint16,
        "uint32": NumpyDatasetDType.uint32,
        "u4": NumpyDatasetDType.uint32,
        "<u4": NumpyDatasetDType.uint32,
        ">u4": NumpyDatasetDType.uint32,
        "uint64": NumpyDatasetDType.uint64,
        "u8": NumpyDatasetDType.uint64,
        "<u8": NumpyDatasetDType.uint64,
        ">u8": NumpyDatasetDType.uint64,
    }
    if name not in aliases:
        raise SystemExit(f"Unsupported token dtype from edullm-data: {dtype_name!r}")
    return aliases[name]


def resolve_train_data(opts: argparse.Namespace) -> ResolvedTrainData:
    """Resolve train paths: optional paths-file override, else edullm-data (+ optional stage)."""
    if opts.paths_file:
        paths = read_paths(Path(opts.paths_file))
        # Escape hatch for shards staged by prepare_refhq_data / a prior --stage-dir
        # run of this script on the same ephemeral machine — not durable scratch.
        for p in paths:
            if LEGACY_DATA_BUCKET in p.replace("\\", "/"):
                raise SystemExit(
                    f"Refusing paths-file entry under legacy {LEGACY_DATA_BUCKET}: {p}\n"
                    f"Resolve from s3://{DEFAULT_DATA_BUCKET}/ via --dataset-id "
                    f"{DEFAULT_DATASET_ID} (omit --paths-file) instead."
                )
        rows = opts.token_budget if opts.token_budget is not None else DEFAULT_TOKEN_BUDGET
        return ResolvedTrainData(
            dataset_id=opts.dataset_id,
            version=opts.dataset_version or "paths-file",
            paths=paths,
            dtype=opts.dtype or "uint32",
            rows=int(rows),
            source="paths-file",
        )

    data = resolve_edullm_data_train(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        data_bucket=opts.data_bucket,
    )

    if opts.stage_dir and not getattr(opts, "dry_run", False):
        stage_dir = Path(opts.stage_dir)
        if get_rank() == 0:
            log.info("Staging %d shards to %s", len(data.paths), stage_dir)
            local = stage_s3_uris(data.paths, stage_dir, workers=opts.stage_workers)
            paths_file = stage_dir / "paths_train.txt"
            paths_file.write_text("\n".join(local) + "\n")
            meta = {
                "dataset_id": data.dataset_id,
                "version": data.version,
                "data_bucket": opts.data_bucket,
                "dtype": data.dtype,
                "rows": data.rows,
                "n_shards": len(local),
                "stage_dir": str(stage_dir.resolve()),
                "paths_file": str(paths_file.resolve()),
            }
            (stage_dir / "stage_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            log.info("Staged paths written to %s", paths_file)
        _barrier()
        paths_file = Path(opts.stage_dir) / "paths_train.txt"
        if not paths_file.is_file():
            raise SystemExit(f"Rank {get_rank()}: missing staged paths file {paths_file}")
        local_paths = read_paths(paths_file)
        return ResolvedTrainData(
            dataset_id=data.dataset_id,
            version=data.version,
            paths=local_paths,
            dtype=data.dtype,
            rows=data.rows,
            source="edullm-data",
        )
    if opts.stage_dir and getattr(opts, "dry_run", False):
        log.info("dry-run: skipping stage to %s; using s3:// URIs", opts.stage_dir)

    # Train directly from s3://edullm-data URIs (olmo_core remote IO).
    return data


def build_config(opts: argparse.Namespace, train_data: ResolvedTrainData) -> ExperimentConfig:
    tokenizer = TokenizerConfig.dolma2()
    model_config = build_olmo2_370m()

    paths = train_data.paths
    dtype = dtype_to_olmo(train_data.dtype)
    dataset_config = NumpyFSLDatasetConfig(
        paths=paths,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer,
        work_dir=opts.work_dir,
        dtype=dtype,
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
        num_workers=opts.num_workers,
    )

    lr = opts.lr if opts.lr is not None else DEFAULT_LR
    token_budget = (
        opts.token_budget if opts.token_budget is not None else train_data.rows
    )

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
    total_steps = token_budget // tokens_per_step
    save_interval = opts.save_interval
    # Shared permanent ladder (step 0 + interval grid + true final; no ephemeral prune).
    ckpt_kwargs = checkpointer_kwargs_for_ladder(
        total_steps, save_interval, save_async=False
    )
    permanent_save_steps = list(ckpt_kwargs["fixed_steps"])

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(token_budget),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(**ckpt_kwargs),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    # W&B: train scalars via olmo_core WandBCallback; checkpoint artifacts via side channel.
    # Production online checkpoint uploads are fail-closed.
    apply_wandb_env_defaults(
        project=getattr(opts, "wandb_project", None) or "token-selection",
        run_name=getattr(opts, "wandb_run_name", None) or opts.name,
        group=getattr(opts, "wandb_group", None) or ARM,
    )
    os.environ["WANDB_MODE"] = wandb_mode_from_args(opts)
    ensure_wandb_not_hard_disabled()
    wb_enabled = wandb_enabled(mode=wandb_mode_from_args(opts), is_main=True)
    production = not bool(getattr(opts, "local_smoke", False))
    if production_online(
        production=production, mode=wandb_mode_from_args(opts)
    ) and not wb_enabled:
        raise SystemExit(
            "production online reference runs require W&B checkpoint artifacts"
        )
    try:
        from olmo_core.train.callbacks import WandBCallback  # type: ignore

        wb_kwargs = wandb_callback_kwargs_from_env(
            run_name=opts.name,
            arm=ARM,
            method="refhq",
            config={
                "arm": ARM,
                "method": "refhq",
                "run_name": opts.name,
                "token_budget": token_budget,
                "total_steps": total_steps,
                "dataset_id": train_data.dataset_id,
                "dataset_version": train_data.version,
            },
            enabled=wb_enabled,
        )
        if getattr(opts, "wandb_project", None):
            wb_kwargs["project"] = str(opts.wandb_project)
        if getattr(opts, "wandb_entity", None):
            wb_kwargs["entity"] = str(opts.wandb_entity)
        trainer_config = trainer_config.with_callback("wandb", WandBCallback(**wb_kwargs))
        trainer_config = trainer_config.with_callback(
            "wandb_artifacts",
            make_wandb_artifacts_callback(
                results_dir=opts.progress_dir,
                save_folder=opts.save_folder,
                total_steps=total_steps,
                interval=save_interval,
                tokens_per_step=tokens_per_step,
                progress_dir=opts.progress_dir,
                production=production,
                wandb_mode=wandb_mode_from_args(opts),
                run_name=opts.name,
            ),
        )
    except ImportError:
        if production_online(production=production, mode=wandb_mode_from_args(opts)):
            raise
        if get_rank() == 0:
            log.warning("olmo_core.WandBCallback unavailable; local smoke has no W&B")

    if get_rank() == 0:
        progress = Path(opts.progress_dir)
        progress.mkdir(parents=True, exist_ok=True)
        dataset_uri = (
            f"s3://{opts.data_bucket}/{train_data.dataset_id}/{train_data.version}/"
            if train_data.source == "edullm-data"
            else f"paths-file:{opts.paths_file}"
        )
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
            "token_budget": token_budget,
            "total_steps": total_steps,
            "save_interval": save_interval,
            "arm": ARM,
            "permanent_save_steps": permanent_save_steps,
            "final_checkpoint": "post_train",
            "max_checkpoints": None,
            "ephemeral_scratch": True,
            "artifact_store": "wandb",
            "compile_model": opts.compile_model,
            "attn_backend": str(resolve_attn_backend()),
            "seed": opts.seed,
            "dataset_id": train_data.dataset_id,
            "dataset_version": train_data.version,
            "dataset": dataset_uri,
            "data_bucket": opts.data_bucket,
            "data_source": train_data.source,
            "dtype": train_data.dtype,
            "reference_job": "rel-ema-5b-scratch-v1 (arch/batch/seq/lr; RefHQ data)",
            "paths": len(paths),
            "world_size": get_world_size(),
            "evals": False,
            "model_size_non_embedding": MODEL_SIZE_FOR_LR,
        }
        (progress / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress / "total_steps.txt").write_text(str(total_steps) + "\n")
        log.info(
            "RefHQ reference: olmo2_370M (%s) CosWithWarmup warmup=%d alpha_f=%s lr=%.6g "
            "steps=%d mbs_seqs=%d seq=%d compile=%s dataset=%s/%s",
            CONFIG_NAME,
            opts.warmup_steps,
            opts.alpha_f,
            lr,
            total_steps,
            opts.rank_microbatch_size // opts.sequence_length,
            opts.sequence_length,
            opts.compile_model,
            train_data.dataset_id,
            train_data.version,
        )

    # Stash resolved budget so dry-run / callers see it.
    opts.token_budget = token_budget
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
        train_data = resolve_train_data(opts)
        if opts.wandb_resume_artifact:
            if get_rank() == 0:
                from token_selection.olmo_ext.wandb_logging import (
                    restore_checkpoint_artifact,
                )

                restored = restore_checkpoint_artifact(
                    opts.wandb_resume_artifact,
                    opts.save_folder,
                    require_fingerprint=False,
                )
                (Path(opts.progress_dir) / "wandb_resume_path.txt").parent.mkdir(
                    parents=True, exist_ok=True
                )
                (Path(opts.progress_dir) / "wandb_resume_path.txt").write_text(
                    str(restored), encoding="utf-8"
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            opts.load_path = (
                Path(opts.progress_dir) / "wandb_resume_path.txt"
            ).read_text(encoding="utf-8").strip()
        cfg = build_config(opts, train_data)
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
    ap.add_argument(
        "--paths-file",
        default=None,
        help=(
            "Optional override: local path list from prepare_refhq_data / a prior "
            "--stage-dir on this ephemeral machine. Default: resolve from s3://edullm-data"
        ),
    )
    ap.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"edullm-data dataset id (default: {DEFAULT_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin a version (e.g. v2). Default: resolve_latest()",
    )
    ap.add_argument(
        "--data-bucket",
        default=DEFAULT_DATA_BUCKET,
        help="Published data bucket (default: edullm-data)",
    )
    ap.add_argument(
        "--stage-dir",
        default=None,
        help=(
            "Ephemeral scratch dir: download resolved s3://edullm-data train shards here "
            "before training. If omitted, train from s3:// URIs directly."
        ),
    )
    ap.add_argument("--stage-workers", type=int, default=8)
    ap.add_argument(
        "--dtype",
        default=None,
        help="Override memmap dtype (default: from edullm_data ResolvedSplit, else uint32)",
    )
    ap.add_argument(
        "--save-folder",
        required=True,
        help="Runtime-scratch DistCP working dir; checkpoints upload to W&B",
    )
    ap.add_argument(
        "--progress-dir",
        required=True,
        help="Runtime-scratch metrics/run_meta dir (uploaded to W&B)",
    )
    ap.add_argument("--work-dir", default=None, help="olmo_core dataset work dir (default: progress-dir)")
    ap.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help=f"Tokens to train (default: published train rows, currently {DEFAULT_TOKEN_BUDGET})",
    )
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
    ap.add_argument(
        "--load-path",
        type=str,
        default=None,
        help=(
            "Local DistCP dir to warm-start when save-folder is empty. "
            "Use --wandb-resume-artifact for cross-job restore."
        ),
    )
    ap.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile (default: on)",
    )
    ap.add_argument(
        "--local-smoke",
        action="store_true",
        help="Allow a non-production local run without required online W&B artifacts",
    )
    ap.add_argument("--dry-run", action="store_true")
    add_wandb_argparse_options(ap, default_run_name=None)
    opts = ap.parse_args(argv)
    opts.work_dir = opts.work_dir or opts.progress_dir
    if not getattr(opts, "wandb_run_name", None):
        opts.wandb_run_name = opts.name
    if opts.data_bucket == LEGACY_DATA_BUCKET:
        ap.error(f"legacy bucket {LEGACY_DATA_BUCKET} is not allowed; use edullm-data")
    if opts.load_path:
        lp = opts.load_path.replace("\\", "/")
        if LEGACY_DATA_BUCKET in lp:
            ap.error(f"--load-path must not reference {LEGACY_DATA_BUCKET}")
        if lp.startswith("s3://"):
            ap.error(
                "--load-path must be a local DistCP directory on this machine; "
                "use --wandb-resume-artifact for cross-job restore"
            )
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
            train_data = resolve_train_data(args)
            cfg = build_config(args, train_data)
            if get_rank() == 0:
                print(cfg)
                print("dataset_id", train_data.dataset_id, train_data.version)
                print("dtype", train_data.dtype)
                print("paths", len(train_data.paths))
                print("lr", args.lr)
                print("token_budget", args.token_budget)
                print("steps", args.token_budget // args.global_batch_size)
        finally:
            teardown_training_environment()
        sys.exit(0)
    main(args)
