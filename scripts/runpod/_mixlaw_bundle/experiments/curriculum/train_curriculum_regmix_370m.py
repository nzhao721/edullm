#!/usr/bin/env python3
"""Curriculum / control CE pretraining on RegMix 10B (OLMo2-370M).

Fork of ``experiments/token-selection/control/train_ce_regmix_olmo_370m.py`` with:

  * Warmup + **constant** LR (``--lr-alpha-f 1.0`` default)
  * ``--pacing`` / ``--difficulty-metric`` selecting the data stream
  * Control arm: random shuffle over flat token memmaps from ``pretrain/regmix-10b``
  * Curriculum arms: ``CurriculumChunkStream`` over a published ``curriculum/*``
    token-order (indices into the same parent pool)

**Ephemeral runtime.** Assumes job-scoped scratch starts empty and is wiped after
the job. Stages train/curriculum bytes from validated ``s3://edullm-data/`` into
``--data-cache-dir`` (or ``$EDULLM_DATA_CACHE``). Does not assume FarmShare
scratch, laptop corpora, old run dirs, or leftover checkpoints already exist.

**Artifact durability.** Permanent ladder checkpoints, progress, metrics, and
task-loss outputs stay on job-local scratch and are uploaded to W&B project
``curriculum``. Production requires ``--wandb-mode online`` and a W&B API key.
Every permanent checkpoint upload is awaited before training continues; upload
failure is **fail-closed** across all ranks. Only explicit local smoke runs may
use ``--wandb-mode disabled --allow-local-only``.

W&B mirrors the SmolLM2 FarmShare protocol: train loss / LR / throughput at
``--log-interval``, task-loss eval metrics + artifacts, checkpoint artifacts,
and runtime progress/config/metrics snapshots. Local W&B dirs stay under
job-scoped scratch (``--progress-dir`` sibling ``wandb/``).

Resume via ``--load-path`` from a local directory or a
``wandb-artifact://entity/project/name:version`` reference downloaded into this
job's scratch. S3 checkpoint paths are rejected. Leftover local checkpoints
without ``--load-path`` / ``--fresh`` fail closed.

Training data is resolved only via ``edullm_data.read.dataset_paths`` /
``resolve_latest``. Unpublished curriculum IDs fail closed. Legacy
``edullm-datasets`` URIs are refused.

Architecture / hparams match the control / RefHQ contract (olmo2_370M, GBS
4_194_304, SkipStepAdamW, z_loss 1e-5, HSDP bf16, compile). Permanent
checkpoint ladder and task_loss hooks are imported from
``token_selection.olmo_ext`` — not duplicated.

Does **not** submit AWS compute jobs. S3 is used only for read-only staging of
published training/curriculum inputs from ``edullm-data`` at run start. No
checkpoint, progress, eval, metric, or other run artifact is written to S3.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    import wandb
except ImportError:  # pragma: no cover - optional until FarmShare venv installs it
    wandb = None  # type: ignore[assignment]

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
from token_selection.olmo_ext.task_loss_hook import resolve_eval_script

from curriculum_pacing import (
    CURRICULUM_DATASET_ID,
    DIFFICULTY_METRICS,
    PACING_NAMES,
    TOTAL_STEPS,
    CurriculumChunkStream,
    curriculum_order_group,
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
DEFAULT_WANDB_PROJECT = "curriculum"

# Canonical published corpora (edullm-data). Never use s3://edullm-datasets/.
DATA_BUCKET = "edullm-data"
LEGACY_DATA_BUCKET = "edullm-datasets"
DEFAULT_TRAIN_DATASET_ID = "pretrain/regmix-10b"
# Single published token-order dataset; each difficulty metric is a group.
DEFAULT_CURRICULUM_DATASET_ID = CURRICULUM_DATASET_ID


def _broadcast_rank0_success(ok: bool) -> bool:
    """Rank 0 supplies ``ok``; all ranks return the broadcast value."""
    if not is_distributed():
        return ok
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.broadcast(flag, src=0)
    return bool(int(flag.item()))


def _abort_all_ranks(message: str, *, ok: bool) -> None:
    """If rank-0 reports failure, every rank raises SystemExit after broadcast."""
    if not _broadcast_rank0_success(ok):
        raise SystemExit(message)


def durable_backend_ok(args: argparse.Namespace) -> tuple[bool, str]:
    """Require live W&B unless this is an explicit local-only smoke run."""
    if bool(getattr(args, "allow_local_only", False)):
        return True, "allow_local_only"
    has_wandb = (
        args.wandb_mode == "online"
        and wandb is not None
        and bool(os.environ.get("WANDB_API_KEY"))
    )
    if has_wandb:
        return True, "wandb=online"
    return False, (
        "production artifact durability requires the wandb package, WANDB_API_KEY, "
        "and --wandb-mode online; S3 artifact export is prohibited. "
        "Local smoke only: --wandb-mode disabled --allow-local-only"
    )


def wandb_enabled(args: argparse.Namespace) -> bool:
    return (
        get_rank() == 0
        and args.wandb_mode != "disabled"
        and wandb is not None
        and bool(os.environ.get("WANDB_API_KEY"))
    )


def init_wandb(args: argparse.Namespace, run_meta: dict, *, wandb_dir: Path) -> object | None:
    if not wandb_enabled(args):
        if get_rank() == 0 and args.wandb_mode != "disabled" and wandb is None:
            log.warning("wandb package missing; continuing without W&B")
        elif get_rank() == 0 and args.wandb_mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
            log.warning("WANDB_API_KEY unset; continuing without W&B")
        return None
    assert wandb is not None
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    id_path = Path(args.progress_dir) / "wandb_run_id.txt"
    run_id = id_path.read_text(encoding="utf-8").strip() if id_path.exists() else None
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **run_meta,
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or args.name,
        id=run_id,
        resume="allow" if run_id else None,
        config=config,
        dir=str(wandb_dir),
    )
    id_path.write_text(str(run.id), encoding="utf-8")
    log.info("wandb run=%s url=%s", run.id, run.url)
    run.alert(
        title="curriculum train job started",
        text=(
            f"run={run.name} id={run.id} arm={args.arm_id} "
            f"slurm_job={os.environ.get('SLURM_JOB_ID', 'n/a')} "
            f"host={os.environ.get('SLURMD_NODENAME', os.environ.get('HOSTNAME', 'n/a'))}"
        ),
        level=wandb.AlertLevel.INFO,
    )
    return run


def wandb_log(run: object | None, metrics: dict, *, step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def _wandb_artifact_name(arm_id: str, suffix: str) -> str:
    arm = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(arm_id))
    arm = arm.strip("-_.")
    if not arm:
        raise ValueError("arm_id must contain an artifact-name character")
    return f"{arm}-{suffix}"


def _wandb_log_artifact_and_wait(
    run: object | None,
    artifact: object,
    *,
    required: bool,
) -> object | None:
    """Log an artifact and synchronously confirm upload when required."""
    if run is None:
        if required:
            raise RuntimeError("W&B artifact upload required but no run is active")
        return None
    logged = run.log_artifact(artifact)
    uploaded = logged if logged is not None else artifact
    if required:
        wait = getattr(uploaded, "wait", None)
        if not callable(wait):
            raise RuntimeError("W&B artifact handle has no wait(); cannot verify upload")
        wait()
    return uploaded


def wandb_log_eval(
    run: object | None,
    payload: dict,
    *,
    step: int,
    eval_path: Path,
    arm_id: str,
    required: bool = False,
) -> None:
    """Log task-loss JSON (curriculum / OLMo-ladder shape) to W&B."""
    if run is None:
        if required:
            raise RuntimeError("W&B eval artifact upload required but no run is active")
        return
    assert wandb is not None
    metrics: Dict[str, float] = {}
    # SmolLM-shaped payloads (macro_mean + labels) and curriculum payloads (task_loss_bpb).
    if "macro_mean" in payload:
        metrics["eval/macro_bpb"] = float(payload["macro_mean"])
        labels = payload.get("labels") or {}
        for k, v in labels.items():
            metrics[f"eval/bpb/{k}"] = float(v)
    else:
        tl = payload.get("task_loss_bpb") or {}
        for k, v in tl.items():
            if not isinstance(v, (int, float)):
                continue
            if k == "macro_mean_task_loss_bpb":
                metrics["eval/macro_bpb"] = float(v)
            elif k == "core_avg_rc_5shot_bpb":
                metrics["eval/core_avg_bpb"] = float(v)
            else:
                metrics[f"eval/bpb/{k}"] = float(v)
    if "macro_mean_accuracy" in payload:
        metrics["eval/macro_acc"] = float(payload["macro_mean_accuracy"])
    for k, v in (payload.get("accuracy_labels") or {}).items():
        metrics[f"eval/acc/{k}"] = float(v)
    for k, v in (payload.get("task_families") or {}).items():
        metrics[f"eval/family_bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_families") or {}).items():
        metrics[f"eval/family_acc/{k}"] = float(v)
    if metrics:
        wandb_log(run, metrics, step=step)
    art = wandb.Artifact(
        name=_wandb_artifact_name(arm_id, f"eval-step{step:07d}"),
        type="eval",
        metadata={"step": step, "arm_id": arm_id},
    )
    art.add_file(str(eval_path), name=eval_path.name)
    _wandb_log_artifact_and_wait(run, art, required=required)


def wandb_log_checkpoint(
    run: object | None,
    ckpt_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    arm_id: str,
) -> str | None:
    if run is None:
        return None
    assert wandb is not None
    wandb_log(
        run,
        {
            "checkpoint/step": step,
            "checkpoint/tokens_seen": tokens_seen,
        },
        step=step,
    )
    art = wandb.Artifact(
        name=_wandb_artifact_name(arm_id, f"checkpoint-step{step:07d}"),
        type="model",
        metadata={"step": step, "tokens_seen": tokens_seen, "arm_id": arm_id},
    )
    art.add_dir(str(ckpt_dir))
    uploaded = _wandb_log_artifact_and_wait(run, art, required=True)
    ref = str(
        getattr(uploaded, "qualified_name", None)
        or getattr(uploaded, "name", None)
        or art.name
    )
    log.info("wandb confirmed checkpoint artifact %s", ref)
    return ref


def wandb_log_runtime_artifacts(
    run: object | None,
    *,
    arm_id: str,
    step: int,
    progress_dir: Path,
    task_loss_dir: Path,
    metrics_dir: Path,
    required: bool,
) -> None:
    """Snapshot non-checkpoint run artifacts from scratch into W&B."""
    if run is None:
        if required:
            raise RuntimeError("W&B runtime artifact upload required but no run is active")
        return
    assert wandb is not None
    art = wandb.Artifact(
        name=_wandb_artifact_name(arm_id, f"runtime-step{step:07d}"),
        type="run-state",
        metadata={"step": step, "arm_id": arm_id},
    )
    added = False
    for local, name in (
        (Path(progress_dir), "progress"),
        (Path(task_loss_dir), "task_loss_results"),
        (Path(metrics_dir), "metrics"),
    ):
        if local.is_dir() and any(p.is_file() for p in local.rglob("*")):
            art.add_dir(str(local), name=name)
            added = True
    if added:
        _wandb_log_artifact_and_wait(run, art, required=required)


def wandb_drain_task_loss_evals(
    run: object | None,
    results_dir: Path,
    logged_steps: Set[int],
    *,
    arm_id: str,
    required: bool = False,
) -> None:
    """Log completed task-loss JSON files not yet mirrored to W&B."""
    if run is None or not results_dir.is_dir():
        return
    for eval_path in sorted(results_dir.glob("step*_task_loss.json")):
        name = eval_path.name
        try:
            step = int(name[len("step") : name.index("_task_loss")])
        except ValueError:
            continue
        if step in logged_steps:
            continue
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("skip incomplete task_loss %s: %s", eval_path, exc)
            continue
        wandb_log_eval(
            run,
            payload,
            step=step,
            eval_path=eval_path,
            arm_id=arm_id,
            required=required,
        )
        logged_steps.add(step)
        log.info("wandb logged task_loss eval step=%s", step)


def wandb_upload_existing(
    run: object | None,
    *,
    save_folder: Path,
    task_loss_dir: Path,
    progress_dir: Path,
    metrics_dir: Path,
    arm_id: str,
    required: bool,
) -> None:
    if run is None:
        if required:
            raise RuntimeError("W&B existing-artifact upload required but no run is active")
        return
    assert wandb is not None
    if save_folder.is_dir():
        for ckpt_dir in sorted(save_folder.glob("step*")):
            if not (ckpt_dir / "state.pt").is_file():
                continue
            step = _checkpoint_step(ckpt_dir)
            tokens_seen = step * GLOBAL_BATCH_TOKENS
            wandb_log_checkpoint(
                run, ckpt_dir, step=step, tokens_seen=tokens_seen, arm_id=arm_id
            )
    logged: Set[int] = set()
    wandb_drain_task_loss_evals(
        run, task_loss_dir, logged, arm_id=arm_id, required=required
    )
    wandb_log_runtime_artifacts(
        run,
        arm_id=arm_id,
        step=max(logged, default=0),
        progress_dir=progress_dir,
        task_loss_dir=task_loss_dir,
        metrics_dir=metrics_dir,
        required=required,
    )


def stage_load_path(
    load_path: str,
    *,
    save_folder: Path,
    wandb_run: object | None,
) -> Path:
    """Resolve a local or W&B checkpoint bootstrap into job scratch."""
    _refuse_legacy_uri(load_path)
    if load_path.startswith("s3://"):
        raise SystemExit(
            "S3 checkpoint resume is prohibited: S3 is input-data/bootstrap staging "
            "only and must not store run checkpoints; use a local path or "
            "wandb-artifact://entity/project/name:version"
        )
    prefix = "wandb-artifact://"
    if not load_path.startswith(prefix):
        return Path(load_path)
    if wandb_run is None:
        raise SystemExit("W&B artifact resume requires an active online W&B run")
    artifact_ref = load_path[len(prefix) :].strip("/")
    if artifact_ref.count("/") < 2 or ":" not in artifact_ref.rsplit("/", 1)[-1]:
        raise SystemExit(
            "W&B checkpoint reference must be "
            "wandb-artifact://entity/project/name:version"
        )
    artifact = wandb_run.use_artifact(artifact_ref, type="model")
    dest = Path(save_folder) / "_wandb_resume"
    log.info("Resuming: download W&B artifact %s → %s", artifact_ref, dest)
    downloaded = Path(artifact.download(root=str(dest)))
    direct = downloaded / "state.pt"
    if direct.is_file():
        return downloaded
    matches = list(downloaded.rglob("state.pt"))
    if len(matches) != 1:
        raise SystemExit(
            f"W&B resume artifact {artifact_ref!r} must contain exactly one state.pt; "
            f"found {len(matches)}"
        )
    return matches[0].parent


def default_data_cache_dir() -> Path:
    """Job-scoped staging root for fetch-if-missing (empty scratch OK)."""
    for key in ("EDULLM_DATA_CACHE", "SCRATCH", "TMPDIR"):
        val = os.environ.get(key)
        if val:
            return Path(val) / "edullm-data-cache"
    return Path.cwd() / "edullm-data-cache"


def _refuse_legacy_uri(uri: str) -> None:
    if LEGACY_DATA_BUCKET in uri:
        raise SystemExit(
            f"refusing legacy training URI (use s3://{DATA_BUCKET}/ via edullm_data): {uri}"
        )


def _parse_edullm_data_uri(uri: str) -> Tuple[str, str]:
    _refuse_legacy_uri(uri)
    if not uri.startswith("s3://"):
        raise SystemExit(f"expected s3:// URI from dataset_paths, got {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if bucket != DATA_BUCKET:
        raise SystemExit(
            f"only s3://{DATA_BUCKET}/ training URIs are allowed; got {uri}"
        )
    if not key:
        raise SystemExit(f"empty key in URI {uri}")
    return bucket, key


def _edullm_s3():
    try:
        from edullm_data.s3 import Boto3S3
    except ImportError as e:
        raise SystemExit(
            "edullm-data package is required "
            '(install: uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@main")'
        ) from e
    return Boto3S3.default()


def resolve_published_split(
    dataset_id: str,
    *,
    version: Optional[str] = None,
    split: str = "train",
    group: Optional[str] = None,
):
    """Resolve a validated edullm-data split via dataset_paths / resolve_latest."""
    try:
        from edullm_data.read import dataset_paths, resolve_latest
    except ImportError as e:
        raise SystemExit(
            "edullm-data package is required "
            '(install: uv add "edullm-data @ git+https://github.com/edu-llm/edullm-data@main")'
        ) from e

    s3 = _edullm_s3()
    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(
            f"no published version of {dataset_id!r} under "
            f"s3://{DATA_BUCKET}/_catalog/ — publish+validate before training"
        )
    resolved = dataset_paths(
        dataset_id, ver, split=split, s3=s3, group=group
    )
    if not resolved.paths:
        raise SystemExit(
            f"{dataset_id}/{ver} split={split!r} resolved to zero paths"
        )
    for uri in resolved.paths:
        _parse_edullm_data_uri(uri)
    return resolved, ver


def stage_edullm_uris(uris: List[str], cache_dir: Path) -> List[str]:
    """Download missing edullm-data objects into cache_dir; return local paths."""
    import boto3

    client = boto3.client("s3", region_name="us-east-1")
    local_paths: List[str] = []
    cache_dir = Path(cache_dir)
    for uri in uris:
        bucket, key = _parse_edullm_data_uri(uri)
        dest = cache_dir / bucket / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        need_fetch = True
        if dest.is_file():
            try:
                head = client.head_object(Bucket=bucket, Key=key)
                if int(dest.stat().st_size) == int(head["ContentLength"]):
                    need_fetch = False
            except Exception as e:
                log.warning("HEAD %s failed (%s); re-fetching", uri, e)
        if need_fetch:
            log.info("staging %s → %s", uri, dest)
            tmp = dest.with_name(dest.name + ".partial")
            client.download_file(bucket, key, str(tmp))
            tmp.replace(dest)
        local_paths.append(str(dest))
    return local_paths


def resolve_and_stage_train_tokens(
    *,
    dataset_id: str,
    version: Optional[str],
    cache_dir: Path,
) -> Tuple[List[str], str, str, Any]:
    """Return (local_paths, resolved_version, numpy_dtype, ResolvedSplit)."""
    resolved, ver = resolve_published_split(
        dataset_id, version=version, split="train"
    )
    dtype = resolved.numpy_dtype or "uint32"
    if resolved.header_bytes:
        raise SystemExit(
            f"{dataset_id}/{ver} declares header_bytes={resolved.header_bytes}; "
            "this trainer only memmaps headerless .u32le.bin shards"
        )
    local = stage_edullm_uris(list(resolved.paths), cache_dir)
    return local, ver, dtype, resolved


def resolve_and_stage_curriculum_order(
    *,
    dataset_id: str,
    version: Optional[str],
    cache_dir: Path,
    group: str,
    expected_parent_dataset_id: str,
    expected_parent_version: str,
) -> Tuple[np.ndarray, str]:
    """Load a published token-order curriculum group as a uint32 ranked index array."""
    resolved, ver = resolve_published_split(
        dataset_id, version=version, split="train", group=group
    )
    validate_curriculum_parent_dependency(
        dataset_id=dataset_id,
        version=ver,
        group=group,
        expected_parent_dataset_id=expected_parent_dataset_id,
        expected_parent_version=expected_parent_version,
    )
    dtype = resolved.numpy_dtype or "<u4"
    local = stage_edullm_uris(list(resolved.paths), cache_dir)
    parts = [np.asarray(np.memmap(p, mode="r", dtype=dtype)) for p in local]
    ranked = np.concatenate(parts) if len(parts) > 1 else parts[0]
    if ranked.ndim != 1:
        raise SystemExit(
            f"{dataset_id}/{ver} group={group!r}: expected 1-D order vector, got shape {ranked.shape}"
        )
    return np.asarray(ranked, dtype=np.uint32), ver


def validate_curriculum_parent_dependency(
    *,
    dataset_id: str,
    version: str,
    group: str,
    expected_parent_dataset_id: str,
    expected_parent_version: str,
) -> dict:
    """Fail closed unless the order group binds the exact staged parent manifest."""
    s3 = _edullm_s3()
    curriculum = json.loads(
        s3.get(DATA_BUCKET, f"{dataset_id}/{version}/dataset.json").decode("utf-8")
    )
    groups = curriculum.get("groups") or []
    matches = [g for g in groups if g.get("name") == group]
    if len(matches) != 1:
        raise SystemExit(
            f"{dataset_id}/{version}: expected one group {group!r}, found {len(matches)}"
        )
    dependencies = matches[0].get("depends_on") or []
    pool_deps = [d for d in dependencies if d.get("role") == "token_pool"]
    if len(pool_deps) != 1:
        raise SystemExit(
            f"{dataset_id}/{version} group={group!r}: expected exactly one token_pool dependency"
        )
    dep = pool_deps[0]
    if (
        dep.get("dataset_id") != expected_parent_dataset_id
        or dep.get("version") != expected_parent_version
    ):
        raise SystemExit(
            f"{dataset_id}/{version} group={group!r} binds "
            f"{dep.get('dataset_id')}/{dep.get('version')}, but trainer staged "
            f"{expected_parent_dataset_id}/{expected_parent_version}"
        )
    expected_hash = dep.get("manifest_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise SystemExit(
            f"{dataset_id}/{version} group={group!r} dependency lacks manifest_sha256"
        )
    parent = json.loads(
        s3.get(
            DATA_BUCKET,
            f"{expected_parent_dataset_id}/{expected_parent_version}/dataset.json",
        ).decode("utf-8")
    )
    parent_hashes = {
        g.get("manifest_sha256")
        for g in parent.get("groups") or []
        if g.get("profile") == "pretrain-tokens/v1"
    }
    if parent_hashes != {expected_hash}:
        raise SystemExit(
            f"curriculum parent manifest hash {expected_hash} does not match "
            f"{expected_parent_dataset_id}/{expected_parent_version}: {sorted(parent_hashes)}"
        )
    return dep


def default_curriculum_order_group(metric: str) -> str:
    try:
        return curriculum_order_group(metric)
    except ValueError as e:
        raise SystemExit(str(e)) from e


@dataclass
class _Bookkeeping:
    """Minimal Trainer duck-type for TrainModule.optim_step / record_metric.

    Captures CE loss from ``record_ce_loss`` so the training loop can log
    ``train/loss`` to W&B (same spirit as SmolLM2 trainers).
    """

    global_step: int
    max_steps: int
    global_batch_size: int
    max_tokens: Optional[int] = None
    global_train_tokens_seen: int = 0
    dp_process_group: Any = None
    device: torch.device = torch.device("cuda")
    last_ce_loss: Optional[float] = None
    ce_loss_window_sum: float = 0.0
    ce_loss_window_n: int = 0

    def record_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_ce_loss(self, value: Any, *args: Any, **kwargs: Any) -> None:
        try:
            if torch.is_tensor(value):
                v = float(value.detach().float().mean().item())
            else:
                v = float(value)
        except Exception:
            return
        self.last_ce_loss = v
        self.ce_loss_window_sum += v
        self.ce_loss_window_n += 1

    def pop_ce_loss_avg(self) -> Optional[float]:
        if self.ce_loss_window_n <= 0:
            return self.last_ce_loss
        avg = self.ce_loss_window_sum / float(self.ce_loss_window_n)
        self.ce_loss_window_sum = 0.0
        self.ce_loss_window_n = 0
        return avg


class MemmapTokenDataset(Dataset):
    """Contiguous SEQ_LEN chunks over one or more uint32 token memmaps."""

    def __init__(
        self,
        paths: List[str],
        chunk_size: int = SEQ_LEN,
        dtype: Any = np.uint32,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=dtype)
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
        indexed: Dataset,
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
        tensors = [self.indexed[int(i)] for i in idxs]
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
    """All ranks gather; rank 0 atomically writes the local permanent checkpoint."""
    train_module_sd = gather_train_module_state_dict(train_module)
    ok = True
    err = "permanent checkpoint save failed"
    if get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=True)
            state = {
                "step": step,
                "train_module": train_module_sd,
                "args": vars(args),
                "meta": meta,
                "architecture": "olmo_core.TransformerConfig.olmo2_370M",
                "config_name": CONFIG_NAME,
                "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (curriculum)",
                "method": (
                    "plain_ce" if args.pacing == "control" else f"curriculum:{args.pacing}"
                ),
                "arm": args.arm_id,
                "run_id": args.name,
                "ephemeral": False,
                "checkpoint_format": "full_state_dict_v1",
            }
            tmp = path / "state.pt.tmp"
            torch.save(state, tmp)
            tmp.replace(path / "state.pt")
            (path / "step.txt").write_text(str(step) + "\n")
            log.info("Saved permanent full checkpoint → %s (step=%s)", path, step)
        except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
            ok = False
            err = f"permanent checkpoint save failed: {exc}"
            log.error("%s", err)
    _abort_all_ranks(err, ok=ok)


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


def _import_task_loss_eval_module(eval_script: Optional[str]) -> Any:
    script = resolve_eval_script(eval_script)
    if script is None:
        raise FileNotFoundError(
            "task-loss evaluator not found; set --task-loss-eval-script"
        )
    spec = importlib.util.spec_from_file_location(
        "eval_task_loss_olmo_core_curriculum", script
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import task-loss evaluator from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset_olmo_world_mesh() -> None:
    try:
        import olmo_core.distributed.parallel as parallel

        if getattr(parallel, "_WORLD_MESH", None) is not None:
            parallel._WORLD_MESH = None  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        log.warning("could not clear olmo_core world mesh: %s", exc)


def pause_eval_reload_curriculum(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    *,
    books: _Bookkeeping,
    lr: float,
    rank_micro_tokens: int,
    tokens_per_step: int,
) -> tuple[TransformerTrainModule, Optional[dict[str, Any]]]:
    """Consume the shared strict all-rank pause/eval/reload helper."""
    if not dist.is_initialized():
        raise RuntimeError(
            "production task-loss requires an initialized all-rank process group; "
            "local smoke may pass --no-task-loss-on-save"
        )
    evaluator = _import_task_loss_eval_module(args.task_loss_eval_script)
    out_path = Path(args.task_loss_results_dir) / f"step{step}_task_loss.json"

    def release_train_state() -> None:
        for attr in ("trainer", "_trainer", "train_module", "_train_module"):
            if hasattr(books, attr):
                try:
                    setattr(books, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def reload_train_state() -> TransformerTrainModule:
        _reset_olmo_world_mesh()
        restored = build_train_module(
            lr=lr,
            lr_warmup_steps=int(args.lr_warmup_steps),
            alpha_f=float(args.lr_alpha_f),
            compile_model=bool(args.compile),
            rank_microbatch_tokens=rank_micro_tokens,
        )
        restored._attach_trainer(books)  # type: ignore[arg-type]
        loaded_step = load_checkpoint(ckpt_dir, restored)
        if loaded_step != int(step):
            raise RuntimeError(
                f"reloaded checkpoint step {loaded_step} != expected {step}"
            )
        books.global_step = loaded_step
        books.global_train_tokens_seen = loaded_step * tokens_per_step
        return restored

    restored, payload = evaluator.pause_eval_reload_distributed(
        ckpt_dir,
        out_path,
        f"{args.name}-step{step}",
        release_train_state=release_train_state,
        reload_train_state=reload_train_state,
        base_config=Path(args.ladder_base_config),
        device_eval_batch_size=int(args.task_loss_eval_batch_size),
        strict=True,
    )
    if get_rank() == 0:
        if not out_path.is_file():
            raise RuntimeError(
                f"strict task-loss eval returned without output {out_path}"
            )
        if payload is None or not payload.get("suite_complete"):
            raise RuntimeError(
                f"strict task-loss eval at step {step} did not complete the label suite"
            )
        curve = Path(args.progress_dir) / "task_loss.jsonl"
        with curve.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return restored, payload


def publish_completed_checkpoint(
    args: argparse.Namespace,
    wandb_run: object | None,
    ckpt_dir: Path,
    step: int,
    eval_payload: Optional[dict[str, Any]],
) -> None:
    """Publish one completed checkpoint bundle to W&B, fail-closed across ranks."""
    required = not bool(args.allow_local_only)
    ok = True
    err = f"W&B checkpoint publication failed for step {step}"
    if get_rank() == 0:
        try:
            if eval_payload is not None:
                eval_path = (
                    Path(args.task_loss_results_dir) / f"step{step}_task_loss.json"
                )
                wandb_log_eval(
                    wandb_run,
                    eval_payload,
                    step=step,
                    eval_path=eval_path,
                    arm_id=args.arm_id,
                    required=required,
                )
            checkpoint_ref = wandb_log_checkpoint(
                wandb_run,
                ckpt_dir,
                step=step,
                tokens_seen=step * GLOBAL_BATCH_TOKENS,
                arm_id=args.arm_id,
            )
            if checkpoint_ref is not None:
                marker = {
                    "step": step,
                    "checkpoint_artifact": checkpoint_ref,
                    "arm_id": args.arm_id,
                }
                (Path(args.progress_dir) / "last_wandb_step.json").write_text(
                    json.dumps(marker, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            wandb_log_runtime_artifacts(
                wandb_run,
                arm_id=args.arm_id,
                step=step,
                progress_dir=Path(args.progress_dir),
                task_loss_dir=Path(args.task_loss_results_dir),
                metrics_dir=Path(args.metrics_dir),
                required=required,
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"W&B checkpoint publication failed for step {step}: {exc}"
            log.error("%s", err)
    _abort_all_ranks(err, ok=ok)


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
        help="Arm id for W&B artifact naming/logging (default derived from pacing+metric)",
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
        "--train-dataset-id",
        type=str,
        default=DEFAULT_TRAIN_DATASET_ID,
        help=f"Published edullm-data corpus (default {DEFAULT_TRAIN_DATASET_ID})",
    )
    ap.add_argument(
        "--train-dataset-version",
        type=str,
        default=None,
        help="Pin version (default: resolve_latest)",
    )
    ap.add_argument(
        "--curriculum-dataset-id",
        type=str,
        default=None,
        help=f"Published token-order curriculum id (default: {DEFAULT_CURRICULUM_DATASET_ID})",
    )
    ap.add_argument(
        "--curriculum-order-group",
        type=str,
        default=None,
        help="Order group within --curriculum-dataset-id (default: mapped from --difficulty-metric)",
    )
    ap.add_argument(
        "--curriculum-dataset-version",
        type=str,
        default=None,
        help="Pin curriculum version (default: resolve_latest)",
    )
    ap.add_argument(
        "--data-cache-dir",
        type=str,
        default=None,
        help="Local staging root (default: $EDULLM_DATA_CACHE/edullm-data-cache or cwd)",
    )
    ap.add_argument(
        "--train-paths-file",
        type=str,
        default=None,
        help="Optional override: already-staged local memmap path list for THIS job "
        "(must not point at legacy datasets bucket). Default: fetch from edullm-data.",
    )
    ap.add_argument(
        "--curriculum-index",
        type=str,
        default=None,
        help="Removed production path. Document-local indexes are rejected; "
        "use a published parent_pool_flat_chunks_v1 token-order dataset.",
    )
    ap.add_argument(
        "--save-folder",
        required=True,
        help="Job-scoped scratch checkpoint dir (uploaded to W&B)",
    )
    ap.add_argument(
        "--progress-dir",
        required=True,
        help="Job-scoped scratch progress dir (uploaded to W&B)",
    )
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
        "--load-path",
        type=str,
        default=None,
        help="Resume checkpoint: local step dir or "
        "wandb-artifact://entity/project/name:version (downloaded into scratch). "
        "S3 checkpoint paths are prohibited. No implicit scratch auto-resume.",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore leftover local checkpoints under --save-folder and start at step 0",
    )
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
    ap.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Job-scoped scratch metrics dir (default: sibling of --progress-dir "
        "named metrics); uploaded to W&B runtime artifacts",
    )
    ap.add_argument("--task-loss-eval-script", type=str, default=None)
    ap.add_argument(
        "--ladder-base-config",
        type=str,
        default=os.environ.get("LADDER_BASE_CONFIG") or None,
        help="Required ai2-olmo ladder config YAML for production task-loss eval",
    )
    ap.add_argument("--task-loss-eval-batch-size", type=int, default=4)
    ap.add_argument(
        "--task-loss-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        help=f"W&B project (default: {DEFAULT_WANDB_PROJECT})",
    )
    ap.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY") or None,
        help="W&B entity (default: account default / WANDB_ENTITY; do not invent)",
    )
    ap.add_argument(
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME") or None,
        help="W&B run name (default: --name / arm id)",
    )
    ap.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", "online"),
        choices=("online", "offline", "disabled"),
        help="W&B mode (default: online). Local smoke: disabled + --allow-local-only.",
    )
    ap.add_argument(
        "--wandb-upload-existing",
        action="store_true",
        help="On start, synchronously upload existing scratch checkpoints/evals to W&B.",
    )
    ap.add_argument(
        "--allow-local-only",
        action="store_true",
        help="Local-smoke escape hatch: allow scratch-only artifacts (not production).",
    )
    args = ap.parse_args()
    if args.pacing != "control" and not args.difficulty_metric:
        raise SystemExit("--difficulty-metric is required when --pacing != control")
    if args.fresh and args.load_path:
        raise SystemExit("--fresh and --load-path are mutually exclusive")
    if not args.fresh and not args.load_path:
        raise SystemExit(
            "choose recovery mode explicitly: pass --fresh or --load-path <stepN>"
        )
    if args.arm_id is None:
        args.arm_id = default_arm_id(args.pacing, args.difficulty_metric)
    if args.name is None:
        args.name = args.arm_id
    if args.data_cache_dir is None:
        args.data_cache_dir = str(default_data_cache_dir())
    if args.pacing != "control" and not args.curriculum_dataset_id and not args.curriculum_index:
        args.curriculum_dataset_id = DEFAULT_CURRICULUM_DATASET_ID
    if args.pacing != "control" and not args.curriculum_order_group and not args.curriculum_index:
        if not args.difficulty_metric:
            raise SystemExit("--difficulty-metric is required to select a curriculum order group")
        args.curriculum_order_group = default_curriculum_order_group(args.difficulty_metric)
    if args.train_paths_file:
        _refuse_legacy_uri(args.train_paths_file)
        if args.pacing != "control":
            raise SystemExit(
                "--train-paths-file is control/local-smoke only; curriculum pacing "
                "must resolve the published parent version and manifest dependency"
            )
    if args.curriculum_index:
        raise SystemExit(
            "--curriculum-index is a legacy document-local coordinate path and "
            "is not allowed for production; publish parent_pool_flat_chunks_v1 orders"
        )
    if args.load_path:
        _refuse_legacy_uri(args.load_path)
    if args.task_loss_on_save:
        if not args.ladder_base_config:
            raise SystemExit(
                "--ladder-base-config (or LADDER_BASE_CONFIG) is required when "
                "--task-loss-on-save is enabled"
            )
        if not Path(args.ladder_base_config).is_file():
            raise SystemExit(
                f"ladder base config not found: {args.ladder_base_config}"
            )
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        # Job-scoped under progress (not repo tree — scratch is ephemeral).
        args.task_loss_results_dir = str(Path(args.progress_dir) / "task_loss_results")
    if args.metrics_dir is None:
        # Keep train metrics beside progress on job-scoped scratch.
        args.metrics_dir = str(Path(args.progress_dir).parent / "metrics")
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

    ok_durable, durable_msg = durable_backend_ok(args)
    if not ok_durable:
        raise SystemExit(durable_msg)
    if rank == 0 and bool(getattr(args, "allow_local_only", False)):
        log.warning("WARN: --allow-local-only set; scratch artifacts may be lost")

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
    metrics_dir = Path(args.metrics_dir)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        Path(args.task_loss_results_dir).mkdir(parents=True, exist_ok=True)

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "arm": args.arm_id,
        "pacing": args.pacing,
        "difficulty_metric": args.difficulty_metric,
        "method": "plain_ce" if args.pacing == "control" else f"curriculum:{args.pacing}",
        "run_id": args.name,
        "artifact_backend": "wandb",
        "artifact_storage": "job_scratch_then_wandb",
        "s3_usage": "published_training_inputs_only",
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
        "ephemeral_runtime": True,
        "train_dataset_id": args.train_dataset_id,
        "train_dataset_version": None,  # filled after resolve
        "curriculum_dataset_id": args.curriculum_dataset_id,
        "curriculum_order_group": args.curriculum_order_group,
        "curriculum_dataset_version": None,
        "data_cache_dir": args.data_cache_dir,
        "train_dataset": f"s3://{DATA_BUCKET}/{args.train_dataset_id}/",
        "seed": args.seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_results_dir": args.task_loss_results_dir,
        "metrics_dir": args.metrics_dir,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "wandb_run_name": args.wandb_run_name or args.name,
        "durable_backend": durable_msg,
        "ema_merge_steps": [2000, 2125, 2250, 2384],
        "ema_alpha": 0.8,
    }
    wb_run: object | None = None
    logged_eval_steps: Set[int] = set()
    wandb_dir = Path(args.progress_dir).parent / "wandb"
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "checkpoint_ladder.json").write_text(
            json.dumps({"steps": ladder, "interval": int(args.save_interval)}, indent=2) + "\n"
        )
        log.info(
            "Plan: arm=%s pacing=%s metric=%s world=%d total=%d ladder_n=%d "
            "mbs=%d seqs/rank=%d lr=%.3e alpha_f=%s durable=%s",
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
            durable_msg,
        )
        if total_steps == TOTAL_STEPS and 2375 in ladder_set:
            raise SystemExit("BUG: ladder for 2384 must omit 2375")
    wandb_init_ok = True
    wandb_init_err = "W&B initialization failed"
    if rank == 0:
        try:
            wb_run = init_wandb(args, meta, wandb_dir=wandb_dir)
            if wb_run is not None and args.wandb_upload_existing:
                log.info("uploading existing scratch checkpoints/evals to wandb...")
                wandb_upload_existing(
                    wb_run,
                    save_folder=save_folder,
                    task_loss_dir=Path(args.task_loss_results_dir),
                    progress_dir=progress_dir,
                    metrics_dir=metrics_dir,
                    arm_id=args.arm_id,
                    required=not bool(args.allow_local_only),
                )
        except Exception as exc:  # noqa: BLE001
            wandb_init_ok = False
            wandb_init_err = f"W&B initialization/publication failed: {exc}"
            log.error("%s", wandb_init_err)
    _abort_all_ranks(wandb_init_err, ok=wandb_init_ok)

    cache_dir = Path(args.data_cache_dir)
    train_dtype: Any = np.uint32
    train_ver: Optional[str] = args.train_dataset_version
    curr_ver: Optional[str] = args.curriculum_dataset_version

    # Rank 0 stages from edullm-data when needed; all ranks share the cache path.
    if is_distributed():
        dist.barrier()

    control_stream: Optional[InfiniteBatchStream] = None
    curr_stream: Optional[CurriculumBatchStream] = None

    if args.train_paths_file:
        train_paths = read_paths(Path(args.train_paths_file))
        for p in train_paths:
            _refuse_legacy_uri(p)
        if rank == 0:
            log.info("Using --train-paths-file override (%d paths)", len(train_paths))
    else:
        if rank == 0:
            log.info(
                "Resolving+staging %s (version=%s) → %s",
                args.train_dataset_id,
                args.train_dataset_version or "latest",
                cache_dir,
            )
            train_paths, train_ver, train_dtype, _resolved = resolve_and_stage_train_tokens(
                dataset_id=args.train_dataset_id,
                version=args.train_dataset_version,
                cache_dir=cache_dir,
            )
            meta["train_dataset_version"] = train_ver
            meta["train_dataset"] = f"s3://{DATA_BUCKET}/{args.train_dataset_id}/{train_ver}/"
            meta["train_dtype"] = str(train_dtype)
            meta["train_n_shards"] = len(train_paths)
            (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            # Broadcast paths via a small sidecar all ranks can read.
            (cache_dir / "_stage_train_paths.json").write_text(
                json.dumps(
                    {
                        "dataset_id": args.train_dataset_id,
                        "version": train_ver,
                        "dtype": str(train_dtype),
                        "paths": train_paths,
                    }
                )
                + "\n"
            )
        if is_distributed():
            dist.barrier()
        stage_meta = json.loads((cache_dir / "_stage_train_paths.json").read_text(encoding="utf-8"))
        train_paths = list(stage_meta["paths"])
        train_ver = stage_meta.get("version")
        train_dtype = stage_meta.get("dtype") or np.uint32
        meta["train_dataset_version"] = train_ver
        meta["train_dataset"] = (
            f"s3://{DATA_BUCKET}/{args.train_dataset_id}/{train_ver}/"
            if train_ver
            else f"s3://{DATA_BUCKET}/{args.train_dataset_id}/"
        )

    train_ds = MemmapTokenDataset(train_paths, SEQ_LEN, dtype=train_dtype)

    if args.pacing == "control":
        workers = args.num_workers if world_size == 1 else max(1, args.num_workers // world_size)
        control_stream = InfiniteBatchStream(
            train_ds, mbs, workers, args.seed, rank=rank, world_size=world_size
        )
    else:
        assert args.difficulty_metric
        assert args.curriculum_dataset_id
        assert args.curriculum_order_group
        indexed: Dataset = train_ds
        if rank == 0:
            if not train_ver:
                raise SystemExit(
                    "curriculum pacing requires a resolved --train-dataset-version "
                    "to validate the exact token_pool dependency; "
                    "--train-paths-file is control/local-smoke only"
                )
            log.info(
                "Resolving+staging curriculum order %s group=%s (version=%s)",
                args.curriculum_dataset_id,
                args.curriculum_order_group,
                args.curriculum_dataset_version or "latest",
            )
            try:
                ranked, curr_ver = resolve_and_stage_curriculum_order(
                    dataset_id=args.curriculum_dataset_id,
                    version=args.curriculum_dataset_version,
                    cache_dir=cache_dir,
                    group=args.curriculum_order_group,
                        expected_parent_dataset_id=args.train_dataset_id,
                        expected_parent_version=str(train_ver),
                )
            except SystemExit:
                raise
            except Exception as e:
                raise SystemExit(
                    f"failed to resolve curriculum dataset "
                    f"{args.curriculum_dataset_id!r} group={args.curriculum_order_group!r}: {e}\n"
                    f"Publish token-order/v1 groups under "
                    f"s3://{DATA_BUCKET}/{args.curriculum_dataset_id}/ "
                    f"(depends_on {args.train_dataset_id}) before curriculum arms can train."
                ) from e
            meta["curriculum_dataset_version"] = curr_ver
            (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            order_path = cache_dir / f"_ranked_{args.difficulty_metric}.npy"
            np.save(order_path, ranked)
            (cache_dir / "_stage_curriculum_meta.json").write_text(
                json.dumps(
                    {
                        "dataset_id": args.curriculum_dataset_id,
                        "order_group": args.curriculum_order_group,
                        "version": curr_ver,
                        "order_path": str(order_path),
                        "n": int(ranked.shape[0]),
                    }
                )
                + "\n"
            )
        if is_distributed():
            dist.barrier()
        cmeta = json.loads(
            (cache_dir / "_stage_curriculum_meta.json").read_text(encoding="utf-8")
        )
        ranked = np.load(cmeta["order_path"], allow_pickle=False)
        curr_ver = cmeta.get("version")
        meta["curriculum_dataset_version"] = curr_ver

        if ranked.ndim != 1 or ranked.size != len(indexed):
            raise SystemExit(
                f"curriculum order length {ranked.size} != parent chunks {len(indexed)}"
            )
        expected = np.arange(len(indexed), dtype=np.uint32)
        if not np.array_equal(np.sort(np.asarray(ranked, dtype=np.uint32)), expected):
            raise SystemExit(
                "curriculum order is not a complete permutation of parent flat chunks"
            )
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
    if args.load_path:
        # Rank 0 stages the local/W&B bootstrap; all ranks share the scratch path.
        load_dir: Optional[Path] = None
        pull_ok = True
        pull_err = "failed to stage --load-path for resume"
        if rank == 0:
            try:
                load_dir = stage_load_path(
                    args.load_path,
                    save_folder=save_folder,
                    wandb_run=wb_run,
                )
                if not (load_dir / "state.pt").is_file():
                    raise SystemExit(
                        f"--load-path {load_dir} has no state.pt; expected a local "
                        "checkpoint or W&B model artifact"
                    )
                (save_folder / "_resume_load_path.txt").write_text(
                    str(load_dir) + "\n", encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                pull_ok = False
                pull_err = f"resume load-path failed: {exc}"
                log.error("%s", pull_err)
        _abort_all_ranks(pull_err, ok=pull_ok)
        if is_distributed():
            dist.barrier()
        load_dir = Path(
            (save_folder / "_resume_load_path.txt").read_text(encoding="utf-8").strip()
        )
        start_step = load_checkpoint(load_dir, train_module)
    elif args.fresh:
        if rank == 0:
            log.info("--fresh: starting from scratch (ephemeral runtime)")
    else:
        leftover = find_latest_checkpoint(save_folder)
        if leftover is not None:
            raise SystemExit(
                f"found local checkpoint {leftover} under job-scoped --save-folder; "
                "ephemeral runs do not auto-resume from scratch leftovers. "
                "Pass --load-path <local|wandb-artifact://entity/project/name:version> "
                "or --fresh to ignore and start at step 0."
            )
        if rank == 0:
            log.info("No --load-path; starting from scratch (ephemeral empty save-folder)")

    t0 = time.time()
    window_t0 = t0
    window_step0 = start_step
    loss_path = metrics_dir / "train_loss.jsonl"

    if is_distributed():
        dist.barrier()

    if start_step == 0 and 0 in ladder_set:
        if is_distributed():
            dist.barrier()
        ckpt0 = save_folder / "step0"
        save_checkpoint(ckpt0, 0, train_module, args, meta)
        eval_payload: Optional[dict[str, Any]] = None
        if args.task_loss_on_save:
            del train_module
            gc.collect()
            torch.cuda.empty_cache()
            train_module, eval_payload = pause_eval_reload_curriculum(
                args,
                ckpt0,
                0,
                books=books,
                lr=lr,
                rank_micro_tokens=rank_micro_tokens,
                tokens_per_step=tokens_per_step,
            )
        publish_completed_checkpoint(args, wb_run, ckpt0, 0, eval_payload)
        if rank == 0:
            logged_eval_steps.add(0)
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
                loss_avg = books.pop_ce_loss_avg()
                try:
                    lr_now = float(train_module.optim.param_groups[0]["lr"])
                except Exception:
                    lr_now = float(lr)
                log.info(
                    "step=%d/%d pacing=%s loss=%s tok/s=%.0f (avg=%.0f) lr=%.2e world=%d",
                    global_step,
                    total_steps,
                    args.pacing,
                    f"{loss_avg:.4f}" if loss_avg is not None else "n/a",
                    tok_s,
                    tok_s_avg,
                    lr_now,
                    world_size,
                )
                row = {
                    "step": global_step,
                    "pacing": args.pacing,
                    "tok_per_s": tok_s,
                    "tok_per_s_avg": tok_s_avg,
                    "lr": lr_now,
                }
                if loss_avg is not None:
                    row["loss"] = loss_avg
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                (progress_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "pacing": args.pacing,
                            "world_size": world_size,
                            "tok_per_s": tok_s,
                            "tok_per_s_avg": tok_s_avg,
                            "lr": lr_now,
                            "loss": loss_avg,
                            "pct": round(100.0 * global_step / total_steps, 4),
                        }
                    )
                    + "\n"
                )
                wb_metrics: Dict[str, float] = {
                    "train/lr": lr_now,
                    "train/tokens_seen": float(global_step * tokens_per_step),
                    "train/tok_per_s": float(tok_s),
                    "train/tok_per_s_avg": float(tok_s_avg),
                    "train/epoch_frac": float(global_step) / float(total_steps),
                }
                if loss_avg is not None:
                    wb_metrics["train/loss"] = float(loss_avg)
                wandb_log(wb_run, wb_metrics, step=global_step)

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            ckpt_dir = save_folder / f"step{global_step}"
            save_checkpoint(ckpt_dir, global_step, train_module, args, meta)
            eval_payload = None
            if args.task_loss_on_save:
                del train_module
                gc.collect()
                torch.cuda.empty_cache()
                train_module, eval_payload = pause_eval_reload_curriculum(
                    args,
                    ckpt_dir,
                    global_step,
                    books=books,
                    lr=lr,
                    rank_micro_tokens=rank_micro_tokens,
                    tokens_per_step=tokens_per_step,
                )
            publish_completed_checkpoint(
                args, wb_run, ckpt_dir, global_step, eval_payload
            )
            if rank == 0:
                logged_eval_steps.add(global_step)
            if is_distributed():
                dist.barrier()

    final_ok = True
    final_err = "final W&B artifact publication failed"
    if rank == 0:
        try:
            wandb_drain_task_loss_evals(
                wb_run,
                Path(args.task_loss_results_dir),
                logged_eval_steps,
                arm_id=args.arm_id,
                required=not bool(args.allow_local_only),
            )
            wandb_log_runtime_artifacts(
                wb_run,
                arm_id=args.arm_id,
                step=total_steps,
                progress_dir=progress_dir,
                task_loss_dir=Path(args.task_loss_results_dir),
                metrics_dir=metrics_dir,
                required=not bool(args.allow_local_only),
            )
            log.info(
                "Training complete at step=%d world_size=%d arm=%s durable=%s wandb=%s",
                total_steps,
                world_size,
                args.arm_id,
                durable_msg,
                getattr(wb_run, "url", None) if wb_run is not None else "off",
            )
            if wb_run is not None:
                wb_run.finish()
        except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
            final_ok = False
            final_err = f"final W&B artifact publication failed: {exc}"
            log.error("%s", final_err)
            if wb_run is not None:
                try:
                    wb_run.finish(exit_code=1)
                except Exception:
                    pass
    _abort_all_ranks(final_err, ok=final_ok)


if __name__ == "__main__":
    main()
