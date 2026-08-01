#!/usr/bin/env python3
"""OLMo2-370M CE trainer for mixlaw validation arms (10B, fixed recipe weights).

Streams from a **working pool** staged from published+validated
``s3://edullm-data/pretrain/olmo-127b`` (via ``edullm_data.read.dataset_paths`` /
``resolve_latest``). Domain proportions come from ``validation_mixtures_10b.json``
(or a per-arm ``mix_weights.json`` sidecar).

**Ephemeral runtime.** Assumes job-scoped scratch starts empty and is wiped after
the job. If ``--pool-dir`` is missing or incomplete, stages the pool from
edullm-data into ``--stage-dir`` (default: ``<progress-dir>/../pool``). Does not
assume FarmShare scratch, laptop corpora, old run dirs, or leftover checkpoints
already exist. Never reads ``s3://edullm-datasets/``.

**Durable saves.** Permanent ladder checkpoints live on job-local ``--save-folder``
/ ``--progress-dir``. Legacy launches record a local ``checkpoint_uri`` and may
upload checkpoint/eval artifacts to W&B. Platform launches additionally publish
each checkpoint beneath ``--checkpoint-prefix`` and progress/task-loss JSON beneath
``--output-prefix``. Both paths are fail-closed.

**W&B.** Training metrics, task-loss evals, and checkpoint artifacts log to
project ``mixlaw`` when ``--wandb-mode online|offline`` and ``WANDB_API_KEY``
are set (FarmShare: source ``wandb-session.env``). Online W&B is the off-scratch
durability layer alongside local scratch checkpoints.

Resume via explicit ``--load-path`` (local dir or ``s3://…/stepN`` which is
pulled into the job save folder). Leftover local checkpoints without
``--load-path`` / ``--fresh`` fail closed.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

_MIXLAW = Path(__file__).resolve().parent
_CUR_ROOT = _MIXLAW.parent.parent / "curriculum"
_TS_ROOT = _MIXLAW.parent.parent / "token-selection"
for _p in (_MIXLAW, _CUR_ROOT, _TS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Curriculum hard-disables W&B at import; snapshot/restore SmolLM-style session env.
from mixlaw_wandb import (  # noqa: E402
    add_wandb_args,
    finish_wandb,
    init_wandb,
    restore_wandb_env,
    snapshot_wandb_env,
    wandb_enabled,
    wandb_log,
    wandb_log_checkpoint,
    wandb_log_eval,
    wandb_upload_existing,
    wandb_wanted,
)

_WANDB_ENV_SNAPSHOT = snapshot_wandb_env()

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import seed_all

from token_selection.olmo_ext.checkpoint_ladder import (
    DEFAULT_CHECKPOINT_INTERVAL,
    permanent_checkpoint_steps,
)
from token_selection.olmo_ext.task_loss_hook import resolve_eval_script
from token_selection.olmo_ext.wandb_logging import task_loss_payload_complete

import train_curriculum_regmix_370m as curr  # noqa: E402

restore_wandb_env(_WANDB_ENV_SNAPSHOT)

from domain_stream import DomainMixtureStream  # noqa: E402
from mixlaw_common import DOMAINS  # noqa: E402
from mixlaw_runtime import (  # noqa: E402
    MIXLAW_DEFAULT_LENGTH_TOKENS,
    collect_dependency_versions,
    production_contract_errors,
)
from stage_validation_pool_from_edullm_data import (  # noqa: E402
    DEFAULT_DATASET_ID,
    pool_is_ready,
    resolve_dataset,
    stage_pool,
)

log = logging.getLogger("train_mixlaw_validation_370m")

SEQ_LEN = curr.SEQ_LEN
GLOBAL_BATCH_TOKENS = curr.GLOBAL_BATCH_TOKENS
MICROBATCH_TOKENS = curr.MICROBATCH_TOKENS
PEAK_LR = curr.PEAK_LR
DEFAULT_SEED = curr.DEFAULT_SEED
DEFAULT_LENGTH_TOKENS = MIXLAW_DEFAULT_LENGTH_TOKENS
CONFIG_NAME = "OLMo-2-370M-mixlaw-validation"
DEFAULT_RECIPE = _MIXLAW / "validation_mixtures_10b.json"


def save_checkpoint(
    path: Path,
    step: int,
    train_module: Any,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    """MixLaw local permanent save (all ranks gather; rank 0 writes ``state.pt``).

    W&B upload and the local durable marker are committed after this returns.
    """
    train_module_sd = curr.gather_train_module_state_dict(train_module)
    ok = True
    err = "MixLaw local checkpoint save failed"
    if get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=True)
            state = {
                "step": step,
                "train_module": train_module_sd,
                "args": vars(args),
                "meta": meta,
                "architecture": meta.get(
                    "architecture", "olmo_core.TransformerConfig.olmo2_370M"
                ),
                "config_name": meta.get("config_name", CONFIG_NAME),
                "train_stack": meta.get(
                    "train_stack",
                    "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
                ),
                "method": meta.get("method", "plain_ce_domain_stream"),
                "arm": args.mix_name,
                "run_id": args.name,
                "ephemeral": False,
                "checkpoint_format": "full_state_dict_v1",
            }
            tmp = path / "state.pt.tmp"
            torch.save(state, tmp)
            tmp.replace(path / "state.pt")
            (path / "step.txt").write_text(str(step) + "\n")
            log.info(
                "Saved MixLaw checkpoint → %s (step=%s mix=%s)",
                path,
                step,
                args.mix_name,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
            ok = False
            err = f"MixLaw local checkpoint save failed: {exc}"
            log.error("%s", err)
    curr._abort_all_ranks(err, ok=ok)


def _broadcast_export_status(
    ok: bool, msg: str, *, device: torch.device
) -> tuple[bool, str]:
    """Share rank-0 durable-export outcome with all ranks (fail-closed DDP)."""
    if not is_distributed():
        return ok, msg
    payload: list[Any] = [bool(ok), str(msg)]
    dist.broadcast_object_list(payload, src=0)
    return bool(payload[0]), str(payload[1])


def _commit_durable_checkpoint(
    args: argparse.Namespace,
    *,
    device: torch.device,
    ckpt_dir: Optional[Path] = None,
    progress_dir: Optional[Path] = None,
    durable_step: Optional[int] = None,
    wb_run: Any = None,
    wb_upload_ok: bool = True,
    what: str = "durable checkpoint commit",
) -> None:
    """Commit local/W&B durability and optional platform S3 publication."""
    from token_selection.olmo_ext.durability import write_last_durable_step

    wandb_production = wandb_wanted(args) and str(args.wandb_mode) == "online"
    platform_production = bool(getattr(args, "checkpoint_prefix", None))
    ok = True
    msg = ""
    if get_rank() == 0:
        remote_checkpoint_uri: Optional[str] = None
        if wandb_production and not wandb_enabled(args, is_main=True):
            ok = False
            msg = f"{what}: WANDB_MODE=online requires WANDB_API_KEY"
        elif wandb_production and wb_run is None:
            ok = False
            msg = f"{what}: W&B run not initialized"
        elif wandb_production and durable_step is not None and not wb_upload_ok:
            ok = False
            msg = f"{what}: W&B checkpoint upload failed for step {durable_step}"
        if ok and platform_production and durable_step is not None:
            if ckpt_dir is None:
                raise ValueError("platform durable step requires ckpt_dir")
            try:
                from platform_artifacts import join_s3_prefix, upload_checkpoint

                remote_checkpoint_uri = upload_checkpoint(
                    ckpt_dir,
                    join_s3_prefix(args.checkpoint_prefix, f"step{durable_step}"),
                    step=durable_step,
                    mix_name=args.mix_name,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed and broadcast
                ok = False
                msg = f"{what}: platform checkpoint upload failed: {exc}"
        if ok and durable_step is not None:
            if progress_dir is None or ckpt_dir is None:
                raise ValueError(
                    "durable_step requires progress_dir and ckpt_dir"
                )
            checkpoint_uri = remote_checkpoint_uri or str(Path(ckpt_dir).resolve())
            durability = "+".join(
                name
                for enabled, name in (
                    (True, "local"),
                    (platform_production, "platform-s3"),
                    (wandb_production, "wandb"),
                )
                if enabled
            )
            write_last_durable_step(
                progress_dir,
                durable_step,
                extra={
                    "checkpoint_uri": checkpoint_uri,
                    "durability": durability,
                    "mix_name": args.mix_name,
                    "run_id": args.name,
                    "recovery_mode": args.recovery_mode,
                    "dataset_id": args.dataset_id,
                    "dataset_version": getattr(
                        args, "resolved_dataset_version", None
                    ),
                    "task_loss_suite_complete": bool(args.task_loss_on_save),
                },
            )
            log.info("%s ok (checkpoint=%s)", what, checkpoint_uri)
        elif ok:
            log.info("%s ok (final flush)", what)
        if ok and getattr(args, "output_prefix", None):
            if progress_dir is None:
                progress_dir = Path(args.progress_dir)
            try:
                from platform_artifacts import upload_run_outputs

                upload_run_outputs(
                    progress_dir,
                    Path(args.task_loss_results_dir),
                    args.output_prefix,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed and broadcast
                ok = False
                msg = f"{what}: platform output upload failed: {exc}"

    ok, msg = _broadcast_export_status(ok, msg, device=device)
    if is_distributed():
        dist.barrier()
    if not ok:
        raise SystemExit(
            msg
            or (
                f"{what} failed; refusing to continue without a durable save. "
                "Fix W&B credentials or pass --wandb-mode disabled for local smoke."
            )
        )


def resolve_load_path(
    load_path: str,
    *,
    save_folder: Path,
    mix_name: str,
    device: torch.device,
) -> Path:
    """Resolve a local scratch checkpoint path."""
    raw = str(load_path).strip()
    if raw.startswith("s3://"):
        raise SystemExit(
            "S3 checkpoint resume is disabled; use local scratch or restore a "
            "W&B artifact to scratch before launch"
        )
    return Path(raw)


def load_mix_weights(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise SystemExit(f"{path}: missing weights dict")
    out = {d: float(weights[d]) for d in DOMAINS}
    return out, payload


def _import_task_loss_eval_module(eval_script: Optional[str] = None) -> Any:
    """Load ``eval_task_loss_olmo_core`` from its file path (not on PYTHONPATH)."""
    import importlib.util

    script = resolve_eval_script(eval_script)
    if script is None:
        raise FileNotFoundError(
            "task_loss eval script not found; set TASK_LOSS_EVAL_SCRIPT "
            "or --task-loss-eval-script"
        )
    spec = importlib.util.spec_from_file_location("eval_task_loss_olmo_core", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load eval module from {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reset_olmo_world_mesh() -> None:
    """Allow a second ``build_train_module`` in this process (pause/reload)."""
    try:
        import olmo_core.distributed.parallel as parallel

        if getattr(parallel, "_WORLD_MESH", None) is not None:
            parallel._WORLD_MESH = None  # type: ignore[attr-defined]
            log.info("Cleared olmo_core _WORLD_MESH for train-module rebuild")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not clear olmo_core world mesh: %s", exc)


def _pause_eval_reload(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    *,
    books: "_MixlawBooks",
    wb_run: Any = None,
    lr: float,
    lr_warmup_steps: int,
    lr_alpha_f: float,
    compile_model: bool,
    rank_micro_tokens: int,
    tokens_per_step: int,
) -> Any:
    """In-process multi-GPU task-loss then rebuild/reload the train module.

    Caller must already have saved ``ckpt_dir`` and dropped its ``train_module``
    reference (FSDP freed) before calling. Always rebuilds + reloads after eval;
    production strictness is enforced only after training state is restored.
    """
    if is_distributed():
        dist.barrier()

    rank = get_rank()
    # Break TrainModule ↔ books backrefs that can keep FSDP alive after del.
    for attr in ("trainer", "_trainer", "train_module", "_train_module"):
        if hasattr(books, attr):
            try:
                setattr(books, attr, None)
            except Exception:
                pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if is_distributed():
        dist.barrier()
    if rank == 0:
        log.info(
            "Pause/free/eval/reload: task-loss at step=%s (%s)",
            step,
            ckpt_dir,
        )
    results_dir = Path(args.task_loss_results_dir)
    out_path = results_dir / f"step{step}_task_loss.json"
    eval_error: Optional[BaseException] = None
    try:
        eval_mod = _import_task_loss_eval_module(args.task_loss_eval_script)
        payload = eval_mod.run_task_loss_eval_distributed(
            ckpt_dir,
            out_path,
            f"{args.name}-step{step}",
            base_config=Path(os.environ["LADDER_BASE_CONFIG"])
            if os.environ.get("LADDER_BASE_CONFIG", "").strip()
            else None,
        )
        if not task_loss_payload_complete(payload):
            raise RuntimeError(
                f"task-loss output at step={step} is not a complete 20-label BPB suite"
            )
        if rank == 0 and wb_run is not None and out_path.is_file():
            try:
                wandb_log_eval(wb_run, payload, step=step, eval_path=out_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("wandb eval log failed for %s: %s", out_path, exc)
    except Exception as exc:  # noqa: BLE001 — restore state before strict failure
        eval_error = exc
        log.error(
            "task-loss eval failed at step=%s (restoring train state): %s",
            step,
            exc,
            exc_info=True,
        )

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if is_distributed():
        dist.barrier()

    if rank == 0:
        log.info("Rebuilding train module and reloading %s", ckpt_dir)
    # olmo-core caches a process-global DeviceMesh; build_train_module would
    # raise "world mesh already exists" without clearing it after the first build.
    _reset_olmo_world_mesh()
    train_module = curr.build_train_module(
        lr=lr,
        lr_warmup_steps=int(lr_warmup_steps),
        alpha_f=float(lr_alpha_f),
        compile_model=bool(compile_model),
        rank_microbatch_tokens=int(rank_micro_tokens),
    )
    train_module._attach_trainer(books)  # type: ignore[arg-type]
    loaded = curr.load_checkpoint(ckpt_dir, train_module)
    books.global_step = int(loaded)
    books.global_train_tokens_seen = int(loaded) * int(tokens_per_step)
    if is_distributed():
        dist.barrier()
    if rank == 0:
        log.info("Resume training from step=%s after task-loss eval", loaded)
    if eval_error is not None and bool(args.task_loss_strict):
        raise RuntimeError(
            f"strict task-loss evaluation failed at step={step}: {eval_error}"
        ) from eval_error
    return train_module


def _maybe_pause_eval_reload(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    *,
    books: "_MixlawBooks",
    wb_run: Any = None,
    lr: float,
    lr_warmup_steps: int,
    lr_alpha_f: float,
    compile_model: bool,
    rank_micro_tokens: int,
    tokens_per_step: int,
) -> Any:
    """Run pause/free/eval/reload. Caller must already have ``del``'d ``train_module``."""
    return _pause_eval_reload(
        args,
        ckpt_dir,
        step,
        books=books,
        wb_run=wb_run,
        lr=lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_alpha_f=lr_alpha_f,
        compile_model=compile_model,
        rank_micro_tokens=rank_micro_tokens,
        tokens_per_step=tokens_per_step,
    )


@dataclass
class _MixlawBooks(curr._Bookkeeping):
    """Capture CE loss from TrainModule for W&B / jsonl (curriculum stub is a no-op)."""

    last_ce_loss: Optional[float] = field(default=None, repr=False)

    def record_ce_loss(self, *args: Any, **kwargs: Any) -> None:
        if not args:
            return
        value = args[0]
        try:
            if hasattr(value, "item"):
                self.last_ce_loss = float(value.item())
            else:
                self.last_ce_loss = float(value)
        except Exception:
            return


def _ensure_pool(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Return a ready local working pool, staging from edullm-data when needed."""
    pool_dir = Path(args.pool_dir) if args.pool_dir else None
    meta: dict[str, Any] = {
        "dataset_id": args.dataset_id,
        "version": args.dataset_version,
        "staged": False,
    }
    if pool_dir is not None and pool_is_ready(pool_dir):
        meta_path = pool_dir / "pool_meta.json"
        if meta_path.is_file():
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        else:
            # Still resolve so run_meta records a validated version even for older pools.
            from edullm_data.s3 import Boto3S3

            _, ver = resolve_dataset(args.dataset_id, args.dataset_version, s3=Boto3S3.default())
            meta["version"] = ver
        return pool_dir, meta

    if args.no_auto_stage:
        missing = pool_dir or "<unset>"
        raise SystemExit(
            f"working pool not ready at {missing}; refuse to auto-stage "
            f"(pass a staged --pool-dir or omit --no-auto-stage)"
        )

    stage_dir = Path(args.stage_dir) if args.stage_dir else (
        Path(args.progress_dir).resolve().parent / "pool"
    )
    if get_rank() == 0:
        log.info(
            "Staging working pool from edullm-data %s into %s",
            args.dataset_id,
            stage_dir,
        )
        summary = stage_pool(
            out_dir=stage_dir,
            mixtures_json=Path(args.mixtures_json),
            budget_tokens=int(args.stage_budget_tokens),
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            seed=int(args.stage_seed),
            mix_name=args.mix_name if args.selected_arm_stage else None,
            deterministic_prefix=bool(args.selected_arm_stage),
            delete_shards=bool(args.delete_stage_shards),
        )
        meta.update(summary)
        meta["staged"] = True
    if is_distributed():
        dist.barrier()
    if not pool_is_ready(stage_dir):
        raise SystemExit(f"pool still incomplete after staging: {stage_dir}")
    if get_rank() != 0:
        meta_path = stage_dir / "pool_meta.json"
        if meta_path.is_file():
            meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
    return stage_dir, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="Run id (e.g. mixlaw-370m-ML-near-opt-4)")
    ap.add_argument("--mix-name", required=True, help="Recipe run_name (for metadata)")
    ap.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Published edullm-data id (default: {DEFAULT_DATASET_ID})",
    )
    ap.add_argument(
        "--dataset-version",
        default=None,
        help="Pin edullm-data version; default resolve_latest()",
    )
    ap.add_argument(
        "--pool-dir",
        default=None,
        help="Optional staged working pool (tokenized/<domain>/<domain>.u32le.bin). "
        "If missing/incomplete, auto-stage from edullm-data unless --no-auto-stage.",
    )
    ap.add_argument(
        "--stage-dir",
        default=None,
        help="Where to stage the working pool when --pool-dir is absent/incomplete",
    )
    ap.add_argument(
        "--mixtures-json",
        default=str(DEFAULT_RECIPE),
        help="Recipe used for peak demand when auto-staging",
    )
    ap.add_argument("--stage-budget-tokens", type=int, default=DEFAULT_LENGTH_TOKENS)
    ap.add_argument("--stage-seed", type=int, default=6198)
    ap.add_argument(
        "--selected-arm-stage",
        action="store_true",
        help="Stage only --mix-name demand using deterministic per-domain shard prefixes",
    )
    ap.add_argument(
        "--delete-stage-shards",
        action="store_true",
        help="Delete temporary downloaded shards after successful concatenation",
    )
    ap.add_argument(
        "--no-auto-stage",
        action="store_true",
        help="Fail if --pool-dir is missing/incomplete instead of fetching from edullm-data",
    )
    ap.add_argument("--mix-weights-json", required=True, help="Per-arm weights sidecar")
    ap.add_argument(
        "--save-folder",
        required=True,
        help="Job-scoped local checkpoint dir (wiped with scratch; durable via S3 export)",
    )
    ap.add_argument(
        "--progress-dir",
        required=True,
        help="Job-scoped local progress dir (wiped with scratch; durable via S3 export)",
    )
    ap.add_argument(
        "--checkpoint-prefix",
        default=None,
        help="Platform S3 prefix for fail-closed checkpoint publication",
    )
    ap.add_argument(
        "--output-prefix",
        default=None,
        help="Platform S3 prefix for progress and task-loss publication",
    )
    ap.add_argument("--length-tokens", type=int, default=DEFAULT_LENGTH_TOKENS)
    ap.add_argument("--device-batch-size", type=int, default=MICROBATCH_TOKENS // SEQ_LEN)
    ap.add_argument("--save-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    ap.add_argument("--seed", type=int, default=None, help="Stream seed (default: recipe seed + mix id)")
    recovery = ap.add_mutually_exclusive_group()
    recovery.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Checkpoint dir to resume: local path or s3://…/stepN "
        "(S3 URI is pulled into --save-folder)",
    )
    recovery.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore leftover local checkpoints and start at step 0",
    )
    ap.add_argument(
        "--recovery-mode",
        choices=("fresh", "resume", "fail", "direct"),
        default="direct",
        help="Resolved launcher recovery policy, recorded in run metadata",
    )
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=0.1)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--task-loss-results-dir", type=str, default=None)
    ap.add_argument("--task-loss-eval-script", type=str, default=None)
    ap.add_argument(
        "--task-loss-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--task-loss-strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort after restoring train state if a permanent-checkpoint eval fails",
    )
    ap.add_argument(
        "--dependency-metadata",
        type=str,
        default=None,
        help="JSON emitted by preflight_validation_370m.py",
    )
    ap.add_argument(
        "--s3-export",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated no-op for checkpoint export (checkpoints use local scratch + W&B)",
    )
    add_wandb_args(ap)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        args.task_loss_results_dir = str(
            Path(args.progress_dir).resolve().parent / "task_loss_results"
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
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device("cuda")
    production_durable = bool(args.checkpoint_prefix) or (
        wandb_wanted(args) and str(args.wandb_mode) == "online"
    )
    contract_errors = production_contract_errors(
        durable_export=production_durable,
        task_loss_on_save=bool(args.task_loss_on_save),
        task_loss_strict=bool(args.task_loss_strict),
    )
    if contract_errors:
        raise SystemExit("; ".join(contract_errors))

    dependency_preflight: dict[str, Any] = {}
    if args.dependency_metadata:
        dependency_path = Path(args.dependency_metadata)
        if not dependency_path.is_file():
            raise SystemExit(f"dependency preflight metadata missing: {dependency_path}")
        dependency_preflight = json.loads(
            dependency_path.read_text(encoding="utf-8")
        )
    elif production_durable:
        raise SystemExit(
            "production W&B durability requires --dependency-metadata from "
            "preflight_validation_370m.py"
        )

    weights, mix_meta = load_mix_weights(Path(args.mix_weights_json))
    stream_seed = int(args.seed) if args.seed is not None else int(
        mix_meta.get("stream_seed", mix_meta.get("recipe_seed", DEFAULT_SEED))
        + int(mix_meta.get("id", 0))
    )
    seed_all(stream_seed + rank)

    pool_dir, pool_meta = _ensure_pool(args)
    dataset_id = str(pool_meta.get("dataset_id") or args.dataset_id)
    dataset_version = str(pool_meta.get("version") or args.dataset_version or "")
    args.resolved_dataset_version = dataset_version

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
    ladder = permanent_checkpoint_steps(total_steps, int(args.save_interval))
    ladder_set: Set[int] = set(ladder)
    lr = float(PEAK_LR)

    stream = DomainMixtureStream(
        pool_dir,
        weights,
        domains=DOMAINS,
        seq_len=SEQ_LEN,
        seed=stream_seed,
        rank=rank,
        world_size=world_size,
    )

    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        Path(args.task_loss_results_dir).mkdir(parents=True, exist_ok=True)
    marker_ok = True
    marker_error = ""
    if rank == 0:
        from token_selection.olmo_ext.durability import LAST_DURABLE_STEP_FILENAME

        durable_marker = progress_dir / LAST_DURABLE_STEP_FILENAME
        if args.fresh:
            # Explicit fresh is the only policy allowed to reset a monotonic
            # local/S3 durable pointer for this mix.
            durable_marker.unlink(missing_ok=True)
        elif not args.load_path and durable_marker.is_file():
            marker_ok = False
            marker_error = (
                f"found {durable_marker}; recovery policy did not select resume. "
                "Use RECOVERY_MODE=resume/--load-path or explicit fresh."
            )
    marker_ok, marker_error = _broadcast_export_status(
        marker_ok,
        marker_error,
        device=device,
    )
    if not marker_ok:
        raise SystemExit(marker_error)
    if is_distributed():
        dist.barrier()

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "arm": args.mix_name,
        "method": "plain_ce_domain_stream",
        "run_id": args.name,
        "artifact_durability": (
            "local_scratch+platform_s3"
            if args.checkpoint_prefix
            else "local_scratch+wandb"
        ),
        "train_stack": "TransformerTrainModule HSDP bf16 SkipStepAdamW compile",
        "tokenizer": curr.TOKENIZER_ID,
        "vocab_size": curr.EMBEDDING_SIZE,
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
        "save_interval": int(args.save_interval),
        "permanent_checkpoint_steps": ladder,
        "pool_dir": str(pool_dir),
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "edullm_data_uri": f"s3://edullm-data/{dataset_id}/{dataset_version}/"
        if dataset_version
        else f"s3://edullm-data/{dataset_id}/",
        "pool_staged": bool(pool_meta.get("staged")),
        "sampling": "domain_stratified_stream",
        "domain_weights": weights,
        "mix_weights_json": str(Path(args.mix_weights_json).resolve()),
        "recipe": mix_meta.get("recipe"),
        "stream_seed": stream_seed,
        "seed": stream_seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_strict": bool(args.task_loss_strict),
        "task_loss_results_dir": args.task_loss_results_dir,
        "recovery_mode": args.recovery_mode,
        "load_path": args.load_path,
        "dependency_preflight": dependency_preflight,
        "dependency_versions": dependency_preflight.get(
            "dependencies", collect_dependency_versions()
        ),
        "ephemeral_runtime": True,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "checkpoint_prefix": args.checkpoint_prefix,
        "output_prefix": args.output_prefix,
    }
    wb_run = None
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "checkpoint_ladder.json").write_text(
            json.dumps({"steps": ladder, "interval": int(args.save_interval)}, indent=2) + "\n"
        )
        log.info(
            "Plan: mix=%s world=%d total=%d weights=%s pool=%s data=%s/%s durable=%s wandb=%s",
            args.mix_name,
            world_size,
            total_steps,
            {k: round(v, 4) for k, v in weights.items()},
            pool_dir,
            dataset_id,
            dataset_version or "latest",
            bool(production_durable),
            args.wandb_mode,
        )
        wb_run = init_wandb(
            args,
            meta,
            id_dir=progress_dir,
            is_main=True,
            tags=["mixlaw", "370m-validation", args.mix_name],
            alert_title="mixlaw 370m validation started",
        )
        if wb_run is not None and args.wandb_upload_existing:
            log.info("uploading existing checkpoints/evals to wandb...")
            wandb_upload_existing(
                wb_run,
                checkpoints_root=save_folder,
                task_loss_dir=Path(args.task_loss_results_dir),
                progress_dir=progress_dir,
                tokens_per_step=tokens_per_step,
            )

    train_module = curr.build_train_module(
        lr=lr,
        lr_warmup_steps=int(args.lr_warmup_steps),
        alpha_f=float(args.lr_alpha_f),
        compile_model=bool(args.compile),
        rank_microbatch_tokens=rank_micro_tokens,
    )
    books = _MixlawBooks(
        global_step=0,
        max_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        device=device,
    )
    train_module._attach_trainer(books)  # type: ignore[arg-type]

    start_step = 0
    if args.load_path:
        load_dir = resolve_load_path(
            args.load_path,
            save_folder=save_folder,
            mix_name=args.mix_name,
            device=device,
        )
        if not (load_dir / "state.pt").is_file():
            raise SystemExit(
                f"--load-path {load_dir} has no state.pt; platform S3 restore is "
                "not supported, so restore the checkpoint to local scratch first"
            )
        start_step = curr.load_checkpoint(load_dir, train_module)
    elif args.fresh:
        if rank == 0:
            log.info("--fresh: starting from scratch (ephemeral runtime)")
    else:
        leftover = curr.find_latest_checkpoint(save_folder)
        if leftover is not None:
            raise SystemExit(
                f"found local checkpoint {leftover} under job-scoped --save-folder; "
                "ephemeral runs do not auto-resume from scratch leftovers. "
                "Pass --load-path <dir|s3://…/stepN> or --fresh "
                "to ignore and start at step 0."
            )
        if rank == 0:
            log.info("No --load-path; starting from scratch (ephemeral empty save-folder)")

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
        if bool(args.task_loss_on_save):
            # Drop the last live ref before eval so FSDP+optim can free.
            del train_module
            train_module = _maybe_pause_eval_reload(
                args,
                ckpt0,
                0,
                books=books,
                wb_run=wb_run,
                lr=lr,
                lr_warmup_steps=int(args.lr_warmup_steps),
                lr_alpha_f=float(args.lr_alpha_f),
                compile_model=bool(args.compile),
                rank_micro_tokens=rank_micro_tokens,
                tokens_per_step=tokens_per_step,
            )
        if rank == 0 and wb_run is not None:
            wb_upload_ok = wandb_log_checkpoint(wb_run, ckpt0, step=0, tokens_seen=0)
        else:
            wb_upload_ok = rank != 0 or not production_durable
        _commit_durable_checkpoint(
            args,
            device=device,
            ckpt_dir=ckpt0,
            progress_dir=progress_dir,
            durable_step=0,
            wb_run=wb_run,
            wb_upload_ok=wb_upload_ok,
            what=f"durable commit step0 ({args.mix_name})",
        )

    for step in range(start_step, total_steps):
        books.global_step = step
        books.global_train_tokens_seen = step * tokens_per_step

        input_ids = stream.next_input_ids(seqs_per_rank, device=device)
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
            tokens_seen = global_step * tokens_per_step
            if rank == 0:
                row = {
                    "step": global_step,
                    "mix_name": args.mix_name,
                    "tok_per_s": tok_s,
                    "tok_per_s_avg": tok_s_avg,
                    "tokens_seen": tokens_seen,
                    "train_loss": books.last_ce_loss,
                    "weights": stream.weights_dict(),
                }
                log.info(
                    "step=%d/%d mix=%s loss=%s tok/s=%.0f (avg=%.0f) world=%d",
                    global_step,
                    total_steps,
                    args.mix_name,
                    f"{books.last_ce_loss:.4f}" if books.last_ce_loss is not None else "n/a",
                    tok_s,
                    tok_s_avg,
                    world_size,
                )
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                wb_metrics: Dict[str, Any] = {
                    "train/tok_per_s": tok_s,
                    "train/tok_per_s_avg": tok_s_avg,
                    "train/tokens_seen": tokens_seen,
                    "train/epoch": global_step / max(total_steps, 1),
                }
                if books.last_ce_loss is not None:
                    wb_metrics["train/loss"] = books.last_ce_loss
                wandb_log(wb_run, wb_metrics, step=global_step)

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            ckpt_dir = save_folder / f"step{global_step}"
            save_checkpoint(ckpt_dir, global_step, train_module, args, meta)
            if bool(args.task_loss_on_save):
                # Drop the last live ref before eval so FSDP+optim can free.
                del train_module
                train_module = _maybe_pause_eval_reload(
                    args,
                    ckpt_dir,
                    global_step,
                    books=books,
                    wb_run=wb_run,
                    lr=lr,
                    lr_warmup_steps=int(args.lr_warmup_steps),
                    lr_alpha_f=float(args.lr_alpha_f),
                    compile_model=bool(args.compile),
                    rank_micro_tokens=rank_micro_tokens,
                    tokens_per_step=tokens_per_step,
                )
            if rank == 0 and wb_run is not None:
                wb_upload_ok = wandb_log_checkpoint(
                    wb_run,
                    ckpt_dir,
                    step=global_step,
                    tokens_seen=global_step * tokens_per_step,
                )
            else:
                wb_upload_ok = rank != 0 or not production_durable
            _commit_durable_checkpoint(
                args,
                device=device,
                ckpt_dir=ckpt_dir,
                progress_dir=progress_dir,
                durable_step=global_step,
                wb_run=wb_run,
                wb_upload_ok=wb_upload_ok,
                what=f"durable commit step{global_step} ({args.mix_name})",
            )

    _commit_durable_checkpoint(
        args,
        device=device,
        progress_dir=progress_dir,
        wb_run=wb_run,
        what=f"final durable flush ({args.mix_name})",
    )
    if rank == 0:
        if wb_run is not None:
            # Catch async eval JSONs that finished after the ladder step.
            wandb_upload_existing(
                wb_run,
                task_loss_dir=Path(args.task_loss_results_dir),
                progress_dir=progress_dir,
                tokens_per_step=tokens_per_step,
            )
            finish_wandb(wb_run)
        log.info(
            "Training complete mix=%s step=%d durable=%s wandb=%s",
            args.mix_name,
            total_steps,
            bool(production_durable),
            args.wandb_mode,
        )


if __name__ == "__main__":
    main()
