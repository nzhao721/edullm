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

**Durable saves.** Permanent ladder checkpoints + progress/task_loss are written
to job-local ``--save-folder`` / ``--progress-dir``, then uploaded to
``s3://edullm-checkpoints/mixlaw/370m-validation/<mix_name>/``. Export is
**fail-closed**: missing aws/creds or a failed ``aws s3 sync`` aborts training
on all ranks (DDP broadcast). Opt out only for local smoke via
``--no-s3-export`` or ``S3_EXPORT=0`` / ``SKIP_S3_UPLOAD=1``.

**W&B.** Training metrics, task-loss evals, and checkpoint artifacts log to
project ``mixlaw`` when ``--wandb-mode online|offline`` and ``WANDB_API_KEY``
are set (FarmShare: source ``wandb-session.env``). W&B is additive — it does
not relax the S3 fail-closed contract.

Resume via explicit ``--load-path`` (local dir or ``s3://…/stepN`` which is
pulled into the job save folder). Leftover local checkpoints without
``--load-path`` / ``--fresh`` fail closed.
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
    wandb_log,
    wandb_log_checkpoint,
    wandb_log_eval,
    wandb_upload_existing,
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
from token_selection.olmo_ext.task_loss_hook import trigger_task_loss_eval

import train_curriculum_regmix_370m as curr  # noqa: E402

restore_wandb_env(_WANDB_ENV_SNAPSHOT)

from domain_stream import DomainMixtureStream  # noqa: E402
from mixlaw_common import DOMAINS  # noqa: E402
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
DEFAULT_LENGTH_TOKENS = curr.DEFAULT_LENGTH_TOKENS
CONFIG_NAME = "OLMo-2-370M-mixlaw-validation"
CHECKPOINT_BUCKET = "edullm-checkpoints"
MIXLAW_S3_ROOT = "mixlaw/370m-validation"
DEFAULT_RECIPE = _MIXLAW / "validation_mixtures_10b.json"


def mixlaw_s3_uri(mix_name: str, *parts: str) -> str:
    base = f"s3://{CHECKPOINT_BUCKET}/{MIXLAW_S3_ROOT}/{mix_name.strip('/')}"
    extra = "/".join(p.strip("/") for p in parts if str(p).strip())
    return f"{base}/{extra}" if extra else f"{base}/"


def export_mixlaw_checkpoint(
    mix_name: str,
    checkpoint_dir: Path,
    *,
    enabled: Optional[bool] = None,
) -> bool:
    """Upload one ``stepN/`` dir under ``mixlaw/370m-validation/<mix>/checkpoints/``."""
    from token_selection.olmo_ext.s3_export import sync_to_s3

    ckpt = Path(checkpoint_dir)
    return sync_to_s3(
        ckpt, mixlaw_s3_uri(mix_name, "checkpoints", ckpt.name), enabled=enabled
    )


def export_mixlaw_artifacts(
    mix_name: str,
    *,
    checkpoints_root: Optional[Path] = None,
    progress_dir: Optional[Path] = None,
    task_loss_dir: Optional[Path] = None,
    enabled: Optional[bool] = None,
) -> tuple[bool, str]:
    """Sync arm trees to durable S3. Returns ``(ok, error_message)``."""
    from token_selection.olmo_ext.s3_export import sync_to_s3

    targets: list[tuple[Path, str]] = []
    if checkpoints_root is not None:
        targets.append(
            (Path(checkpoints_root), mixlaw_s3_uri(mix_name, "checkpoints"))
        )
    if progress_dir is not None:
        targets.append((Path(progress_dir), mixlaw_s3_uri(mix_name, "progress")))
    if task_loss_dir is not None and Path(task_loss_dir).is_dir():
        targets.append(
            (Path(task_loss_dir), mixlaw_s3_uri(mix_name, "task_loss_results"))
        )
    for local, remote in targets:
        if not local.exists():
            return False, f"S3 export skip: missing local path {local}"
        if not sync_to_s3(local, remote, enabled=enabled):
            return False, f"S3 export failed ({local} → {remote})"
    return True, ""


def _broadcast_export_status(
    ok: bool, msg: str, *, device: torch.device
) -> tuple[bool, str]:
    """Share rank-0 durable-export outcome with all ranks (fail-closed DDP)."""
    if not is_distributed():
        return ok, msg
    payload: list[Any] = [bool(ok), str(msg)]
    dist.broadcast_object_list(payload, src=0)
    return bool(payload[0]), str(payload[1])


def _require_durable_export(
    args: argparse.Namespace,
    *,
    device: torch.device,
    ckpt_dir: Optional[Path] = None,
    progress_dir: Optional[Path] = None,
    checkpoints_root: Optional[Path] = None,
    task_loss_dir: Optional[Path] = None,
    what: str = "durable S3 export",
) -> None:
    """Fail-closed S3 export after permanent saves / end-of-run.

    When ``--no-s3-export`` / ``S3_EXPORT=0`` / ``SKIP_S3_UPLOAD=1``, skip
    (local smoke). Otherwise aws/creds/sync failures abort every rank.
    """
    from token_selection.olmo_ext.s3_export import s3_export_enabled

    ok = True
    msg = ""
    if get_rank() == 0:
        # --no-s3-export forces off; otherwise honor S3_EXPORT=0 / SKIP_S3_UPLOAD=1.
        want = bool(args.s3_export) and s3_export_enabled()
        if not want:
            log.info(
                "S3 export disabled (--no-s3-export / S3_EXPORT=0); "
                "local-only smoke — artifacts will be lost when scratch is wiped"
            )
        else:
            try:
                if ckpt_dir is not None:
                    if not export_mixlaw_checkpoint(
                        args.mix_name, ckpt_dir, enabled=True
                    ):
                        ok = False
                        msg = (
                            f"{what} failed for checkpoint {ckpt_dir} → "
                            f"{mixlaw_s3_uri(args.mix_name, 'checkpoints', Path(ckpt_dir).name)}"
                        )
                if ok and (
                    progress_dir is not None
                    or checkpoints_root is not None
                    or task_loss_dir is not None
                ):
                    art_ok, art_msg = export_mixlaw_artifacts(
                        args.mix_name,
                        checkpoints_root=checkpoints_root,
                        progress_dir=progress_dir,
                        task_loss_dir=task_loss_dir,
                        enabled=True,
                    )
                    if not art_ok:
                        ok = False
                        msg = art_msg or f"{what} failed"
            except Exception as exc:  # noqa: BLE001
                ok = False
                msg = f"{what} raised: {exc}"
            if ok:
                log.info("%s ok", what)
            else:
                log.error("%s", msg)

    ok, msg = _broadcast_export_status(ok, msg, device=device)
    if is_distributed():
        dist.barrier()
    if not ok:
        raise SystemExit(
            msg
            or (
                f"{what} failed; refusing to continue without a durable save. "
                "Fix aws/creds, or pass --no-s3-export / S3_EXPORT=0 for local smoke."
            )
        )


def resolve_load_path(
    load_path: str,
    *,
    save_folder: Path,
    mix_name: str,
    device: torch.device,
) -> Path:
    """Resolve ``--load-path`` (local dir or ``s3://…/stepN``) to a local checkpoint."""
    raw = str(load_path).strip()
    if not raw.startswith("s3://"):
        return Path(raw)

    step_name = Path(raw.rstrip("/")).name
    if not step_name.startswith("step"):
        raise SystemExit(
            f"--load-path S3 URI must end with stepN (got {raw!r})"
        )
    dest = save_folder / step_name
    ok = True
    msg = ""
    if get_rank() == 0:
        try:
            from token_selection.olmo_ext.s3_export import sync_from_s3

            log.info("Pulling --load-path %s → %s", raw, dest)
            sync_from_s3(raw, dest, enabled=True, raise_on_error=True)
            if not (dest / "state.pt").is_file():
                ok = False
                msg = f"S3 --load-path {raw} synced to {dest} but state.pt is missing"
        except Exception as exc:  # noqa: BLE001
            ok = False
            msg = f"failed to pull --load-path {raw}: {exc}"
    ok, msg = _broadcast_export_status(ok, msg, device=device)
    if is_distributed():
        dist.barrier()
    if not ok:
        raise SystemExit(msg or f"failed to materialize --load-path {raw}")
    return dest


def load_mix_weights(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise SystemExit(f"{path}: missing weights dict")
    out = {d: float(weights[d]) for d in DOMAINS}
    return out, payload


def _maybe_task_loss(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    *,
    wb_run: Any = None,
) -> None:
    if get_rank() != 0:
        return
    results_dir = Path(args.task_loss_results_dir)
    out_path = results_dir / f"step{step}_task_loss.json"
    trigger_task_loss_eval(
        ckpt_dir,
        run_name=f"{args.name}-step{step}",
        out_path=out_path,
        eval_script=args.task_loss_eval_script,
        enabled=None if args.task_loss_on_save else False,
        # Sync when W&B is live so eval metrics land on the same step.
        async_=wb_run is None,
    )
    if wb_run is not None and out_path.is_file():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            wandb_log_eval(wb_run, payload, step=step, eval_path=out_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb eval log failed for %s: %s", out_path, exc)


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
    ap.add_argument("--length-tokens", type=int, default=DEFAULT_LENGTH_TOKENS)
    ap.add_argument("--device-batch-size", type=int, default=MICROBATCH_TOKENS // SEQ_LEN)
    ap.add_argument("--save-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    ap.add_argument("--seed", type=int, default=None, help="Stream seed (default: recipe seed + mix id)")
    ap.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Checkpoint dir to resume: local path or s3://…/stepN "
        "(S3 URI is pulled into --save-folder)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore leftover local checkpoints and start at step 0",
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
        "--s3-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail-closed upload of permanent saves to "
        "s3://edullm-checkpoints/mixlaw/370m-validation/<mix>/ "
        "(opt out with --no-s3-export or S3_EXPORT=0 for local smoke only)",
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

    weights, mix_meta = load_mix_weights(Path(args.mix_weights_json))
    stream_seed = int(args.seed) if args.seed is not None else int(
        mix_meta.get("stream_seed", mix_meta.get("recipe_seed", DEFAULT_SEED))
        + int(mix_meta.get("id", 0))
    )
    seed_all(stream_seed + rank)

    pool_dir, pool_meta = _ensure_pool(args)
    dataset_id = str(pool_meta.get("dataset_id") or args.dataset_id)
    dataset_version = str(pool_meta.get("version") or args.dataset_version or "")

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

    s3_prefix = mixlaw_s3_uri(args.mix_name)
    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "arm": args.mix_name,
        "method": "plain_ce_domain_stream",
        "run_id": args.name,
        "s3_prefix": s3_prefix,
        "s3_export": bool(args.s3_export),
        "s3_export_prefix": s3_prefix,
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
        "task_loss_results_dir": args.task_loss_results_dir,
        "ephemeral_runtime": True,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
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
            bool(args.s3_export),
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
                f"--load-path {load_dir} has no state.pt; use a local checkpoint "
                f"or s3://{CHECKPOINT_BUCKET}/{MIXLAW_S3_ROOT}/{args.mix_name}/"
                "checkpoints/stepN"
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
        curr.save_checkpoint(ckpt0, 0, train_module, args, meta)
        _maybe_task_loss(args, ckpt0, 0, wb_run=wb_run)
        if rank == 0 and wb_run is not None:
            wandb_log_checkpoint(wb_run, ckpt0, step=0, tokens_seen=0)
        _require_durable_export(
            args,
            device=device,
            ckpt_dir=ckpt0,
            progress_dir=progress_dir,
            task_loss_dir=Path(args.task_loss_results_dir),
            what=f"durable export step0 ({args.mix_name})",
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
            curr.save_checkpoint(ckpt_dir, global_step, train_module, args, meta)
            _maybe_task_loss(args, ckpt_dir, global_step, wb_run=wb_run)
            if rank == 0 and wb_run is not None:
                wandb_log_checkpoint(
                    wb_run,
                    ckpt_dir,
                    step=global_step,
                    tokens_seen=global_step * tokens_per_step,
                )
            _require_durable_export(
                args,
                device=device,
                ckpt_dir=ckpt_dir,
                progress_dir=progress_dir,
                task_loss_dir=Path(args.task_loss_results_dir),
                what=f"durable export step{global_step} ({args.mix_name})",
            )

    _require_durable_export(
        args,
        device=device,
        checkpoints_root=save_folder,
        progress_dir=progress_dir,
        task_loss_dir=Path(args.task_loss_results_dir),
        what=f"final durable export ({args.mix_name})",
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
            bool(args.s3_export),
            args.wandb_mode,
        )


if __name__ == "__main__":
    main()
