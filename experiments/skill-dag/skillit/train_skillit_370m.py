#!/usr/bin/env python3
"""Skill-It dual-arm OLMo2-370M trainer (10B tokens, curriculum contract).

Fork of ``experiments/curriculum/train_curriculum_regmix_370m.py`` with:

  * ``DomainMixtureStream`` over a working pool staged from published
    ``s3://edullm-data/`` (``edullm_data.read.dataset_paths`` / ``resolve_latest``)
  * Skill-It updates at steps 500 / 875 / 1250 / 1625 / 2000 (eta=0.2, w=1)
  * ``A_MODE=probe`` — fixed offline A from ``artifacts/probes_full/A_offline.npy``
    (legacy fallback: ``artifacts/A_offline.npy``)
  * ``A_MODE=derivative`` — recompute A(r) from mixlaw Chinchilla fit each update
  * Persist full A + p_before/p_after to ``progress/skillit_updates*``
  * Fail-closed durable S3 export to ``s3://edullm-checkpoints/skillit/<arm>/``
    (default on; opt out only with ``--no-s3-export`` / ``S3_EXPORT=0``)
  * W&B online logging (project ``skillit``) for train metrics, Skill-It
    weight updates, task-loss evals, and checkpoint artifacts — SmolLM-style
    enablement via ``WANDB_API_KEY`` + ``--wandb-mode`` (orthogonal to S3)

Data path: pass ``--dataset-id`` (default ``pretrain/olmo-original-30b``) and a
``--pool-dir`` staging root. Rank 0 stages train shards from edullm-data on a
clean machine when the pool marker is missing; training never assumes FarmShare
scratch, laptop-local pools, or ``s3://edullm-datasets/`` already exist.

Ephemeral-runtime contract:
  - Permanent checkpoints fail-closed sync to
    ``s3://edullm-checkpoints/skillit/<arm>/`` (not ``curriculum/``).
  - W&B is an additional durable sink when enabled; missing ``wandb`` must
    never disable S3 export.
  - Never weakens edullm-data staging.

Does **not** submit AWS/FarmShare jobs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

_SKILLIT = Path(__file__).resolve().parent
# skillit/ → skill-dag/ → experiments/  (same layout as mixlaw/train_mixlaw_validation_370m.py)
_EXPERIMENTS = _SKILLIT.parent.parent
_MIXLAW = _SKILLIT.parent / "mixlaw"
_CUR_ROOT = _EXPERIMENTS / "curriculum"
_TS_ROOT = _EXPERIMENTS / "token-selection"
# Prefer mixlaw for shared DomainMixtureStream; FarmShare staging copies
# mixlaw/domain_stream.py into RUN_DIR alongside this script. When the trainer
# itself is copied to an ephemeral RUN_DIR, __file__-relative roots miss the
# repo — launch/submit must put curriculum + token-selection on PYTHONPATH.
for _p in (_MIXLAW, _SKILLIT, _CUR_ROOT, _TS_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
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
from domain_stream import DomainMixtureStream  # noqa: E402  # mixlaw (or staged copy)
from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, DOMAINS, task_family  # noqa: E402
from prepare_skillit_370m_data import (  # noqa: E402
    DEFAULT_DATASET_ID,
    SOURCE_MARKER,
    stage_working_pool,
)
from skillit_math import (  # noqa: E402
    ETA_DEFAULT,
    A_to_named_dict,
    load_fit_json,
    load_offline_A,
    losses_dict_to_vector,
    online_A_from_fit,
    regmix_weight_vector,
    skillit_update,
)
from wandb_logging import (  # noqa: E402
    add_wandb_args,
    init_wandb,
    wandb_log_checkpoint,
    wandb_log_eval,
    wandb_log_skillit_update,
    wandb_log_train,
    wandb_upload_existing,
)

log = logging.getLogger("train_skillit_370m")

SEQ_LEN = curr.SEQ_LEN
GLOBAL_BATCH_TOKENS = curr.GLOBAL_BATCH_TOKENS
MICROBATCH_TOKENS = curr.MICROBATCH_TOKENS
PEAK_LR = curr.PEAK_LR
DEFAULT_SEED = curr.DEFAULT_SEED
DEFAULT_LENGTH_TOKENS = curr.DEFAULT_LENGTH_TOKENS
CONFIG_NAME = "OLMo-2-370M-skillit"
CHECKPOINT_BUCKET = "edullm-checkpoints"
SKILLIT_S3_ROOT = "skillit"

SKILLIT_UPDATE_STEPS: tuple[int, ...] = (500, 875, 1250, 1625, 2000)
_A_OFFLINE_CANDIDATES = (
    _SKILLIT / "artifacts" / "probes_full" / "A_offline.npy",
    _SKILLIT / "artifacts" / "A_offline.npy",
)
DEFAULT_A_OFFLINE = next(
    (p for p in _A_OFFLINE_CANDIDATES if p.is_file()),
    _A_OFFLINE_CANDIDATES[0],
)
DEFAULT_FIT_JSON = _MIXLAW / "mixlaw_fit_chinchilla.json"
DEFAULT_RECIPE = _SKILLIT / "skillit_train_recipe.json"


def load_arm_weights(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise SystemExit(f"{path}: missing weights dict")
    out = {d: float(weights[d]) for d in DOMAINS}
    return out, payload


def arm_s3_prefix(arm_id: str) -> str:
    arm = str(arm_id).strip().strip("/")
    if not arm:
        raise ValueError("arm_id must be non-empty")
    return f"{SKILLIT_S3_ROOT}/{arm}"


def arm_s3_uri(arm_id: str, *parts: str) -> str:
    """Build ``s3://edullm-checkpoints/skillit/<arm>[/parts…]``."""
    prefix = arm_s3_prefix(arm_id)
    extra = "/".join(p.strip("/") for p in parts if str(p).strip())
    if extra:
        return f"s3://{CHECKPOINT_BUCKET}/{prefix}/{extra}"
    return f"s3://{CHECKPOINT_BUCKET}/{prefix}/"


def _redirect_curriculum_s3_to_skillit() -> None:
    """Point curriculum ``save_checkpoint`` S3 helpers at the skillit/ prefix.

    ``curr.save_checkpoint`` calls ``export_curriculum_*`` which default to
    ``curriculum/<arm>/``. Override the URI builders so fail-closed export
    lands under ``skillit/<arm>/`` without disabling S3.
    """
    curr.curriculum_s3_uri = arm_s3_uri  # type: ignore[attr-defined]
    curr.arm_s3_prefix = arm_s3_prefix  # type: ignore[attr-defined]
    log.info(
        "Curriculum S3 helpers redirected → s3://%s/%s/<arm>/",
        CHECKPOINT_BUCKET,
        SKILLIT_S3_ROOT,
    )


def curve_family_losses_from_task_loss(path: Path) -> Dict[str, float]:
    """Extract the 6 Skill-It curve-family bpb values from a task_loss JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    label_src = (
        payload.get("task_loss_bpb")
        or payload.get("labels")
        or payload.get("task_loss_labels")
        or {}
    )
    fam_src = payload.get("task_families") or payload.get("task_loss_families") or {}
    out: Dict[str, float] = {}
    if fam_src:
        for fam in CURVE_FAMILIES:
            if fam in fam_src:
                out[fam] = float(fam_src[fam])
    for label, value in label_src.items():
        if label not in CURVE_TASK_LOSS_LABELS:
            continue
        fam = task_family(label)
        if fam in CURVE_FAMILIES and fam not in out:
            out[fam] = float(value)
    missing = [f for f in CURVE_FAMILIES if f not in out]
    if missing:
        raise RuntimeError(f"{path}: missing curve families {missing}")
    return {f: out[f] for f in CURVE_FAMILIES}


def write_skillit_snapshot(
    progress_dir: Path,
    *,
    step: int,
    arm_id: str,
    a_mode: str,
    A: np.ndarray,
    p_before: np.ndarray,
    p_after: np.ndarray,
    losses: Optional[Mapping[str, float]],
    r_for_deriv: Optional[np.ndarray] = None,
    eta: float = ETA_DEFAULT,
    w: float = 1.0,
    note: str = "",
) -> Dict[str, Any]:
    """Append JSONL + per-step A/weights snapshots. Returns the record written."""
    updates_dir = progress_dir / "skillit_updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    named_A = A_to_named_dict(A, domains=DOMAINS, families=CURVE_FAMILIES)
    record: Dict[str, Any] = {
        "step": int(step),
        "arm_id": arm_id,
        "a_mode": a_mode,
        "domain_order": list(DOMAINS),
        "family_order": list(CURVE_FAMILIES),
        "A": named_A["A"],
        "rows_by_domain": named_A["rows_by_domain"],
        "p_before": {d: float(p_before[i]) for i, d in enumerate(DOMAINS)},
        "p_after": {d: float(p_after[i]) for i, d in enumerate(DOMAINS)},
        "eta": float(eta),
        "w": float(w),
    }
    if losses is not None:
        record["losses"] = {k: float(v) for k, v in losses.items()}
    if a_mode == "derivative" and r_for_deriv is not None:
        record["r"] = {d: float(r_for_deriv[i]) for i, d in enumerate(DOMAINS)}
    if note:
        record["note"] = note

    jsonl = progress_dir / "skillit_updates.jsonl"
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    a_path = updates_dir / f"step{step}_A.json"
    a_path.write_text(json.dumps(named_A, indent=2) + "\n", encoding="utf-8")
    weights_payload: Dict[str, Any] = {
        "step": int(step),
        "arm_id": arm_id,
        "a_mode": a_mode,
        "p_before": record["p_before"],
        "p_after": record["p_after"],
    }
    if losses is not None:
        weights_payload["losses"] = record["losses"]
    if note:
        weights_payload["note"] = note
    weights_path = updates_dir / f"step{step}_weights.json"
    weights_path.write_text(json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8")
    record["_weights_path"] = str(weights_path)
    record["_A_path"] = str(a_path)
    return record


def resolve_A(
    a_mode: str,
    p: np.ndarray,
    *,
    offline_A: Optional[np.ndarray],
    fit: Optional[dict],
) -> np.ndarray:
    if a_mode == "probe":
        if offline_A is None:
            raise SystemExit("probe A_MODE requires offline A")
        return offline_A
    if a_mode == "derivative":
        if fit is None:
            raise SystemExit("derivative A_MODE requires --mixlaw-fit-json")
        return online_A_from_fit(fit, p, domains=DOMAINS, families=CURVE_FAMILIES)
    raise SystemExit(f"unknown a_mode={a_mode!r}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=None)
    ap.add_argument("--arm-id", required=True, help="e.g. skillit-probe | skillit-deriv")
    ap.add_argument(
        "--a-mode",
        choices=("probe", "derivative"),
        required=True,
        help="probe = fixed offline A; derivative = online mixing-law A(r)",
    )
    ap.add_argument(
        "--pool-dir",
        type=str,
        default=None,
        help="Local staging root for edullm-data domain shards "
        "(tokens/<domain>/train-*.u32le.bin). Created/fetched when missing.",
    )
    ap.add_argument(
        "--dataset-id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Published edullm-data dataset id (resolved via resolve_latest)",
    )
    ap.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Pin edullm-data version (default: resolve_latest)",
    )
    ap.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=DEFAULT_LENGTH_TOKENS,
        help="Cap staged tokens per domain when fetching from edullm-data",
    )
    ap.add_argument(
        "--arm-weights-json",
        type=str,
        default=None,
        help="Per-arm recipe sidecar from prepare_skillit_370m_data.py (initial weights + stream seed)",
    )
    ap.add_argument(
        "--a-offline",
        type=str,
        default=str(DEFAULT_A_OFFLINE),
        help="Path to A_offline.npy (probe arm; also used for step-0 baseline on both)",
    )
    ap.add_argument(
        "--mixlaw-fit-json",
        type=str,
        default=str(DEFAULT_FIT_JSON),
        help="mixlaw_fit_chinchilla.json for derivative arm",
    )
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--length-tokens", type=int, default=DEFAULT_LENGTH_TOKENS)
    ap.add_argument(
        "--device-batch-size",
        type=int,
        default=MICROBATCH_TOKENS // SEQ_LEN,
    )
    ap.add_argument("--save-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    ap.add_argument("--num-workers", type=int, default=0, help="Unused (stream is in-process)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--load-path", type=str, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=24)
    ap.add_argument("--lr-alpha-f", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=ETA_DEFAULT)
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
        help=(
            "Fail-closed upload of checkpoints/progress/task_loss to "
            f"s3://{CHECKPOINT_BUCKET}/{SKILLIT_S3_ROOT}/<arm>/ (default on). "
            "Disable only for local smoke: --no-s3-export or S3_EXPORT=0 / SKIP_S3_UPLOAD=1."
        ),
    )
    add_wandb_args(ap)
    args = ap.parse_args()
    if args.name is None:
        args.name = args.arm_id
    if not args.pool_dir:
        args.pool_dir = str(Path(args.progress_dir).resolve().parent / "pool")
    # Compatibility with curriculum save_checkpoint meta fields.
    args.pacing = f"skillit:{args.a_mode}"
    args.difficulty_metric = None
    # Honor S3_EXPORT / SKIP_S3_UPLOAD env without disabling the fail-closed default
    # when the flag is left at its default True.
    if args.s3_export:
        from token_selection.olmo_ext.s3_export import s3_export_enabled

        if not s3_export_enabled(None):
            args.s3_export = False
    return args


def _ensure_pool(args: argparse.Namespace) -> dict[str, Any]:
    """Stage edullm-data domain shards into ``args.pool_dir`` if needed (rank 0)."""
    pool_dir = Path(args.pool_dir)
    marker = pool_dir / SOURCE_MARKER
    if get_rank() == 0:
        source = stage_working_pool(
            pool_dir=pool_dir,
            dataset_id=str(args.dataset_id),
            version=args.dataset_version,
            max_tokens_per_domain=int(args.max_tokens_per_domain)
            if args.max_tokens_per_domain
            else None,
            skip_if_complete=True,
        )
        log.info(
            "Working pool ready: dataset=%s/%s pool_dir=%s dtype=%s",
            source.get("dataset_id"),
            source.get("version"),
            pool_dir,
            source.get("dtype"),
        )
    else:
        source = {}
    if is_distributed():
        dist.barrier()
        if get_rank() != 0:
            if not marker.is_file():
                raise SystemExit(f"rank {get_rank()}: missing pool marker {marker}")
            source = json.loads(marker.read_text(encoding="utf-8"))
    return source


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        # Match curriculum: keep 20-label ladder evals under ephemeral progress/,
        # then sync with other arm artifacts to s3://edullm-checkpoints/skillit/.
        args.task_loss_results_dir = str(Path(args.progress_dir) / "task_loss_results")
    # Non-main ranks must not touch W&B (SmolLM-style single-writer).
    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) not in ("0", ""):
        os.environ["WANDB_MODE"] = "disabled"
    _redirect_curriculum_s3_to_skillit()
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

    arm_meta: dict[str, Any] = {}
    if args.arm_weights_json:
        weights_map, arm_meta = load_arm_weights(Path(args.arm_weights_json))
        p = np.array([weights_map[d] for d in DOMAINS], dtype=np.float64)
        stream_seed = int(
            arm_meta.get("stream_seed", arm_meta.get("recipe_seed", args.seed))
        )
        if arm_meta.get("a_mode") and arm_meta["a_mode"] != args.a_mode:
            raise SystemExit(
                f"arm_weights a_mode={arm_meta['a_mode']!r} != --a-mode {args.a_mode!r}"
            )
    else:
        p = regmix_weight_vector(DOMAINS)
        stream_seed = int(args.seed)

    seed_all(stream_seed + rank)

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
    update_set: Set[int] = set(SKILLIT_UPDATE_STEPS)
    lr = float(PEAK_LR)

    offline_path = Path(args.a_offline)
    offline_A = load_offline_A(offline_path) if offline_path.is_file() else None
    fit = None
    fit_path = Path(args.mixlaw_fit_json)
    if fit_path.is_file():
        fit = load_fit_json(fit_path)
    if args.a_mode == "derivative" and fit is None:
        raise SystemExit(f"missing mixlaw fit: {fit_path}")
    if args.a_mode == "probe" and offline_A is None:
        raise SystemExit(f"missing offline A: {offline_path}")

    progress_dir = Path(args.progress_dir)
    save_folder = Path(args.save_folder)
    if rank == 0:
        progress_dir.mkdir(parents=True, exist_ok=True)
        save_folder.mkdir(parents=True, exist_ok=True)
        Path(args.task_loss_results_dir).mkdir(parents=True, exist_ok=True)

    pool_source = _ensure_pool(args)
    stream_dtype = pool_source.get("dtype") or "uint32"
    stream = DomainMixtureStream(
        args.pool_dir,
        p,
        domains=DOMAINS,
        seq_len=SEQ_LEN,
        seed=stream_seed,
        rank=rank,
        world_size=world_size,
        dtype=stream_dtype,
    )

    meta = {
        "architecture": "olmo_core.TransformerConfig.olmo2_370M",
        "config_name": CONFIG_NAME,
        "arm": args.arm_id,
        "a_mode": args.a_mode,
        "method": f"skillit:{args.a_mode}",
        "run_id": args.name,
        "s3_prefix": arm_s3_prefix(args.arm_id),
        "s3_uri": arm_s3_uri(args.arm_id),
        "s3_export": bool(args.s3_export),
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
        "z_loss_multiplier": 1e-5,
        "max_grad_norm": 1.0,
        "compile": bool(args.compile),
        "save_interval": int(args.save_interval),
        "permanent_checkpoint_steps": ladder,
        "skillit_update_steps": list(SKILLIT_UPDATE_STEPS),
        "eta": float(args.eta),
        "pool_dir": str(args.pool_dir),
        "edullm_data": {
            "dataset_id": pool_source.get("dataset_id", args.dataset_id),
            "version": pool_source.get("version"),
            "dtype": stream_dtype,
            "bucket": "edullm-data",
        },
        "arm_weights_json": str(args.arm_weights_json) if args.arm_weights_json else None,
        "recipe": arm_meta.get("recipe"),
        "stream_seed": stream_seed,
        "a_offline": str(offline_path),
        "mixlaw_fit_json": str(args.mixlaw_fit_json),
        "seed": args.seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_results_dir": args.task_loss_results_dir,
        "initial_weights": {d: float(p[i]) for i, d in enumerate(DOMAINS)},
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
            "run_name": args.wandb_run_name or args.name,
            "group": args.wandb_group,
        },
    }
    wb_run = None
    logged_eval_steps: Set[int] = set()
    if rank == 0:
        (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        (progress_dir / "total_steps.txt").write_text(str(total_steps) + "\n")
        (progress_dir / "checkpoint_ladder.json").write_text(
            json.dumps({"steps": ladder, "interval": int(args.save_interval)}, indent=2) + "\n"
        )
        if total_steps == 2384 and 2375 in ladder_set:
            raise SystemExit("BUG: ladder for 2384 must omit 2375")

        log.info(
            "Plan: arm=%s a_mode=%s world=%d total=%d updates=%s",
            args.arm_id,
            args.a_mode,
            world_size,
            total_steps,
            list(SKILLIT_UPDATE_STEPS),
        )
        wb_run = init_wandb(
            args,
            meta,
            is_main=True,
            output_dir=progress_dir,
            default_name=str(args.name),
        )
        if wb_run is not None:
            meta["wandb"]["run_id"] = getattr(wb_run, "id", None)
            meta["wandb"]["url"] = getattr(wb_run, "url", None)
            (progress_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            if args.wandb_upload_existing:
                log.info("uploading existing checkpoints/evals to wandb...")
                wandb_upload_existing(
                    wb_run,
                    save_folder=save_folder,
                    progress_dir=progress_dir,
                    task_loss_results_dir=Path(args.task_loss_results_dir),
                )

    train_module = curr.build_train_module(
        lr=lr,
        lr_warmup_steps=int(args.lr_warmup_steps),
        alpha_f=float(args.lr_alpha_f),
        compile_model=bool(args.compile),
        rank_microbatch_tokens=rank_micro_tokens,
    )
    books = curr._Bookkeeping(
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
        load_dir = Path(args.load_path) if args.load_path else curr.find_latest_checkpoint(save_folder)
        if load_dir is not None:
            start_step = curr.load_checkpoint(load_dir, train_module)

    # Restore domain weights after the last Skill-It update at or before start_step.
    if start_step > 0:
        restored = _restore_weights_from_jsonl(progress_dir, start_step)
        if restored is not None:
            stream.set_weights(restored)
            if rank == 0:
                log.info("Restored Skill-It weights for resume at step=%d", start_step)
    elif rank == 0:
        # Step-0 baseline once at train start (no weight change).
        if not _has_snapshot_step(progress_dir / "skillit_updates.jsonl", 0):
            A0 = resolve_A(args.a_mode, p, offline_A=offline_A, fit=fit)
            record0 = write_skillit_snapshot(
                progress_dir,
                step=0,
                arm_id=args.arm_id,
                a_mode=args.a_mode,
                A=A0,
                p_before=p,
                p_after=p,
                losses=None,
                r_for_deriv=p if args.a_mode == "derivative" else None,
                eta=ETA_DEFAULT,
                w=1.0,
                note="baseline RegMix weights; no Skill-It update yet",
            )
            wandb_log_skillit_update(
                wb_run,
                record0,
                step=0,
                snapshot_path=Path(record0["_weights_path"]),
                a_snapshot_path=Path(record0["_A_path"]),
            )

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
        _maybe_task_loss(args, ckpt0, 0, async_=True)
        if rank == 0:
            wandb_log_checkpoint(
                wb_run,
                ckpt0,
                step=0,
                tokens_seen=0,
                arm_id=args.arm_id,
            )
            _maybe_log_task_loss_wandb(
                wb_run,
                Path(args.task_loss_results_dir),
                step=0,
                logged=logged_eval_steps,
            )
        if is_distributed():
            dist.barrier()

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
            if rank == 0:
                log.info(
                    "step=%d/%d a_mode=%s tok/s=%.0f (avg=%.0f) world=%d",
                    global_step,
                    total_steps,
                    args.a_mode,
                    tok_s,
                    tok_s_avg,
                    world_size,
                )
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "a_mode": args.a_mode,
                                "tok_per_s": tok_s,
                                "tok_per_s_avg": tok_s_avg,
                                "weights": stream.weights_dict(),
                            }
                        )
                        + "\n"
                    )
                (progress_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "step": global_step,
                            "total_steps": total_steps,
                            "a_mode": args.a_mode,
                            "world_size": world_size,
                            "tok_per_s": tok_s,
                            "pct": round(100.0 * global_step / total_steps, 4),
                            "weights": stream.weights_dict(),
                        }
                    )
                    + "\n"
                )
                wandb_log_train(
                    wb_run,
                    step=global_step,
                    tok_per_s=tok_s,
                    tok_per_s_avg=tok_s_avg,
                    tokens_seen=global_step * tokens_per_step,
                    weights=stream.weights_dict(),
                    a_mode=args.a_mode,
                )

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            ckpt_dir = save_folder / f"step{global_step}"
            curr.save_checkpoint(ckpt_dir, global_step, train_module, args, meta)
            # Skill-It update steps need sync 20-label eval so losses are ready.
            need_sync = global_step in update_set
            _maybe_task_loss(args, ckpt_dir, global_step, async_=not need_sync)
            if rank == 0 and need_sync:
                _apply_skillit_update(
                    args,
                    progress_dir=progress_dir,
                    stream=stream,
                    step=global_step,
                    offline_A=offline_A,
                    fit=fit,
                    eta=ETA_DEFAULT,
                    wb_run=wb_run,
                    logged_eval_steps=logged_eval_steps,
                )
            if is_distributed():
                # Broadcast new weights from rank 0.
                p_t = torch.tensor(stream.weights, dtype=torch.float64, device=device)
                dist.broadcast(p_t, src=0)
                stream.set_weights(p_t.detach().cpu().numpy())
                dist.barrier()

            if rank == 0:
                wandb_log_checkpoint(
                    wb_run,
                    ckpt_dir,
                    step=global_step,
                    tokens_seen=global_step * tokens_per_step,
                    arm_id=args.arm_id,
                )
                if not need_sync:
                    _maybe_log_task_loss_wandb(
                        wb_run,
                        Path(args.task_loss_results_dir),
                        step=global_step,
                        logged=logged_eval_steps,
                    )
                # Durable S3 export is fail-closed inside curr.save_checkpoint
                # (URI redirected to skillit/ via _redirect_curriculum_s3_to_skillit).

    final_ok = True
    final_err = (
        "final durable S3 export failed "
        "(use S3_EXPORT=0 / --no-s3-export only for local smoke)"
    )
    if rank == 0:
        try:
            log.info(
                "Training complete at step=%d world_size=%d arm=%s durable=%s wandb=%s",
                total_steps,
                world_size,
                args.arm_id,
                arm_s3_uri(args.arm_id) if args.s3_export else "s3-off",
                getattr(wb_run, "url", None) if wb_run is not None else "off",
            )
            # Sweep async evals that finished after their ladder step.
            tl_dir = Path(args.task_loss_results_dir)
            for eval_path in sorted(tl_dir.glob("step*_task_loss.json")):
                try:
                    estep = int(eval_path.name.split("_")[0].replace("step", ""))
                except ValueError:
                    continue
                _maybe_log_task_loss_wandb(wb_run, tl_dir, step=estep, logged=logged_eval_steps)
            # Fail-closed final tree sync under skillit/ (patched curriculum helpers).
            curr.export_curriculum_artifacts(
                args.arm_id,
                checkpoints_root=save_folder,
                progress_dir=progress_dir,
                task_loss_dir=tl_dir,
                enabled=bool(args.s3_export),
            )
            if wb_run is not None:
                try:
                    wb_run.finish()
                except Exception as exc:  # noqa: BLE001
                    log.warning("wandb.finish failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
            final_ok = False
            final_err = f"final durable S3 export / teardown failed: {exc}"
            log.error("%s", final_err)
    curr._abort_all_ranks(final_err, ok=final_ok)


def _restore_weights_from_jsonl(progress_dir: Path, start_step: int) -> Optional[np.ndarray]:
    path = progress_dir / "skillit_updates.jsonl"
    if not path.is_file():
        return None
    best: Optional[dict] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if int(rec["step"]) <= int(start_step):
            best = rec
    if best is None:
        return None
    order = best.get("domain_order") or list(DOMAINS)
    p_after = best["p_after"]
    return np.array([float(p_after[d]) for d in order], dtype=np.float64)


def _has_snapshot_step(path: Path, step: int) -> bool:
    """Return whether a valid Skill-It update record already exists for ``step``."""
    if not path.is_file():
        return False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid Skill-It update JSON") from exc
        if int(record.get("step", -1)) == int(step):
            return True
    return False


def _maybe_log_task_loss_wandb(
    wb_run: object | None,
    results_dir: Path,
    *,
    step: int,
    logged: Set[int],
) -> None:
    if wb_run is None or int(step) in logged:
        return
    path = results_dir / f"step{step}_task_loss.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    wandb_log_eval(wb_run, payload, step=step, eval_path=path)
    logged.add(int(step))


def _maybe_task_loss(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    *,
    async_: bool,
) -> None:
    if get_rank() != 0:
        return
    results_dir = Path(args.task_loss_results_dir)
    trigger_task_loss_eval(
        ckpt_dir,
        run_name=f"{args.arm_id}-step{step}",
        out_path=results_dir / f"step{step}_task_loss.json",
        eval_script=args.task_loss_eval_script,
        enabled=None if args.task_loss_on_save else False,
        async_=async_,
    )


def _apply_skillit_update(
    args: argparse.Namespace,
    *,
    progress_dir: Path,
    stream: DomainMixtureStream,
    step: int,
    offline_A: Optional[np.ndarray],
    fit: Optional[dict],
    eta: float,
    wb_run: object | None = None,
    logged_eval_steps: Optional[Set[int]] = None,
) -> None:
    results_path = Path(args.task_loss_results_dir) / f"step{step}_task_loss.json"
    if not results_path.is_file():
        log.warning(
            "Skill-It update at step %d skipped: missing %s "
            "(ensure TASK_LOSS_EVAL_SCRIPT works and sync eval succeeded)",
            step,
            results_path,
        )
        return
    if logged_eval_steps is None:
        logged_eval_steps = set()
    _maybe_log_task_loss_wandb(
        wb_run,
        Path(args.task_loss_results_dir),
        step=step,
        logged=logged_eval_steps,
    )
    losses = curve_family_losses_from_task_loss(results_path)
    p_before = stream.weights
    A = resolve_A(
        args.a_mode,
        p_before,
        offline_A=offline_A,
        fit=fit,
    )
    L = losses_dict_to_vector(losses, CURVE_FAMILIES)
    p_after = skillit_update(A, L, eta=eta, w=1.0)
    stream.set_weights(p_after)
    record = write_skillit_snapshot(
        progress_dir,
        step=step,
        arm_id=args.arm_id,
        a_mode=args.a_mode,
        A=A,
        p_before=p_before,
        p_after=p_after,
        losses=losses,
        r_for_deriv=p_before if args.a_mode == "derivative" else None,
        eta=eta,
        w=1.0,
    )
    wandb_log_skillit_update(
        wb_run,
        record,
        step=step,
        snapshot_path=Path(record["_weights_path"]),
        a_snapshot_path=Path(record["_A_path"]),
    )
    log.info(
        "Skill-It update step=%d a_mode=%s p_after=%s",
        step,
        args.a_mode,
        {d: round(float(p_after[i]), 4) for i, d in enumerate(DOMAINS)},
    )


if __name__ == "__main__":
    main()
