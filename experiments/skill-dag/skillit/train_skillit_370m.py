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
  * Runtime artifacts remain on scratch; no checkpoint, progress, or eval writes
    go to S3.
  * W&B online logging (project ``skillit``) is the production durable sink for
    train metrics, Skill-It updates, evals, and checkpoints. Checkpoint uploads
    are synchronous and fail closed before training advances.

Data path is pinned to ``pretrain/olmo-127b/v1``. Pass a
``--pool-dir`` staging root. Rank 0 stages train shards from edullm-data on a
clean machine when the pool marker is missing; training never assumes FarmShare
scratch, laptop-local pools, or ``s3://edullm-datasets/`` already exist.

Ephemeral-runtime contract:
  - Checkpoints, progress, and evals remain on runtime scratch.
  - Production requires online W&B; local-only smoke runs require an explicit
    ``--allow-local-only``.
  - S3 is read only for edullm-data staging and an explicitly selected legacy
    resume bootstrap at run start.

Does **not** submit AWS/FarmShare jobs.
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
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set

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
from token_selection.olmo_ext.task_loss_hook import resolve_eval_script

import train_curriculum_regmix_370m as curr  # noqa: E402
from domain_stream import DomainMixtureStream  # noqa: E402  # mixlaw (or staged copy)
from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, DOMAINS, task_family  # noqa: E402
from prepare_skillit_370m_data import (  # noqa: E402
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_VERSION,
    stage_working_pool,
    validate_pool_source,
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
    wandb_log_runtime_artifacts,
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
LEGACY_RESUME_S3_ROOT = "s3://edullm-checkpoints/skillit"

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


def _import_task_loss_eval_module(eval_script: Optional[str] = None) -> Any:
    """Import the shared step-1 evaluator by resolved file path."""
    script = resolve_eval_script(eval_script)
    if script is None:
        raise FileNotFoundError(
            "shared task-loss evaluator not found; set TASK_LOSS_EVAL_SCRIPT "
            "or --task-loss-eval-script"
        )
    spec = importlib.util.spec_from_file_location("eval_task_loss_olmo_core", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import shared task-loss evaluator from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "pause_eval_reload_distributed"):
        raise ImportError(
            f"{script} lacks shared pause_eval_reload_distributed helper"
        )
    return module


def _reset_olmo_world_mesh() -> None:
    """Allow rebuilding the train module after releasing FSDP for evaluation."""
    try:
        import olmo_core.distributed.parallel as parallel

        if getattr(parallel, "_WORLD_MESH", None) is not None:
            parallel._WORLD_MESH = None  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not reset olmo-core world mesh: %s", exc)


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
        default=DEFAULT_DATASET_VERSION,
        help=f"Pinned edullm-data version (required: {DEFAULT_DATASET_VERSION})",
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
    ap.add_argument("--device-eval-batch-size", type=int, default=4)
    ap.add_argument(
        "--task-loss-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--allow-local-only",
        action="store_true",
        help="Explicit local smoke mode; permits W&B offline/disabled.",
    )
    add_wandb_args(ap)
    args = ap.parse_args()
    if args.name is None:
        args.name = args.arm_id
    if bool(args.fresh) == bool(args.load_path):
        ap.error("choose exactly one resume mode: --fresh or --load-path <local|s3://...>")
    if args.dataset_id != DEFAULT_DATASET_ID or args.dataset_version != DEFAULT_DATASET_VERSION:
        ap.error(
            "SkillIt source is pinned to "
            f"{DEFAULT_DATASET_ID}/{DEFAULT_DATASET_VERSION}"
        )
    if not args.pool_dir:
        args.pool_dir = str(Path(args.progress_dir).resolve().parent / "pool")
    # Compatibility with curriculum save_checkpoint meta fields.
    args.pacing = f"skillit:{args.a_mode}"
    args.difficulty_metric = None
    if not args.allow_local_only and args.wandb_mode != "online":
        ap.error(
            "production runs require --wandb-mode online; "
            "use --allow-local-only only for local smoke"
        )
    return args


def _ensure_pool(args: argparse.Namespace) -> dict[str, Any]:
    """Stage edullm-data domain shards into ``args.pool_dir`` if needed (rank 0)."""
    pool_dir = Path(args.pool_dir)
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
    source = validate_pool_source(
        pool_dir,
        dataset_id=DEFAULT_DATASET_ID,
        version=DEFAULT_DATASET_VERSION,
        require_370m_layout=True,
    )
    return source


def _stage_resume_path(
    args: argparse.Namespace,
    *,
    save_folder: Path,
    progress_dir: Path,
) -> Path:
    """Stage an explicit local/S3 bootstrap checkpoint at run start."""
    raw = str(args.load_path).strip()
    if not raw.startswith("s3://"):
        path = Path(raw)
        if not (path / "state.pt").is_file():
            raise SystemExit(f"--load-path {path} is missing state.pt")
        return path

    expected_root = f"{LEGACY_RESUME_S3_ROOT}/{args.arm_id}"
    checkpoint_root = f"{expected_root}/checkpoints/"
    if not raw.startswith(checkpoint_root):
        raise SystemExit(
            f"SkillIt S3 --load-path must be under {checkpoint_root}; got {raw}"
        )
    step_name = raw.rstrip("/").rsplit("/", 1)[-1]
    if not step_name.startswith("step"):
        raise SystemExit(f"SkillIt S3 --load-path must end in stepN; got {raw}")

    dest = save_folder / step_name
    ok = True
    error = ""
    if get_rank() == 0:
        try:
            # This is a read-only bootstrap at run start; SkillIt never writes
            # checkpoints or progress back to S3.
            dest = curr.stage_load_path(
                raw,
                save_folder=save_folder,
                s3_export=True,
            )
            # A checkpoint at an update step is pre-update; restoring the durable
            # progress history is required to recover the post-update stream weights.
            from token_selection.olmo_ext.s3_export import sync_from_s3

            sync_from_s3(
                f"{expected_root}/progress/",
                progress_dir,
                enabled=True,
                raise_on_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            error = f"failed to stage SkillIt resume artifacts from {raw}: {exc}"
    curr._abort_all_ranks(error or "SkillIt S3 resume staging failed", ok=ok)
    if is_distributed():
        dist.barrier()
    if not (dest / "state.pt").is_file():
        raise SystemExit(f"staged --load-path {dest} is missing state.pt")
    return dest


def _validate_checkpoint_source(
    checkpoint_dir: Path,
    *,
    arm_id: str,
    a_mode: str,
) -> None:
    """Reject resumes from another source, arm, or SkillIt update mode."""
    payload = torch.load(
        checkpoint_dir / "state.pt",
        map_location="cpu",
        weights_only=False,
    )
    meta = payload.get("meta") or {}
    source = meta.get("edullm_data") or {}
    got_source = (source.get("dataset_id"), source.get("version"))
    expected_source = (DEFAULT_DATASET_ID, DEFAULT_DATASET_VERSION)
    if got_source != expected_source:
        raise SystemExit(
            f"{checkpoint_dir}: checkpoint source {got_source!r} does not match "
            f"pinned source {expected_source!r}"
        )
    if str(meta.get("arm")) != arm_id:
        raise SystemExit(
            f"{checkpoint_dir}: checkpoint arm={meta.get('arm')!r} != {arm_id!r}"
        )
    if str(meta.get("a_mode")) != a_mode:
        raise SystemExit(
            f"{checkpoint_dir}: checkpoint a_mode={meta.get('a_mode')!r} != {a_mode!r}"
        )


def _validate_task_loss_payload(
    payload: Mapping[str, Any],
    *,
    step: int,
    path: Path,
) -> None:
    """Require a fresh, complete 20-label result for the exact update step."""
    if int(payload.get("step", -1)) != int(step):
        raise RuntimeError(
            f"{path}: stale task-loss payload step={payload.get('step')!r}; expected {step}"
        )
    labels = payload.get("labels") or {}
    if not payload.get("suite_complete") or int(payload.get("raw_label_count", 0)) != 20:
        raise RuntimeError(f"{path}: task-loss suite is not contract-complete")
    if not isinstance(labels, dict) or len(labels) != 20:
        raise RuntimeError(f"{path}: expected exactly 20 raw task-loss labels")


def _pause_eval_reload(
    args: argparse.Namespace,
    ckpt_dir: Path,
    step: int,
    *,
    books: Any,
    lr: float,
    rank_micro_tokens: int,
    tokens_per_step: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Use the shared all-rank pause/eval/reload helper in strict mode."""
    eval_mod = _import_task_loss_eval_module(args.task_loss_eval_script)
    out_path = Path(args.task_loss_results_dir) / f"step{step}_task_loss.json"
    if get_rank() == 0 and out_path.exists():
        out_path.unlink()
    if is_distributed():
        dist.barrier()

    def release_train_state() -> None:
        for attr in ("trainer", "_trainer", "train_module", "_train_module"):
            if hasattr(books, attr):
                try:
                    setattr(books, attr, None)
                except Exception:
                    pass

    def reload_train_state() -> Any:
        _reset_olmo_world_mesh()
        module = curr.build_train_module(
            lr=lr,
            lr_warmup_steps=int(args.lr_warmup_steps),
            alpha_f=float(args.lr_alpha_f),
            compile_model=bool(args.compile),
            rank_microbatch_tokens=rank_micro_tokens,
        )
        module._attach_trainer(books)  # type: ignore[arg-type]
        loaded = curr.load_checkpoint(ckpt_dir, module)
        books.global_step = int(loaded)
        books.global_train_tokens_seen = int(loaded) * int(tokens_per_step)
        return module

    module, payload = eval_mod.pause_eval_reload_distributed(
        ckpt_dir,
        out_path,
        f"{args.arm_id}-step{step}",
        release_train_state=release_train_state,
        reload_train_state=reload_train_state,
        base_config=Path(os.environ["LADDER_BASE_CONFIG"]),
        device_eval_batch_size=int(args.device_eval_batch_size),
        strict=True,
    )
    if payload is None:
        raise RuntimeError(f"shared evaluator returned no payload for step {step}")
    _validate_task_loss_payload(payload, step=step, path=out_path)
    return module, payload


def _wandb_upload_or_abort(
    args: argparse.Namespace,
    wb_run: object | None,
    upload: Callable[[], bool],
    *,
    what: str,
) -> None:
    """Run a rank-0 W&B upload and fail all ranks for production failures."""
    ok = True
    error = ""
    if get_rank() == 0:
        try:
            uploaded = bool(upload()) if wb_run is not None else False
            if not uploaded and not bool(args.allow_local_only):
                raise RuntimeError("online W&B run is unavailable")
        except Exception as exc:  # noqa: BLE001
            if not bool(args.allow_local_only):
                ok = False
                error = f"{what} failed: {exc}"
            else:
                log.warning("%s failed in local-only smoke mode: %s", what, exc)
    curr._abort_all_ranks(error or f"{what} failed", ok=ok)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        # Keep 20-label ladder evals under runtime scratch; W&B is the only
        # production artifact sink.
        args.task_loss_results_dir = str(Path(args.progress_dir) / "task_loss_results")
    # Non-main ranks must not touch W&B (SmolLM-style single-writer).
    if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) not in ("0", ""):
        os.environ["WANDB_MODE"] = "disabled"
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
    if arm_meta:
        arm_source = arm_meta.get("edullm_data") or {}
        arm_identity = (arm_source.get("dataset_id"), arm_source.get("version"))
        pool_identity = (pool_source.get("dataset_id"), pool_source.get("version"))
        if arm_identity != pool_identity:
            raise SystemExit(
                f"{args.arm_weights_json}: source {arm_identity!r} does not match "
                f"staged pool {pool_identity!r}"
            )
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
        "artifact_backend": "wandb",
        "artifact_storage": "runtime_scratch",
        "allow_local_only": bool(args.allow_local_only),
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
            "uri": (
                f"s3://edullm-data/{pool_source.get('dataset_id')}/"
                f"{pool_source.get('version')}/"
            ),
            "identity": (
                f"{pool_source.get('dataset_id')}@{pool_source.get('version')}"
            ),
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
    wandb_init_ok = True
    wandb_init_error = ""
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
        try:
            wb_run = init_wandb(
                args,
                meta,
                is_main=True,
                output_dir=progress_dir,
                default_name=str(args.name),
            )
            if wb_run is None and not bool(args.allow_local_only):
                raise RuntimeError("online W&B initialization returned no run")
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
        except Exception as exc:  # noqa: BLE001
            if not bool(args.allow_local_only):
                wandb_init_ok = False
                wandb_init_error = f"production W&B initialization failed: {exc}"
            else:
                log.warning("W&B initialization failed in local-only smoke mode: %s", exc)
            wb_run = None
    curr._abort_all_ranks(
        wandb_init_error or "production W&B initialization failed",
        ok=wandb_init_ok,
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
            log.info("--fresh: starting from scratch; local checkpoints are ignored")
    else:
        load_dir = _stage_resume_path(
            args,
            save_folder=save_folder,
            progress_dir=progress_dir,
        )
        _validate_checkpoint_source(
            load_dir,
            arm_id=args.arm_id,
            a_mode=args.a_mode,
        )
        start_step = curr.load_checkpoint(load_dir, train_module)

    # Restore domain weights after the last Skill-It update at or before start_step.
    if start_step > 0:
        expected_update = max(
            (step for step in SKILLIT_UPDATE_STEPS if step <= start_step),
            default=None,
        )
        restored = _restore_weights_from_jsonl(
            progress_dir,
            start_step,
            required_update_step=expected_update,
        )
        if expected_update is not None and restored is None:
            raise SystemExit(
                f"resume at step {start_step} requires durable post-update weights "
                f"from step {expected_update}; skillit_updates.jsonl is missing/incomplete"
            )
        if restored is not None:
            stream.set_weights(restored)
        if rank == 0:
            log.info(
                "Restored explicit SkillIt resume at step=%d (last_update=%s)",
                start_step,
                expected_update,
            )
    else:
        baseline_ok = True
        baseline_error = ""
        if rank == 0:
            # Step-0 baseline once at train start (no weight change).
            try:
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
                    if wb_run is not None:
                        wandb_log_skillit_update(
                            wb_run,
                            record0,
                            step=0,
                            snapshot_path=Path(record0["_weights_path"]),
                            a_snapshot_path=Path(record0["_A_path"]),
                        )
            except Exception as exc:  # noqa: BLE001
                if not bool(args.allow_local_only):
                    baseline_ok = False
                    baseline_error = f"step-0 W&B Skill-It artifact upload failed: {exc}"
                else:
                    log.warning("step-0 artifact upload failed in local-only mode: %s", exc)
        curr._abort_all_ranks(
            baseline_error or "step-0 W&B Skill-It artifact upload failed",
            ok=baseline_ok,
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
        if bool(args.task_loss_on_save):
            del train_module
            train_module, payload0 = _pause_eval_reload(
                args,
                ckpt0,
                0,
                books=books,
                lr=lr,
                rank_micro_tokens=rank_micro_tokens,
                tokens_per_step=tokens_per_step,
            )
        def upload_step0_bundle() -> bool:
            if bool(args.task_loss_on_save):
                eval0 = Path(args.task_loss_results_dir) / "step0_task_loss.json"
                wandb_log_eval(wb_run, payload0, step=0, eval_path=eval0)
                logged_eval_steps.add(0)
            return wandb_log_checkpoint(
                wb_run,
                ckpt0,
                step=0,
                tokens_seen=0,
                arm_id=args.arm_id,
            )

        _wandb_upload_or_abort(
            args,
            wb_run,
            upload_step0_bundle,
            what="W&B checkpoint upload for step 0",
        )
        if rank == 0:
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
            need_sync = global_step in update_set
            eval_payload: Optional[Mapping[str, Any]] = None
            if bool(args.task_loss_on_save):
                del train_module
                train_module, eval_payload = _pause_eval_reload(
                    args,
                    ckpt_dir,
                    global_step,
                    books=books,
                    lr=lr,
                    rank_micro_tokens=rank_micro_tokens,
                    tokens_per_step=tokens_per_step,
                )
            elif need_sync:
                raise SystemExit(
                    f"Skill-It update step {global_step} requires strict task-loss eval; "
                    "--no-task-loss-on-save is only valid for smoke runs that stop "
                    "before the first update"
                )
            update_ok = True
            update_err = ""
            if rank == 0 and need_sync:
                try:
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
                except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
                    update_ok = False
                    update_err = (
                        f"Skill-It update at step {global_step} failed: {exc}"
                    )
                    log.error("%s", update_err)
            if need_sync:
                # Prefer crash over continuing with stale RegMix domain weights.
                if is_distributed():
                    curr._abort_all_ranks(
                        update_err or "Skill-It update failed",
                        ok=update_ok,
                    )
                elif not update_ok:
                    raise SystemExit(update_err or "Skill-It update failed")
            if is_distributed():
                # Broadcast new weights from rank 0.
                p_t = torch.tensor(stream.weights, dtype=torch.float64, device=device)
                dist.broadcast(p_t, src=0)
                stream.set_weights(p_t.detach().cpu().numpy())
                dist.barrier()

            def upload_checkpoint_bundle() -> bool:
                if eval_payload is not None and global_step not in logged_eval_steps:
                    eval_path = (
                        Path(args.task_loss_results_dir)
                        / f"step{global_step}_task_loss.json"
                    )
                    wandb_log_eval(
                        wb_run,
                        dict(eval_payload),
                        step=global_step,
                        eval_path=eval_path,
                    )
                    logged_eval_steps.add(global_step)
                return wandb_log_checkpoint(
                    wb_run,
                    ckpt_dir,
                    step=global_step,
                    tokens_seen=global_step * tokens_per_step,
                    arm_id=args.arm_id,
                )

            _wandb_upload_or_abort(
                args,
                wb_run,
                upload_checkpoint_bundle,
                what=f"W&B checkpoint upload for step {global_step}",
            )
            if rank == 0:
                if not need_sync:
                    _maybe_log_task_loss_wandb(
                        wb_run,
                        Path(args.task_loss_results_dir),
                        step=global_step,
                        logged=logged_eval_steps,
                    )
    final_ok = True
    final_err = "final W&B runtime artifact upload failed"
    if rank == 0:
        try:
            log.info(
                "Training complete at step=%d world_size=%d arm=%s artifacts=%s wandb=%s",
                total_steps,
                world_size,
                args.arm_id,
                "wandb" if wb_run is not None else "runtime-scratch-only",
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
            uploaded = wandb_log_runtime_artifacts(
                wb_run,
                progress_dir=progress_dir,
                task_loss_results_dir=tl_dir,
            )
            if not uploaded and not bool(args.allow_local_only):
                raise RuntimeError("online W&B run unavailable for final artifact upload")
            if wb_run is not None:
                try:
                    wb_run.finish()
                except Exception as exc:  # noqa: BLE001
                    log.warning("wandb.finish failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
            final_ok = False
            final_err = f"final W&B artifact upload / teardown failed: {exc}"
            log.error("%s", final_err)
    curr._abort_all_ranks(final_err, ok=final_ok)


def _restore_weights_from_jsonl(
    progress_dir: Path,
    start_step: int,
    *,
    required_update_step: Optional[int] = None,
) -> Optional[np.ndarray]:
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
    if required_update_step is not None and int(best.get("step", -1)) != int(
        required_update_step
    ):
        raise RuntimeError(
            f"{path}: latest snapshot through resume step {start_step} is "
            f"step {best.get('step')!r}; required post-update step {required_update_step}"
        )
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
        raise RuntimeError(
            f"Skill-It update at step {step} required {results_path}; "
            "task-loss eval must succeed before domain-weight updates "
            "(refusing to continue with stale RegMix weights). "
            "Ensure TASK_LOSS_EVAL_SCRIPT works and sync eval succeeded."
        )
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    _validate_task_loss_payload(payload, step=step, path=results_path)
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
