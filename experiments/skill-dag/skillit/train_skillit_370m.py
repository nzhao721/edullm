#!/usr/bin/env python3
"""Skill-It dual-arm OLMo2-370M trainer (10B tokens, curriculum contract).

Fork of ``experiments/curriculum/train_curriculum_regmix_370m.py`` with:

  * ``DomainMixtureStream`` over an olmohq working pool (mid-run ``set_weights``)
  * Skill-It updates at steps 500 / 875 / 1250 / 1625 / 2000 (eta=0.2, w=1)
  * ``A_MODE=probe`` — fixed offline A from ``artifacts/A_offline.npy``
  * ``A_MODE=derivative`` — recompute A(r) from mixlaw Chinchilla fit each update
  * Persist full A + p_before/p_after to ``progress/skillit_updates*``
  * Optional S3 export via token_selection ``sync_to_s3`` (``S3_EXPORT``)

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
from typing import Any, Dict, List, Mapping, Optional, Set

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

_SKILLIT = Path(__file__).resolve().parent
_MIXLAW = _SKILLIT.parent / "mixlaw"
_CUR_ROOT = Path(__file__).resolve().parents[1] / "curriculum"
_TS_ROOT = Path(__file__).resolve().parents[1] / "token-selection"
for _p in (_SKILLIT, _MIXLAW, _CUR_ROOT, _TS_ROOT):
    if str(_p) not in sys.path:
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
from domain_stream import DomainMixtureStream  # noqa: E402
from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, DOMAINS, task_family  # noqa: E402
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
DEFAULT_A_OFFLINE = _SKILLIT / "artifacts" / "A_offline.npy"
DEFAULT_FIT_JSON = _MIXLAW / "mixlaw_fit_chinchilla.json"


def arm_s3_uri(arm_id: str, *parts: str) -> str:
    arm = str(arm_id).strip().strip("/")
    extra = "/".join(p.strip("/") for p in parts if str(p).strip())
    base = f"s3://{CHECKPOINT_BUCKET}/{SKILLIT_S3_ROOT}/{arm}"
    return f"{base}/{extra}" if extra else f"{base}/"


def _maybe_s3_sync(local: Path, remote: str) -> None:
    try:
        from token_selection.olmo_ext.s3_export import sync_to_s3

        sync_to_s3(local, remote)
    except Exception as exc:  # noqa: BLE001
        log.warning("S3 sync skipped (%s → %s): %s", local, remote, exc)


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
) -> None:
    """Append JSONL + per-step A/weights snapshots."""
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

    (updates_dir / f"step{step}_A.json").write_text(
        json.dumps(named_A, indent=2) + "\n", encoding="utf-8"
    )
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
    (updates_dir / f"step{step}_weights.json").write_text(
        json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8"
    )


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
        required=True,
        help="Olmohq working pool root (tokenized/<domain>/<domain>.npy)",
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
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--task-loss-results-dir", type=str, default=None)
    ap.add_argument("--task-loss-eval-script", type=str, default=None)
    ap.add_argument(
        "--task-loss-on-save",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()
    if args.name is None:
        args.name = args.arm_id
    # Compatibility with curriculum save_checkpoint meta fields.
    args.pacing = f"skillit:{args.a_mode}"
    args.difficulty_metric = None
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    if args.task_loss_results_dir is None:
        args.task_loss_results_dir = str(_SKILLIT / "task_loss_results" / args.arm_id)
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

    p = regmix_weight_vector(DOMAINS)
    stream = DomainMixtureStream(
        args.pool_dir,
        p,
        domains=DOMAINS,
        seq_len=SEQ_LEN,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

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
        "a_mode": args.a_mode,
        "method": f"skillit:{args.a_mode}",
        "run_id": args.name,
        "s3_prefix": f"{SKILLIT_S3_ROOT}/{args.arm_id}",
        "s3_uri": arm_s3_uri(args.arm_id),
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
        "a_offline": str(offline_path),
        "mixlaw_fit_json": str(args.mixlaw_fit_json),
        "seed": args.seed,
        "task_loss_on_save": bool(args.task_loss_on_save),
        "task_loss_results_dir": args.task_loss_results_dir,
        "initial_weights": {d: float(p[i]) for i, d in enumerate(DOMAINS)},
    }
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
            write_skillit_snapshot(
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
                )
            if is_distributed():
                # Broadcast new weights from rank 0.
                p_t = torch.tensor(stream.weights, dtype=torch.float64, device=device)
                dist.broadcast(p_t, src=0)
                stream.set_weights(p_t.detach().cpu().numpy())
                dist.barrier()

            if rank == 0:
                _maybe_s3_sync(ckpt_dir, arm_s3_uri(args.arm_id, "checkpoints", ckpt_dir.name))
                _maybe_s3_sync(progress_dir, arm_s3_uri(args.arm_id, "progress"))

    if rank == 0:
        log.info(
            "Training complete at step=%d world_size=%d arm=%s",
            total_steps,
            world_size,
            args.arm_id,
        )
        _maybe_s3_sync(save_folder, arm_s3_uri(args.arm_id, "checkpoints"))
        _maybe_s3_sync(progress_dir, arm_s3_uri(args.arm_id, "progress"))
        _maybe_s3_sync(
            Path(args.task_loss_results_dir),
            arm_s3_uri(args.arm_id, "task_loss_results"),
        )


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
    write_skillit_snapshot(
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
    log.info(
        "Skill-It update step=%d a_mode=%s p_after=%s",
        step,
        args.a_mode,
        {d: round(float(p_after[i]), 4) for i, d in enumerate(DOMAINS)},
    )


if __name__ == "__main__":
    main()
