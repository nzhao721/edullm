#!/usr/bin/env python3
"""Replay a curriculum training run to a new W&B run with clean eval ladder + train history."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r"step(\d+)_task_loss\.json$", re.I)
SKIP_KEYS = {"_step", "_timestamp", "_runtime", "_wandb", "eval_checkpoint"}
TRAIN_PREFIXES = ("train/", "throughput/", "optim/", "checkpoint/")
BACKFILL_STEP_MIN = 2205
BACKFILL_STEP_MAX = 2222
EMA_WANDB_STEP = 2385


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="eduLLM")
    p.add_argument("--project", default="curriculum")
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--through-step", type=int, default=2384)
    p.add_argument("--task-loss-dir", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--run-id-out", type=Path, required=True)
    p.add_argument("--prev-id-out", type=Path, default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--upload-artifacts", action="store_true")
    return p.parse_args()


def is_train_metric(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in TRAIN_PREFIXES)


def train_metrics_from_row(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        if key in SKIP_KEYS or key.startswith("_"):
            continue
        if not is_train_metric(key):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        out[key] = float(value)
    return out


def is_backfill_pollution(step: int, row: dict[str, Any]) -> bool:
    if not (BACKFILL_STEP_MIN <= step <= BACKFILL_STEP_MAX):
        return False
    has_eval = any(str(k).startswith("eval/") for k in row)
    has_train = any(is_train_metric(str(k)) for k in row)
    return has_eval and not has_train


def load_eval_from_json(task_loss_dir: Path, through_step: int) -> dict[int, dict[str, float]]:
    from production_contract.task_loss import task_loss_metrics

    by_step: dict[int, dict[str, float]] = {}
    for path in sorted(task_loss_dir.glob("step*_task_loss.json")):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > through_step and step != EMA_WANDB_STEP:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_step[step] = task_loss_metrics(payload)
    return by_step


def main() -> None:
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY unset")

    import wandb

    api = wandb.Api()
    src_path = f"{args.entity}/{args.project}/{args.source_run_id}"
    old = api.run(src_path)
    print(f"source={src_path} name={old.name}", flush=True)

    train_by_step: dict[int, dict[str, float]] = {}
    n_rows = 0
    skipped_pollution = 0
    for row in old.scan_history(page_size=5000):
        n_rows += 1
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        if step > args.through_step:
            continue
        if is_backfill_pollution(step, row):
            skipped_pollution += 1
            continue
        metrics = train_metrics_from_row(row)
        if metrics:
            train_by_step.setdefault(step, {}).update(metrics)

    eval_by_step = load_eval_from_json(args.task_loss_dir, args.through_step)
    print(
        f"history_rows={n_rows} train_steps={len(train_by_step)} "
        f"eval_steps={sorted(eval_by_step)} skipped_pollution={skipped_pollution}",
        flush=True,
    )
    if 2000 not in eval_by_step:
        raise SystemExit("step 2000 eval missing from local task_loss JSONs")
    if EMA_WANDB_STEP not in eval_by_step:
        raise SystemExit(f"step {EMA_WANDB_STEP} EMA eval missing; run post-hoc EMA first")

    try:
        tags = list(old.tags or [])
        for tag in ("superseded", "dirty-wandb-history"):
            if tag not in tags:
                tags.append(tag)
        old.tags = tags
        old.notes = (
            (old.notes or "")
            + f"\nSuperseded by clean replay: {args.run_name}"
        ).strip()
        old.update()
        print(f"tagged old run {args.source_run_id}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not tag old run: {exc}", flush=True)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        job_type="clean_replay",
        tags=["clean-replay", "linear10-learn", "full-eval-ladder"],
        config={
            "source_run_id": args.source_run_id,
            "through_step": args.through_step,
            "ema_wandb_step": EMA_WANDB_STEP,
            "replay_notes": args.notes,
        },
        notes=args.notes,
        reinit=True,
    )
    print(f"new_run={run.id} url={run.url}", flush=True)

    train_steps = sorted(train_by_step)
    eval_ladder_steps = sorted(s for s in eval_by_step if s <= args.through_step)
    merged_steps = sorted(set(train_steps) | set(eval_ladder_steps))
    for step in merged_steps:
        metrics: dict[str, float] = {}
        metrics.update(train_by_step.get(step, {}))
        metrics.update(eval_by_step.get(step, {}))
        if not metrics:
            continue
        run.log(metrics, step=step)
        if step % 250 == 0 or step == args.through_step:
            print(f"logged step={step} n_metrics={len(metrics)}", flush=True)
        if args.upload_artifacts and step in eval_by_step:
            path = args.task_loss_dir / f"step{step}_task_loss.json"
            if path.is_file():
                art = wandb.Artifact(name=f"eval-step{step:07d}", type="eval")
                art.add_file(str(path), name=path.name)
                run.log_artifact(art)

    ema_metrics = dict(eval_by_step[EMA_WANDB_STEP])
    ema_metrics["final_model"] = 1.0
    ema_metrics["posthoc_ema"] = 1.0
    run.log(ema_metrics, step=EMA_WANDB_STEP)
    print(f"logged ema step={EMA_WANDB_STEP} macro={ema_metrics.get('eval/macro_bpb')}", flush=True)
    ema_path = args.task_loss_dir / f"step{EMA_WANDB_STEP}_task_loss.json"
    if args.upload_artifacts and ema_path.is_file():
        art = wandb.Artifact(name="eval-step02385-ema", type="eval")
        art.add_file(str(ema_path), name=ema_path.name)
        run.log_artifact(art)

    run.alert(
        title="clean curriculum W&B replay ready",
        text=f"Replayed from {args.source_run_id}; eval ladder includes step 2000.",
        level=wandb.AlertLevel.INFO,
    )
    run.finish()

    args.run_id_out.parent.mkdir(parents=True, exist_ok=True)
    args.run_id_out.write_text(str(run.id) + "\n", encoding="utf-8")
    if args.prev_id_out is not None:
        args.prev_id_out.write_text(args.source_run_id + "\n", encoding="utf-8")
    print(f"wrote {args.run_id_out} -> {run.id}", flush=True)


if __name__ == "__main__":
    main()
