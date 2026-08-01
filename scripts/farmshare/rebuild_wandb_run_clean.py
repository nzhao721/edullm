#!/usr/bin/env python3
"""Rebuild a clean W&B run through a checkpoint step (no fork/rewind API).

Creates a **new** run that:
  1. Replays train/* metrics from ``--source-run-id`` for ``_step <= --through-step``
     (drops ARC eval keys and anything after the broken segment).
  2. Backfills HellaSwag / PIQA / OpenBookQA evals from local JSON onto the **same**
     training-step x-axis (``wandb.log(..., step=ckpt_step)``).
  3. Writes the new run id to ``--run-id-out`` (and optionally tags the old run).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


STEP_RE = re.compile(r"step(\d+)_", re.IGNORECASE)
SKIP_HISTORY_KEYS = {"_step", "_timestamp", "_runtime", "_wandb"}
KEEP_EVAL_SUBSTR = ("hellaswag", "piqa", "openbookqa", "macro_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="edullm-smollm2")
    p.add_argument("--entity", default="eduLLM")
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--through-step", type=int, required=True)
    p.add_argument("--run-name", default="smollm2-135m-750m-27ep-fresh")
    p.add_argument("--task-loss-dir", type=Path, required=True)
    p.add_argument("--piqa-obqa-dir", type=Path, required=True)
    p.add_argument("--run-id-out", type=Path, required=True)
    p.add_argument("--prev-id-out", type=Path, default=None)
    p.add_argument("--wandb-dir", type=Path, default=None)
    return p.parse_args()


def keep_metric_key(key: str) -> bool:
    k = key.lower()
    if key in SKIP_HISTORY_KEYS or key.startswith("_"):
        return False
    if "arc_easy" in k or "arc_challenge" in k:
        return False
    if k.startswith("eval/") or "/eval" in k or k.startswith("eval"):
        return any(s in k for s in KEEP_EVAL_SUBSTR)
    return True


def metrics_from_payload(payload: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (payload.get("labels") or {}).items():
        mk = f"eval/bpb/{k}"
        if keep_metric_key(mk):
            out[mk] = float(v)
    for k, v in (payload.get("accuracy_labels") or {}).items():
        mk = f"eval/acc/{k}"
        if keep_metric_key(mk):
            out[mk] = float(v)
    for k, v in (payload.get("task_families") or {}).items():
        mk = f"eval/family_bpb/{k}"
        if keep_metric_key(mk):
            out[mk] = float(v)
    for k, v in (payload.get("accuracy_families") or {}).items():
        mk = f"eval/family_acc/{k}"
        if keep_metric_key(mk):
            out[mk] = float(v)
    fam_bpb = [v for k, v in out.items() if k.startswith("eval/family_bpb/")]
    fam_acc = [v for k, v in out.items() if k.startswith("eval/family_acc/")]
    if fam_bpb:
        out["eval/macro_bpb"] = sum(fam_bpb) / len(fam_bpb)
    if fam_acc:
        out["eval/macro_acc"] = sum(fam_acc) / len(fam_acc)
    return out


def load_local_evals(task_loss_dir: Path, piqa_obqa_dir: Path, through_step: int) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for path in sorted(task_loss_dir.glob("step*_task_loss.json")):
        m = STEP_RE.search(path.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > through_step:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_step.setdefault(step, {}).update(metrics_from_payload(payload))
    if piqa_obqa_dir.is_dir():
        for path in sorted(piqa_obqa_dir.glob("step*_*.json")):
            m = STEP_RE.search(path.name)
            if not m:
                continue
            step = int(m.group(1))
            if step > through_step:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            by_step.setdefault(step, {}).update(metrics_from_payload(payload))
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

    # Collect train history <= through_step.
    hist_by_step: dict[int, dict[str, float]] = {}
    n_rows = 0
    for row in old.scan_history(page_size=1000):
        n_rows += 1
        step = row.get("_step")
        if step is None:
            continue
        step = int(step)
        if step > args.through_step:
            continue
        metrics = {k: float(v) for k, v in row.items() if keep_metric_key(k) and isinstance(v, (int, float))}
        if not metrics:
            continue
        hist_by_step.setdefault(step, {}).update(metrics)
    print(f"scanned_history_rows={n_rows} kept_steps={len(hist_by_step)}", flush=True)

    eval_by_step = load_local_evals(args.task_loss_dir, args.piqa_obqa_dir, args.through_step)
    print(f"local_eval_steps={len(eval_by_step)}", flush=True)

    # Tag old run as superseded (best-effort).
    try:
        tags = list(old.tags or [])
        for tag in ("broken-1gpu-discard", "superseded"):
            if tag not in tags:
                tags.append(tag)
        old.tags = tags
        if old.name and "BROKEN" not in old.name:
            old.name = f"{old.name}-BROKEN-1gpu"
        old.notes = (
            (old.notes or "")
            + f"\nSuperseded: clean rebuild through step {args.through_step} (no ARC; PIQA/OBQA/HellaSwag only)."
        ).strip()
        old.update()
        print(f"tagged old run {args.source_run_id}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not tag old run: {exc}", flush=True)

    init_kwargs = {
        "project": args.project,
        "entity": args.entity,
        "name": args.run_name,
        "job_type": "train",
        "tags": ["clean-rebuild", "no-arc", "piqa", "openbookqa", "hellaswag"],
        "config": {
            "source_run_id": args.source_run_id,
            "through_step": args.through_step,
            "eval_tasks": ["HellaSwag", "PIQA", "OpenBookQA"],
        },
        "reinit": True,
    }
    if args.wandb_dir is not None:
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(args.wandb_dir)

    run = wandb.init(**init_kwargs)
    print(f"new_run={run.id} url={run.url}", flush=True)

    # Merge eval metrics into the same step dicts, then log once per step on the
    # default training-step x-axis.
    all_steps = sorted(set(hist_by_step) | set(eval_by_step))
    for step in all_steps:
        metrics = {}
        metrics.update(hist_by_step.get(step, {}))
        metrics.update(eval_by_step.get(step, {}))
        # Drop any residual ARC keys from history replay.
        metrics = {k: v for k, v in metrics.items() if keep_metric_key(k)}
        if not metrics:
            continue
        run.log(metrics, step=step)
    print(f"logged_steps={len(all_steps)} through_step={args.through_step}", flush=True)

    run.alert(
        title="clean W&B rebuild ready",
        text=f"Rebuilt from {args.source_run_id} through step {args.through_step}; ARC dropped.",
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
