#!/usr/bin/env python3
"""Sequentially log offline eval JSON files into an existing W&B run (CPU-only).

Merges multiple task JSON files that share the same training step into one
``wandb.log(..., step=ckpt_step)`` call on the default training-step x-axis
(so eval curves share x with ``train/*``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


STEP_RE = re.compile(r"step(\d+)_", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-id", required=True)
    p.add_argument("--glob", default="*.json", help="Filename glob under eval-dir")
    p.add_argument(
        "--upload-artifacts",
        action="store_true",
        help="Also upload each JSON as a W&B artifact (slower).",
    )
    return p.parse_args()


def metrics_from_payload(payload: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k, v in (payload.get("labels") or {}).items():
        metrics[f"eval/bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_labels") or {}).items():
        metrics[f"eval/acc/{k}"] = float(v)
    for k, v in (payload.get("task_families") or {}).items():
        metrics[f"eval/family_bpb/{k}"] = float(v)
    for k, v in (payload.get("accuracy_families") or {}).items():
        metrics[f"eval/family_acc/{k}"] = float(v)
    return metrics


def recompute_macros(metrics: dict[str, float]) -> None:
    fam_bpb = [v for k, v in metrics.items() if k.startswith("eval/family_bpb/")]
    fam_acc = [v for k, v in metrics.items() if k.startswith("eval/family_acc/")]
    if fam_bpb:
        metrics["eval/macro_bpb"] = sum(fam_bpb) / len(fam_bpb)
    if fam_acc:
        metrics["eval/macro_acc"] = sum(fam_acc) / len(fam_acc)


def main() -> None:
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY unset")

    import wandb

    paths = sorted(args.eval_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.eval_dir}/{args.glob}")

    by_step: dict[int, dict[str, float]] = {}
    files_by_step: dict[int, list[Path]] = {}
    for path in paths:
        m = STEP_RE.search(path.name)
        if not m:
            print(f"skip (no step in name): {path.name}", flush=True)
            continue
        step = int(m.group(1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_step.setdefault(step, {}).update(metrics_from_payload(payload))
        files_by_step.setdefault(step, []).append(path)

    for step, metrics in by_step.items():
        recompute_macros(metrics)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        id=args.wandb_run_id,
        resume="must",
        job_type="offline_eval_backfill",
        reinit=True,
    )
    # Log on the default training-step axis so eval curves share x with train/*.
    print(
        f"wandb backfill run={run.id} url={run.url} steps={len(by_step)} files={len(paths)}",
        flush=True,
    )

    for step in sorted(by_step):
        metrics = dict(by_step[step])
        # Drop ARC if present in older combined JSON payloads.
        metrics = {
            k: v
            for k, v in metrics.items()
            if "arc_easy" not in k.lower() and "arc_challenge" not in k.lower()
        }
        run.log(metrics, step=step)
        print(
            f"logged step={step} n_metrics={len(metrics)} "
            f"files={[p.name for p in files_by_step[step]]}",
            flush=True,
        )
        if args.upload_artifacts:
            art = wandb.Artifact(name=f"eval-step{step:07d}-offline", type="eval")
            for path in files_by_step[step]:
                art.add_file(str(path), name=path.name)
            run.log_artifact(art)

    run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
