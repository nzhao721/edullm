#!/usr/bin/env python3
"""Log complete 20-label RefHQ task-loss JSONs to W&B (MixLaw-comparable).

Creates or resumes a single run under project ``refhq`` and logs
``eval/macro_bpb`` + ``eval/bpb/<label>`` at each checkpoint step — the same
keys MixLaw 370M validation uses.

Rejects incomplete / legacy 11-label payloads so curves stay scientifically
comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

STEP_RE = re.compile(r"step(\d+)_task_loss\.json$", re.IGNORECASE)

# Exact MixLaw / OLMo-ladder RC 5-shot suite (20 labels).
TASK_LOSS_RAW_LABELS = (
    "arc_challenge_val_rc_5shot_bpb",
    "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
)
_RAW = frozenset(TASK_LOSS_RAW_LABELS)


def _label_values(payload: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in ("labels", "task_loss_bpb"):
        source = payload.get(field) or {}
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[str(key)] = float(value)
    return values


def suite_complete(payload: Mapping[str, Any]) -> bool:
    return _RAW.issubset(_label_values(payload))


def metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    labels = _label_values(payload)
    if not _RAW.issubset(labels):
        missing = sorted(_RAW - set(labels))
        raise ValueError(f"incomplete suite: missing {len(missing)} labels e.g. {missing[:3]}")
    metrics = {f"eval/bpb/{k}": labels[k] for k in TASK_LOSS_RAW_LABELS}
    metrics["eval/macro_bpb"] = sum(labels[k] for k in TASK_LOSS_RAW_LABELS) / len(
        TASK_LOSS_RAW_LABELS
    )
    # Derived aggregates (same as MixLaw eval JSON / W&B summary extras).
    mmlu_keys = [k for k in TASK_LOSS_RAW_LABELS if k.startswith("mmlu_")]
    metrics["eval/bpb/mmlu_avg_rc_5shot_bpb"] = sum(labels[k] for k in mmlu_keys) / len(
        mmlu_keys
    )
    core_keys = [
        "hellaswag_val_rc_5shot_bpb",
        "arc_challenge_test_rc_5shot_bpb",
        "arc_easy_test_rc_5shot_bpb",
        "piqa_val_rc_5shot_bpb",
        "csqa_val_rc_5shot_bpb",
        "socialiqa_val_rc_5shot_bpb",
        "openbookqa_test_rc_5shot_bpb",
        "boolq_val_rc_5shot_bpb",
        "winogrande_val_rc_5shot_bpb",
    ]
    metrics["eval/bpb/core_avg_rc_5shot_bpb"] = sum(labels[k] for k in core_keys) / len(
        core_keys
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--wandb-project", default="refhq")
    p.add_argument("--wandb-entity", default="eduLLM")
    p.add_argument("--wandb-run-name", default="refhq-370m-posthoc-20label")
    p.add_argument(
        "--wandb-run-id",
        default=None,
        help="Resume this run id; if unset, create a new run and write --run-id-out",
    )
    p.add_argument("--run-id-out", type=Path, default=None)
    p.add_argument("--upload-artifacts", action="store_true")
    p.add_argument(
        "--require-all-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any step*_task_loss.json is incomplete (default: true)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY unset", file=sys.stderr)
        return 2

    import wandb

    paths = sorted(args.eval_dir.glob("step*_task_loss.json"))
    if not paths:
        print(f"no step*_task_loss.json under {args.eval_dir}", file=sys.stderr)
        return 2

    by_step: dict[int, tuple[Path, dict[str, float]]] = {}
    incomplete: list[str] = []
    for path in paths:
        m = STEP_RE.search(path.name)
        if not m:
            print(f"skip (bad name): {path.name}", flush=True)
            continue
        step = int(m.group(1))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not suite_complete(payload):
            incomplete.append(path.name)
            print(f"INCOMPLETE {path.name}", flush=True)
            continue
        by_step[step] = (path, metrics_from_payload(payload))

    if incomplete and args.require_all_complete:
        print(
            f"abort: {len(incomplete)} incomplete payloads "
            f"(first={incomplete[0]!r}); re-run evals with the 20-label suite",
            file=sys.stderr,
        )
        return 2
    if not by_step:
        print("no complete 20-label payloads to log", file=sys.stderr)
        return 2

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity or None,
        "name": args.wandb_run_name,
        "job_type": "refhq_posthoc_task_loss",
        "tags": ["refhq", "olmo2-370m", "task-loss-20label", "posthoc"],
        "config": {
            "suite": "olmes_rc_5shot_bpb",
            "raw_label_count": len(TASK_LOSS_RAW_LABELS),
            "labels": list(TASK_LOSS_RAW_LABELS),
            "comparable_to": "mixlaw-370m-validation",
            "eval_dir": str(args.eval_dir),
        },
        "reinit": True,
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    if args.run_id_out is not None:
        args.run_id_out.parent.mkdir(parents=True, exist_ok=True)
        args.run_id_out.write_text(str(run.id) + "\n", encoding="utf-8")

    print(
        f"wandb run={run.id} url={run.url} "
        f"steps={sorted(by_step)} n={len(by_step)}",
        flush=True,
    )

    for step in sorted(by_step):
        path, metrics = by_step[step]
        run.log(metrics, step=step)
        print(
            f"logged step={step} macro_bpb={metrics['eval/macro_bpb']:.6f} "
            f"file={path.name}",
            flush=True,
        )
        if args.upload_artifacts:
            art = wandb.Artifact(name=f"eval-step{step:07d}", type="eval")
            art.add_file(str(path), name=path.name)
            run.log_artifact(art)

    run.finish()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
