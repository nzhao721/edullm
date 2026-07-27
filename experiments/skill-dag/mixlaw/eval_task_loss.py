#!/usr/bin/env python3
"""Full OLMo-ladder task-loss evaluation of a finished mixture checkpoint.

This produces the numbers the mixing law is actually fitted to. During training
only a cheap subset of labels is evaluated on a capped number of batches; here the
complete 20-label OLMES 5-shot RC suite is run to convergence on the whole task
set, which matters because the mixing law is fitted across only 24 points and
per-label noise propagates directly into the coefficients.

Task loss is bits-per-byte of the gold continuation:

    bpb = -log2 p(continuation | context) / utf8_bytes(continuation)

which is what ``ICLMetric(metric_type="bpb")`` computes, and what Bhagia et al.
(arXiv:2412.04403) fit power laws in. It is byte-normalized, so it is comparable
across tokenizers and across the 24 runs regardless of how each mixture shifts
token statistics.

Writes ``task_loss_final.json`` next to the checkpoint's progress directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import timedelta
from pathlib import Path

os.environ["WANDB_DISABLED"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.distributed as dist
from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig
from olmo.eval import build_evaluator
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.torch_util import get_local_rank
from olmo.util import add_cached_path_clients, prepare_cli_environment

from mixlaw_common import LADDER_TASK_LOSS_LABELS, task_family

log = logging.getLogger("eval_task_loss")


def evaluate_label(
    model: OLMo,
    cfg: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    label: str,
    device_eval_batch_size: int,
    max_batches: int | None = None,
) -> tuple[float, int]:
    """Return (task_loss_bpb, num_batches) for one ladder label."""
    evaluator = build_evaluator(
        cfg,
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            device_eval_batch_size=device_eval_batch_size,
            subset_num_batches=max_batches,
        ),
        tokenizer,
        device,
    )
    evaluator.reset_metrics()

    n_batches = 0
    for batch in evaluator.eval_loader:
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(input_ids=batch["input_ids"]).logits
        evaluator.eval_metric.update(batch, logits)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    metrics = evaluator.compute_metrics()
    # compute_metrics returns {"eval/<label>": tensor}; a bpb metric yields the
    # mean per-instance bits-per-byte of the gold continuation.
    (value,) = list(metrics.values())
    return float(value), n_batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Unsharded OLMo checkpoint dir")
    ap.add_argument("--out", type=Path, required=True, help="Where task_loss_final.json is written")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device-eval-batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Subset of the 20 ladder bpb labels (default: all)",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per label (smoke tests: 1 exercises every label without a full pass)",
    )
    args = ap.parse_args()

    labels = list(args.labels) if args.labels else list(LADDER_TASK_LOSS_LABELS)
    unknown = [lbl for lbl in labels if lbl not in LADDER_TASK_LOSS_LABELS]
    if unknown:
        raise SystemExit(f"not OLMo-ladder task-loss labels: {unknown}")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    # ICLMetric is a torchmetrics Metric with sync_on_compute, so it needs a
    # process group even for a single-rank evaluation.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()
    device = torch.device("cuda")

    cfg = TrainConfig.load(args.checkpoint / "config.yaml")
    cfg.data.num_workers = args.num_workers
    cfg.device_eval_batch_size = args.device_eval_batch_size
    cfg.evaluators = []

    model = OLMo.from_checkpoint(str(args.checkpoint), device="cuda").eval()
    tokenizer = Tokenizer.from_train_config(cfg)

    results: dict[str, float] = {}
    for label in labels:
        bpb, n_batches = evaluate_label(
            model,
            cfg,
            tokenizer,
            device,
            label,
            args.device_eval_batch_size,
            max_batches=args.max_batches,
        )
        results[label] = bpb
        log.info("%s %s: task_loss_bpb=%.6f (%d batches)", args.run_name, label, bpb, n_batches)
        print(f"{label}\t{bpb:.6f}")

    # Family means collapse val/test splits of the same task, which is the level
    # the mixing law is fitted at (one L_i per task, not per split).
    families: dict[str, list[float]] = {}
    for label, bpb in results.items():
        families.setdefault(task_family(label), []).append(bpb)
    family_means = {fam: sum(vals) / len(vals) for fam, vals in sorted(families.items())}

    payload = {
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "metric": "task_loss_bpb",
        "definition": "-log2 p(gold continuation | context) / utf8 bytes of continuation",
        "max_batches": args.max_batches,
        "smoke": args.max_batches is not None,
        "labels": results,
        "task_families": family_means,
        "macro_mean": sum(family_means.values()) / len(family_means),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"macro_mean_task_loss_bpb\t{payload['macro_mean']:.6f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
