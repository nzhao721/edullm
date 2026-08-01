#!/usr/bin/env python3
"""Full OLMo-ladder task-loss evaluation of a finished mixture checkpoint.

This produces the numbers the mixing law is fitted to. By default only the six
in-run curve labels are evaluated; pass ``--full-suite`` for all 20 ladder labels.

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
import importlib.util
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

from mixlaw_common import (
    CURVE_TASK_LOSS_LABELS,
    LADDER_TASK_LOSS_LABELS,
    patch_torch_load_for_olmo_checkpoints,
    task_family,
)

log = logging.getLogger("eval_task_loss")
_SHARED_EVAL = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "farmshare"
    / "task_loss"
    / "eval_task_loss_olmo_core.py"
)


def load_compatible_train_config(
    checkpoint: Path,
    base_config: Path | None = None,
) -> TrainConfig:
    """Reuse the shared null/schema-compatible ladder config loader."""
    if not _SHARED_EVAL.is_file():
        raise FileNotFoundError(f"shared task-loss evaluator missing: {_SHARED_EVAL}")
    spec = importlib.util.spec_from_file_location(
        "shared_eval_task_loss_olmo_core",
        _SHARED_EVAL,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import shared evaluator from {_SHARED_EVAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = base_config
    if candidate is None and (checkpoint / "config.yaml").is_file():
        candidate = checkpoint / "config.yaml"
    return module.build_train_config(candidate)


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
    patch_torch_load_for_olmo_checkpoints()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Unsharded OLMo checkpoint dir")
    ap.add_argument("--out", type=Path, required=True, help="Where task_loss_final.json is written")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device-eval-batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Optional ladder/OLMo YAML loaded through the shared compatible config loader",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Subset of ladder bpb labels (default: six in-run curve labels)",
    )
    ap.add_argument(
        "--full-suite",
        action="store_true",
        help="Evaluate all 20 ladder bpb labels instead of the curve subset",
    )
    ap.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per label (smoke tests: 1 exercises every label without a full pass)",
    )
    args = ap.parse_args()

    if args.labels is not None:
        labels = list(args.labels)
    elif args.full_suite:
        labels = list(LADDER_TASK_LOSS_LABELS)
    else:
        labels = list(CURVE_TASK_LOSS_LABELS)
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

    cfg = load_compatible_train_config(args.checkpoint, args.base_config)
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
