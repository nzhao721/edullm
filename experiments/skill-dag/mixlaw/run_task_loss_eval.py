#!/usr/bin/env python3
"""Run OLMo-ladder task loss (downstream RC bpb) on a single checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import get_local_rank, peak_gpu_memory
from olmo.train import Trainer
from olmo.util import add_cached_path_clients, prepare_cli_environment

log = logging.getLogger("task_loss_eval")

from mixlaw_common import CURVE_TASK_LOSS_LABELS


def build_eval_config(base_config: Path, load_path: Path, device_eval_batch_size: int) -> TrainConfig:
    cfg = TrainConfig.load(str(base_config), [])
    cfg.load_path = str(load_path)
    cfg.try_load_latest_save = False
    cfg.eval_on_load = False
    cfg.wandb = None
    cfg.device_eval_batch_size = device_eval_batch_size
    # Full downstream eval: do not cap batches (None breaks olmo.train.eval()).
    cfg.eval_subset_num_batches = 10**9
    cfg.evaluators = [
        EvaluatorConfig(label=label, type=EvaluatorType.downstream)
        for label in CURVE_TASK_LOSS_LABELS
    ]
    cfg.run_name = f"task-loss-{load_path.name}"
    return cfg


def main(cfg: TrainConfig, output_path: Path) -> None:
    if cfg.load_path is None:
        raise OLMoConfigurationError("load_path is required")

    device = torch.device("cuda")
    evaluators = build_evaluators(cfg, device)
    if not evaluators:
        raise OLMoConfigurationError("No evaluators configured")

    log.info("Building model...")
    olmo_model = OLMo(cfg.model)
    log.info("Non-embedding params: %s", f"{olmo_model.num_params(include_embedding=False):,d}")
    log.info("Peak GPU memory before DDP: %s MB", int(peak_gpu_memory() or 0))

    dist_model = torch.nn.parallel.DistributedDataParallel(olmo_model.to(device))
    olmo_model.reset_parameters()

    optim = build_optimizer(cfg, dist_model)
    scheduler = build_scheduler(cfg)

    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        dist_model=dist_model,
        device=device,
        evaluators=evaluators,
        optim=optim,
        scheduler=scheduler,
        train_loader=None,  # type: ignore[arg-type]
    ) as trainer:
        log.info("Loading checkpoint from %s", cfg.load_path)
        trainer.restore_checkpoint(
            cfg.load_path,
            load_optimizer_state=False,
            load_trainer_state=False,
            sharded_checkpointer=cfg.load_path_sharded_checkpointer,
        )
        log.info("Evaluating downstream task loss...")
        metrics = trainer.eval()

    selected: dict[str, float] = {}
    for label in CURVE_TASK_LOSS_LABELS:
        for key in (
            f"eval/downstream_bpb/{label}",
            f"eval/downstream_bpb/{label}_bpb",
        ):
            if key in metrics:
                selected[label] = float(metrics[key])
                break
        else:
            log.warning("Missing metric for %s", label)

    mmlu_keys = [k for k in selected if k.startswith("mmlu_") and k.endswith("_val_rc_5shot_bpb")]
    if mmlu_keys:
        selected["mmlu_avg_val_rc_5shot_bpb"] = sum(selected[k] for k in mmlu_keys) / len(mmlu_keys)

    payload = {
        "load_path": cfg.load_path,
        "step": int(Path(cfg.load_path).name.replace("step", "").replace("-unsharded", "")),
        "task_loss_bpb": selected,
        "all_eval_metrics": {k: float(v) for k, v in metrics.items() if "downstream_bpb" in k},
    }

    if dist.get_rank() == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--load-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device-eval-batch-size", type=int, default=4)
    args = ap.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    prepare_cli_environment()
    add_cached_path_clients()

    cfg = build_eval_config(Path(args.base_config), Path(args.load_path), args.device_eval_batch_size)
    main(cfg, Path(args.output))
