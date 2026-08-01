#!/usr/bin/env python3
"""RunPod DDP copy of the FarmShare SmolLM2 trainer for fact-masked CLM."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class FactMaskedDataset(Dataset):
    def __init__(self, data_dir: Path):
        ready = data_dir / "_READY.json"
        if not ready.is_file():
            raise FileNotFoundError(f"prepared corpus marker missing: {ready}")
        self.meta = json.loads(ready.read_text(encoding="utf-8"))
        self.seq_len = int(self.meta["seq_len"])
        self.tokens: list[np.memmap] = []
        self.masks: list[np.memmap] = []
        self.cumulative = [0]
        for shard in self.meta["shards"]:
            sequences = int(shard["sequences"])
            token_path = data_dir / shard["tokens"]
            mask_path = data_dir / shard["mask"]
            expected = sequences * self.seq_len
            if token_path.stat().st_size != expected * 4:
                raise ValueError(f"bad token shard size: {token_path}")
            if mask_path.stat().st_size != expected:
                raise ValueError(f"bad mask shard size: {mask_path}")
            self.tokens.append(
                np.memmap(token_path, mode="r", dtype="<u4", shape=(sequences, self.seq_len))
            )
            self.masks.append(
                np.memmap(mask_path, mode="r", dtype=np.uint8, shape=(sequences, self.seq_len))
            )
            self.cumulative.append(self.cumulative[-1] + sequences)
        if not self.tokens:
            raise ValueError("prepared corpus contains no shards")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        shard_idx = bisect.bisect_right(self.cumulative, index) - 1
        row = index - self.cumulative[shard_idx]
        ids = np.asarray(self.tokens[shard_idx][row], dtype=np.int64).copy()
        mask = np.asarray(self.masks[shard_idx][row], dtype=np.bool_).copy()
        return {
            "input_ids": torch.from_numpy(ids),
            "loss_mask": torch.from_numpy(mask),
        }


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    tokens_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
) -> Path:
    checkpoint = output_dir / "checkpoints" / f"step{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(checkpoint, safe_serialization=True)
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "world_size": world_size(),
            "per_device_batch_size": args.per_device_batch_size,
            "seq_len": args.seq_len,
        },
        checkpoint / "trainer_state.pt",
    )
    (output_dir / "latest_checkpoint.txt").write_text(str(checkpoint), encoding="utf-8")
    return checkpoint


def prune_local_checkpoints(output_dir: Path, keep: int) -> None:
    if keep < 1:
        return
    root = output_dir / "checkpoints"
    checkpoints = sorted(root.glob("step*"))
    for checkpoint in checkpoints[:-keep]:
        shutil.rmtree(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--run-name", default="smollm2-135m-colmlm-20b")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--max-train-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--checkpoint-every-tokens", type=int, default=250_000_000)
    parser.add_argument("--keep-local-checkpoints", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--wandb-project", default="edullm-smollm2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--no-wandb-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if size > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = FactMaskedDataset(args.data_dir)
    if dataset.seq_len != args.seq_len:
        raise ValueError(f"prepared seq_len={dataset.seq_len}, requested {args.seq_len}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "progress").mkdir(exist_ok=True)

    global_batch_tokens = args.per_device_batch_size * size * args.seq_len
    per_rank_samples = math.ceil(len(dataset) / size) if size > 1 else len(dataset)
    steps_per_epoch = math.floor(per_rank_samples / args.per_device_batch_size)
    if steps_per_epoch < 1:
        raise ValueError("corpus is smaller than one global batch")
    total_steps = min(
        args.num_epochs * steps_per_epoch,
        math.ceil(args.max_train_tokens / global_batch_tokens),
    )
    checkpoint_every = max(1, round(args.checkpoint_every_tokens / global_batch_tokens))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    run_meta = {
        "run_name": args.run_name,
        "data_dir": str(args.data_dir),
        "model_id": args.model_id,
        "world_size": size,
        "corpus_tokens": int(dataset.meta["tokens"]),
        "masked_targets": int(dataset.meta["masked_targets"]),
        "masked_fraction": float(dataset.meta["masked_fraction"]),
        "global_batch_tokens": global_batch_tokens,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "max_train_tokens": args.max_train_tokens,
    }
    if rank() == 0:
        (args.output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(run_meta, indent=2), flush=True)

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    config = AutoConfig.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_config(config)
    model.gradient_checkpointing_enable()
    model.to(device=device, dtype=torch.bfloat16)
    if size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    step = tokens_seen = 0
    if args.resume_from:
        state = torch.load(
            args.resume_from / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        if (
            int(state["world_size"]) != size
            or int(state["per_device_batch_size"]) != args.per_device_batch_size
            or int(state["seq_len"]) != args.seq_len
        ):
            raise ValueError("refusing resume with changed batch geometry")
        loaded = AutoModelForCausalLM.from_pretrained(args.resume_from, torch_dtype=torch.bfloat16)
        (model.module if hasattr(model, "module") else model).load_state_dict(loaded.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, tokens_seen = int(state["step"]), int(state["tokens_seen"])

    wb_run = None
    if rank() == 0 and args.wandb_mode != "disabled" and wandb is not None:
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            mode=args.wandb_mode,
            config={
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                **run_meta,
            },
            dir=str(args.output_dir / "wandb"),
        )

    model.train()
    data_epoch = step // steps_per_epoch
    if sampler is not None:
        sampler.set_epoch(data_epoch)
    data_iter = iter(loader)
    for _ in range(step % steps_per_epoch):
        next(data_iter)
    tokens_seen_at_start = tokens_seen
    interval_start = train_start = time.perf_counter()
    interval_loss = 0.0
    interval_local_unmasked = 0
    interval_steps = 0
    next_checkpoint = ((step // checkpoint_every) + 1) * checkpoint_every
    progress_path = args.output_dir / "progress" / "train.jsonl"
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_epoch += 1
            if sampler is not None:
                sampler.set_epoch(data_epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device=device, non_blocking=True)
        labels = input_ids.clone()
        labels.masked_fill_(loss_mask, -100)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        tokens_seen += global_batch_tokens
        interval_loss += float(loss.detach())
        # HF causal-LM loss shifts labels left, so label position 0 is never a target.
        interval_local_unmasked += int((~loss_mask[:, 1:]).sum().item())
        interval_steps += 1

        if step % args.log_every == 0:
            stats = torch.tensor(
                [interval_loss, float(interval_local_unmasked)],
                dtype=torch.float64,
                device=device,
            )
            if size > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            if rank() == 0:
                now = time.perf_counter()
                elapsed = max(now - interval_start, 1e-9)
                total_elapsed = max(now - train_start, 1e-9)
                recent_tps = interval_steps * global_batch_tokens / elapsed
                unmasked_tps = float(stats[1].item()) / elapsed
                metrics = {
                    "step": step,
                    "loss": float(stats[0].item()) / (interval_steps * size),
                    "tokens_seen": tokens_seen,
                    "recent_tps": recent_tps,
                    "avg_tps": (tokens_seen - tokens_seen_at_start) / total_elapsed,
                    "recent_unmasked_tps": unmasked_tps,
                    "lr": scheduler.get_last_lr()[0],
                    "world_size": size,
                }
                print(
                    " ".join(
                        [
                            f"step={step}/{total_steps}",
                            f"loss={metrics['loss']:.4f}",
                            f"tokens_seen={tokens_seen:,}",
                            f"recent_tps={recent_tps:,.0f}",
                            f"avg_tps={metrics['avg_tps']:,.0f}",
                            f"unmasked_tps={unmasked_tps:,.0f}",
                            f"lr={metrics['lr']:.2e}",
                        ]
                    ),
                    flush=True,
                )
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                if wb_run is not None:
                    wb_run.log({f"train/{k}": v for k, v in metrics.items() if k != "step"}, step=step)
                interval_start = now
            interval_loss = 0.0
            interval_local_unmasked = 0
            interval_steps = 0

        if step >= next_checkpoint:
            if rank() == 0:
                checkpoint = save_checkpoint(
                    args.output_dir,
                    step=step,
                    tokens_seen=tokens_seen,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                )
                print(f"saved checkpoint {checkpoint}", flush=True)
                if wb_run is not None and not args.no_wandb_artifacts:
                    artifact = wandb.Artifact(f"{args.run_name}-step{step}", type="model")
                    artifact.add_dir(str(checkpoint))
                    wb_run.log_artifact(artifact).wait()
                prune_local_checkpoints(args.output_dir, args.keep_local_checkpoints)
            if size > 1:
                dist.barrier()
            next_checkpoint += checkpoint_every

    if rank() == 0:
        checkpoint = save_checkpoint(
            args.output_dir,
            step=step,
            tokens_seen=tokens_seen,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        AutoTokenizer.from_pretrained(args.model_id).save_pretrained(checkpoint)
        if wb_run is not None and not args.no_wandb_artifacts:
            artifact = wandb.Artifact(f"{args.run_name}-final", type="model")
            artifact.add_dir(str(checkpoint))
            wb_run.log_artifact(artifact).wait()
        prune_local_checkpoints(args.output_dir, args.keep_local_checkpoints)
        if wb_run is not None:
            wb_run.finish()
    if size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
