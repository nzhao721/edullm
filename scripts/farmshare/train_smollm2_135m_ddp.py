#!/usr/bin/env python3
"""Multi-GPU DDP pretraining for SmolLM2-135M on uint32 token memmaps."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSample
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from eval_arc_task_loss_smollm import run_suite


class MemmapChunkDataset(Dataset):
    def __init__(self, tokens_mm: np.memmap, seq_len: int) -> None:
        if len(tokens_mm) <= seq_len:
            raise ValueError(f"memmap too short for seq_len={seq_len}: {len(tokens_mm)} tokens")
        self.tokens_mm = tokens_mm
        self.seq_len = seq_len
        self.num_chunks = (len(tokens_mm) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.num_chunks
        start = idx * self.seq_len
        input_ids = np.asarray(self.tokens_mm[start : start + self.seq_len], dtype=np.int64)
        return {"input_ids": torch.from_numpy(input_ids.copy())}


def load_token_memmap(data_dir: Path) -> tuple[np.memmap, dict]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    num_tokens = int(meta["num_tokens"])
    tokens_mm = np.memmap(data_dir / "train_tokens.bin", dtype=np.uint32, mode="r", shape=(num_tokens,))
    return tokens_mm, meta


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rendezvous_file(path: Path, *, create: bool, timeout_s: float = 10_800.0) -> None:
    """Filesystem rendezvous so long rank-0 evals do not sit in NCCL barriers."""
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("done\n", encoding="utf-8")
        tmp.replace(path)
        return
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"timed out waiting for rendezvous file {path}")
        time.sleep(1.0)


def sync_after_side_work(output_dir: Path, tag: str) -> None:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    flag = output_dir / "progress" / f".sync_{tag}"
    if is_main_process():
        rendezvous_file(flag, create=True)
    else:
        rendezvous_file(flag, create=False)
    # Short collective once everyone is past the long rank-0 work.
    dist.barrier()
    if is_main_process() and flag.exists():
        flag.unlink(missing_ok=True)


def clear_sync_flag(output_dir: Path, tag: str) -> None:
    if not is_main_process():
        return
    flag = output_dir / "progress" / f".sync_{tag}"
    flag.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2-135M multi-GPU pretrain.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--run-name", default="smollm2-135m-500m-40ep")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--max-train-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--checkpoint-interval-epochs", type=float, default=0.5)
    parser.add_argument("--eval-interval-epochs", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def save_checkpoint(
    output_dir: Path,
    *,
    step: int,
    epoch: float,
    tokens_seen: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
) -> Path:
    ckpt_dir = output_dir / "checkpoints" / f"step{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(ckpt_dir)
    state = {
        "step": step,
        "epoch": epoch,
        "tokens_seen": tokens_seen,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(state, ckpt_dir / "trainer_state.pt")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt_dir), encoding="utf-8")
    return ckpt_dir


EVAL_TASKS = ("ARC-Easy", "ARC-Challenge", "HellaSwag")


def run_task_eval(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    out_path: Path,
    run_name: str,
    device: torch.device,
    *,
    checkpoint: str | None = None,
) -> dict | None:
    """Run ARC+HellaSwag eval on all ranks (sharded); rank 0 writes JSON."""
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    payload = run_suite(
        model,
        tokenizer,
        list(EVAL_TASKS),
        n_shot=5,
        device=device,
        seed=42,
        run_name=run_name,
        checkpoint=checkpoint,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    model.train()
    return payload if is_main_process() else None


def eval_payload_has_hellaswag(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    labels = payload.get("labels") or {}
    return "hellaswag_val_rc_5shot_bpb" in labels


def append_task_loss_curve(progress_dir: Path, step: int, payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    record = {
        "step": step,
        "task_loss_bpb": payload["labels"],
        "accuracy": payload.get("accuracy_labels", {}),
    }
    path = progress_dir / "task_loss.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) != step:
                existing.append(row)
    existing.append(record)
    path.write_text("".join(json.dumps(r) + "\n" for r in existing), encoding="utf-8")


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        # Default NCCL timeout is 10m; ARC+HellaSwag eval on rank 0 exceeds that.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "progress").mkdir(parents=True, exist_ok=True)

    tokens_mm, meta = load_token_memmap(args.data_dir)
    seq_len = int(args.seq_len or meta.get("seq_len", 2048))
    corpus_tokens = int(meta["num_tokens"])

    global_batch_tokens = args.per_device_batch_size * world_size * seq_len
    steps_per_epoch = math.ceil(corpus_tokens / global_batch_tokens)
    total_steps = min(args.num_epochs * steps_per_epoch, math.ceil(args.max_train_tokens / global_batch_tokens))
    checkpoint_every = max(1, round(args.checkpoint_interval_epochs * steps_per_epoch))
    eval_every = max(1, round(args.eval_interval_epochs * steps_per_epoch))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    if is_main_process():
        run_meta = {
            "run_name": args.run_name,
            "corpus_tokens": corpus_tokens,
            "num_epochs": args.num_epochs,
            "max_train_tokens": args.max_train_tokens,
            "global_batch_tokens": global_batch_tokens,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "checkpoint_every_steps": checkpoint_every,
            "eval_every_steps": eval_every,
            "world_size": world_size,
        }
        (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        print(json.dumps(run_meta, indent=2), flush=True)

    dataset = MemmapChunkDataset(tokens_mm, seq_len=seq_len)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    config = AutoConfig.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_config(config)
    model.gradient_checkpointing_enable()
    model.to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_step = 0
    tokens_seen = 0
    if args.resume_from is not None:
        state = torch.load(args.resume_from / "trainer_state.pt", map_location="cpu", weights_only=False)
        model_to_load = model.module if hasattr(model, "module") else model
        loaded = AutoModelForCausalLM.from_pretrained(args.resume_from, torch_dtype=torch.bfloat16)
        model_to_load.load_state_dict(loaded.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        tokens_seen = int(state.get("tokens_seen", start_step * global_batch_tokens))

    step = start_step
    running_loss = 0.0
    train_start = time.perf_counter()
    data_iter = iter(loader)
    next_checkpoint = ((start_step // checkpoint_every) + 1) * checkpoint_every
    next_eval = ((start_step // eval_every) + 1) * eval_every

    # On resume, re-run eval at the checkpoint step if HellaSwag (or any suite) is missing.
    if args.resume_from is not None:
        eval_out = args.output_dir / "task_loss" / f"step{start_step:07d}_task_loss.json"
        need_eval = False
        if is_main_process():
            need_eval = not eval_payload_has_hellaswag(eval_out)
        if world_size > 1:
            flag = torch.tensor([1 if need_eval else 0], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            need_eval = bool(int(flag.item()))
        if need_eval:
            if is_main_process():
                print(f"running sharded task eval at resumed step {start_step}", flush=True)
            eval_model = model.module if hasattr(model, "module") else model
            run_task_eval(
                eval_model,
                tokenizer,
                eval_out,
                args.run_name,
                device,
                checkpoint=str(args.resume_from),
            )
            if is_main_process():
                append_task_loss_curve(args.output_dir / "progress", start_step, eval_out)
                print(f"task loss eval wrote {eval_out}", flush=True)
            if world_size > 1:
                dist.barrier()
        next_eval = ((start_step // eval_every) + 1) * eval_every

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step // max(len(loader), 1))
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=input_ids).loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        tokens_seen += global_batch_tokens
        running_loss += float(loss.detach())

        if step % args.log_every == 0 and is_main_process():
            elapsed = time.perf_counter() - train_start
            avg_tps = (tokens_seen - start_step * global_batch_tokens) / max(elapsed, 1e-9)
            epoch = step / steps_per_epoch
            print(
                f"step={step:7d}/{total_steps} epoch={epoch:6.3f} "
                f"loss={running_loss / args.log_every:.4f} tokens_seen={tokens_seen:,} "
                f"avg_tps={avg_tps:,.0f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
            running_loss = 0.0

        if step >= next_checkpoint:
            if is_main_process():
                epoch = step / steps_per_epoch
                ckpt_dir = save_checkpoint(
                    args.output_dir,
                    step=step,
                    epoch=epoch,
                    tokens_seen=tokens_seen,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                )
                print(f"saved checkpoint {ckpt_dir}", flush=True)
            if world_size > 1:
                dist.barrier()
            next_checkpoint += checkpoint_every

        if step >= next_eval:
            if is_main_process():
                print(f"running sharded task eval at step {step}", flush=True)
            eval_out = args.output_dir / "task_loss" / f"step{step:07d}_task_loss.json"
            eval_model = model.module if hasattr(model, "module") else model
            run_task_eval(eval_model, tokenizer, eval_out, args.run_name, device)
            if is_main_process():
                append_task_loss_curve(args.output_dir / "progress", step, eval_out)
                print(f"task loss eval wrote {eval_out}", flush=True)
            if world_size > 1:
                dist.barrier()
            next_eval += eval_every

    if is_main_process():
        final_ckpt = save_checkpoint(
            args.output_dir,
            step=step,
            epoch=step / steps_per_epoch,
            tokens_seen=tokens_seen,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        print(f"finished training at step={step}, checkpoint={final_ckpt}", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
