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

from eval_arc_task_loss_smollm import run_suite

EVAL_TASKS = ("HellaSwag", "PIQA", "OpenBookQA")
REQUIRED_EVAL_BPB_LABELS = (
    "hellaswag_val_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
)


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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Remove torch.compile and DDP wrappers for save/load operations."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        if isinstance(model, DDP):
            model = model.module
            continue
        return model


def resolve_resume_geometry(
    state: dict,
    *,
    world_size: int,
    per_device_batch_size: int,
    seq_len: int,
) -> str:
    """Return how to resume: ``exact`` or ``same_global_batch``.

    Cross-GPU-count resumes are allowed only when tokens/step stay identical
    (e.g. 4x40 -> 8x20). Optimizer Adam state is parameter-shaped and still
    loads; the LR schedule is rebuilt to the resumed step.
    """
    old_seq = int(state["seq_len"])
    if old_seq != seq_len:
        raise ValueError(
            f"refusing resume with changed seq_len: checkpoint={old_seq} current={seq_len}"
        )
    old_world = int(state["world_size"])
    old_batch = int(state["per_device_batch_size"])
    old_global = int(
        state.get("global_batch_tokens", old_world * old_batch * old_seq)
    )
    new_global = world_size * per_device_batch_size * seq_len
    if old_world == world_size and old_batch == per_device_batch_size:
        return "exact"
    if old_global != new_global:
        raise ValueError(
            "refusing resume with changed global batch tokens: "
            f"checkpoint={old_global} (world={old_world} batch={old_batch}) "
            f"current={new_global} (world={world_size} batch={per_device_batch_size})"
        )
    return "same_global_batch"


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
    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(checkpoint, safe_serialization=True)
    size = world_size()
    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "world_size": size,
            "per_device_batch_size": args.per_device_batch_size,
            "seq_len": args.seq_len,
            "global_batch_tokens": args.per_device_batch_size * size * args.seq_len,
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


def _is_kept_eval_metric(key: str) -> bool:
    k = key.lower()
    if "arc_easy" in k or "arc_challenge" in k or "/arc_" in k:
        return False
    return True


def wandb_log_eval(
    run: object | None,
    payload: dict,
    *,
    step: int,
    eval_path: Path,
    log_artifacts: bool = True,
) -> None:
    if run is None:
        return
    metrics: dict[str, float] = {}
    for k, v in (payload.get("labels") or {}).items():
        mk = f"eval/bpb/{k}"
        if _is_kept_eval_metric(mk):
            metrics[mk] = float(v)
    for k, v in (payload.get("accuracy_labels") or {}).items():
        mk = f"eval/acc/{k}"
        if _is_kept_eval_metric(mk):
            metrics[mk] = float(v)
    for k, v in (payload.get("task_families") or {}).items():
        mk = f"eval/family_bpb/{k}"
        if _is_kept_eval_metric(mk):
            metrics[mk] = float(v)
    for k, v in (payload.get("accuracy_families") or {}).items():
        mk = f"eval/family_acc/{k}"
        if _is_kept_eval_metric(mk):
            metrics[mk] = float(v)
    fam_bpb = [v for k, v in metrics.items() if k.startswith("eval/family_bpb/")]
    fam_acc = [v for k, v in metrics.items() if k.startswith("eval/family_acc/")]
    if fam_bpb:
        metrics["eval/macro_bpb"] = sum(fam_bpb) / len(fam_bpb)
    if fam_acc:
        metrics["eval/macro_acc"] = sum(fam_acc) / len(fam_acc)
    run.log(metrics, step=step)
    if log_artifacts:
        art = wandb.Artifact(name=f"eval-step{step:07d}", type="eval")
        art.add_file(str(eval_path), name=eval_path.name)
        run.log_artifact(art)


def eval_payload_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    labels = payload.get("labels") or {}
    return all(label in labels for label in REQUIRED_EVAL_BPB_LABELS)


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


def run_task_eval(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    out_path: Path,
    run_name: str,
    device: torch.device,
) -> dict | None:
    model.eval()
    payload = run_suite(
        model,
        tokenizer,
        list(EVAL_TASKS),
        n_shot=5,
        device=device,
        seed=42,
        run_name=run_name,
        rank=rank(),
        world_size=world_size(),
    )
    if rank() == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    model.train()
    return payload if rank() == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--run-name", default="smollm2-135m-colmlm-20b")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=40)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--max-train-tokens", type=int, default=20_000_000_000)
    parser.add_argument("--checkpoint-every-tokens", type=int, default=250_000_000)
    parser.add_argument("--eval-interval-tokens", type=int, default=250_000_000)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable HellaSwag/PIQA/OpenBookQA task-loss eval.",
    )
    parser.add_argument("--keep-local-checkpoints", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the DDP-wrapped model (default: enabled).",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trade throughput for memory; disabled by default on 48+ GB GPUs.",
    )
    parser.add_argument(
        "--liger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse Llama layers and linear cross-entropy (default: enabled).",
    )
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--wandb-project", default="edullm-smollm2-colmlm")
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
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
    eval_every = (
        0
        if args.no_eval
        else max(1, round(args.eval_interval_tokens / global_batch_tokens))
    )
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
        "eval_every_steps": eval_every,
        "eval_interval_tokens": args.eval_interval_tokens,
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
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_llama

        apply_liger_kernel_to_llama(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
        )

    config = AutoConfig.from_pretrained(args.model_id)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device=device, dtype=torch.bfloat16)
    if size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=False,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
        )
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
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
        resume_mode = resolve_resume_geometry(
            state,
            world_size=size,
            per_device_batch_size=args.per_device_batch_size,
            seq_len=args.seq_len,
        )
        loaded = AutoModelForCausalLM.from_pretrained(args.resume_from, torch_dtype=torch.bfloat16)
        unwrap_model(model).load_state_dict(loaded.state_dict())
        optimizer.load_state_dict(state["optimizer"])
        step, tokens_seen = int(state["step"]), int(state["tokens_seen"])
        if resume_mode == "exact":
            scheduler.load_state_dict(state["scheduler"])
        else:
            # Same tokens/step, different rank count: rebuild cosine schedule at
            # the resumed step so warmup/total match the new process group.
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                last_epoch=step - 1 if step > 0 else -1,
            )
            if rank() == 0:
                print(
                    f"resume mode={resume_mode}: "
                    f"world {int(state['world_size'])}x{int(state['per_device_batch_size'])} "
                    f"-> {size}x{args.per_device_batch_size} "
                    f"(global_batch_tokens={global_batch_tokens})",
                    flush=True,
                )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

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
    next_eval = ((step // eval_every) + 1) * eval_every if eval_every else 0
    if eval_every and step > 0:
        eval_out = args.output_dir / "task_loss" / f"step{step:07d}_task_loss.json"
        need_eval = False
        if rank() == 0:
            need_eval = not eval_payload_complete(eval_out)
        if size > 1:
            flag = torch.tensor([1 if need_eval else 0], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            need_eval = bool(int(flag.item()))
        if need_eval:
            if rank() == 0:
                print(f"running sharded task eval at resumed step {step}", flush=True)
            run_task_eval(unwrap_model(model), tokenizer, eval_out, args.run_name, device)
            if rank() == 0:
                append_task_loss_curve(args.output_dir / "progress", step, eval_out)
                payload = json.loads(eval_out.read_text(encoding="utf-8"))
                wandb_log_eval(
                    wb_run,
                    payload,
                    step=step,
                    eval_path=eval_out,
                    log_artifacts=not args.no_wandb_artifacts,
                )
                print(f"task loss eval wrote {eval_out}", flush=True)
            if size > 1:
                dist.barrier()
        next_eval = ((step // eval_every) + 1) * eval_every
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
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, foreach=True
            )
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

        if eval_every and step >= next_eval:
            if rank() == 0:
                print(f"running sharded task eval at step {step}", flush=True)
            eval_out = args.output_dir / "task_loss" / f"step{step:07d}_task_loss.json"
            run_task_eval(unwrap_model(model), tokenizer, eval_out, args.run_name, device)
            if rank() == 0:
                append_task_loss_curve(args.output_dir / "progress", step, eval_out)
                payload = json.loads(eval_out.read_text(encoding="utf-8"))
                wandb_log_eval(
                    wb_run,
                    payload,
                    step=step,
                    eval_path=eval_out,
                    log_artifacts=not args.no_wandb_artifacts,
                )
                print(f"task loss eval wrote {eval_out}", flush=True)
            if size > 1:
                dist.barrier()
            next_eval += eval_every

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
