#!/usr/bin/env python3
"""Single-GPU smoke-test pretrain for SmolLM2-135M on uint32 token memmaps."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class ThroughputStats:
    steps: int
    tokens_seen: int
    wall_time_s: float
    tokens_per_sec: float
    steps_per_sec: float
    final_loss: float
    peak_tokens_per_sec: float
    batch_size: int
    seq_len: int
    model: st
    data_dir: st
    device: st


class MemmapChunkDataset(Dataset):
    """Random-access fixed-length chunks over a flat uint32 token memmap."""

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
        input_ids = np.asarray(
            self.tokens_mm[start : start + self.seq_len],
            dtype=np.int64,
        )
        return {"input_ids": torch.from_numpy(input_ids.copy())}


def load_token_memmap(data_dir: Path) -> tuple[np.memmap, dict]:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokens_path = data_dir / "train_tokens.bin"
    if not tokens_path.exists():
        raise FileNotFoundError(f"missing {tokens_path}")
    num_tokens = int(meta["num_tokens"])
    tokens_mm = np.memmap(tokens_path, dtype=np.uint32, mode="r", shape=(num_tokens,))
    return tokens_mm, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolLM2-135M single-GPU smoke train.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Tokenized corpus dir with train_tokens.bin and meta.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def lr_at_step(step: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return base_l
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_l


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this smoke test")

    tokens_mm, meta = load_token_memmap(args.data_dir)
    seq_len = int(args.seq_len or meta.get("seq_len", 2048))
    dataset = MemmapChunkDataset(tokens_mm, seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    config = AutoConfig.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_config(config)
    model.to(device=device, dtype=torch.bfloat16)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    tokens_per_step = args.batch_size * seq_len
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    step_times: list[float] = []
    peak_tps = 0.0
    train_start = time.perf_counter()
    last_log = train_start
    data_iter = iter(loader)

    print(
        f"train_smollm2_135m_smoke: data={args.data_dir} "
        f"chunks={len(dataset):,} seq_len={seq_len} batch={args.batch_size} "
        f"max_steps={args.max_steps} device={device}",
        flush=True,
    )
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        step_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        labels = input_ids.clone()

        for group in optimizer.param_groups:
            group["lr"] = lr_at_step(step, args.lr, args.warmup_steps)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        tokens_seen += tokens_per_step
        running_loss += float(loss.detach())
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        step_tps = tokens_per_step / max(step_time, 1e-9)
        peak_tps = max(peak_tps, step_tps)

        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            now = time.perf_counter()
            elapsed = now - train_start
            avg_tps = tokens_seen / max(elapsed, 1e-9)
            recent = step_times[-args.log_every :]
            recent_tps = tokens_per_step / max(sum(recent) / len(recent), 1e-9)
            print(
                f"step={step:5d} loss={running_loss / args.log_every:.4f} "
                f"recent_tps={recent_tps:,.0f} avg_tps={avg_tps:,.0f} "
                f"step_ms={1000.0 * step_time:.1f} lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            running_loss = 0.0
            last_log = now

    wall_time = time.perf_counter() - train_start
    stats = ThroughputStats(
        steps=step,
        tokens_seen=tokens_seen,
        wall_time_s=wall_time,
        tokens_per_sec=tokens_seen / max(wall_time, 1e-9),
        steps_per_sec=step / max(wall_time, 1e-9),
        final_loss=float(loss.detach()),
        peak_tokens_per_sec=peak_tps,
        batch_size=args.batch_size,
        seq_len=seq_len,
        model=args.model_id,
        data_dir=str(args.data_dir),
        device=torch.cuda.get_device_name(device),
    )
    stats_path = args.output_dir / "throughput.json"
    stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    print(json.dumps(asdict(stats), indent=2), flush=True)

    if args.save_checkpoint:
        ckpt_dir = args.output_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_save.save_pretrained(ckpt_dir)
        print(f"saved checkpoint to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
