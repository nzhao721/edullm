#!/usr/bin/env python3
"""Multi-task RC task loss (bpb) and accuracy for HuggingFace causal LMs.

Supports ARC-Easy, ARC-Challenge, and HellaSwag (5-shot val by default).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class EvalMetrics:
    mean_bpb: float
    accuracy: float
    n_examples: int


@dataclass(frozen=True)
class TaskSpec:
    key: str
    bpb_label: str
    family: str
    loader: Callable[..., tuple[object, object]]  # (eval_ds, shot_ds)
    format_row: Callable[[dict], tuple[str, list[str], int]]
    # format_row -> (context, endings, gold_index)


def bpb_label_to_acc_label(bpb_label: str) -> str:
    return bpb_label.replace("_bpb", "_acc")


def continuation_bpb(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    continuation: str,
    device: torch.device,
) -> float:
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    cont_ids = full_ids[len(ctx_ids) :]
    if not cont_ids:
        raise ValueError("empty continuation")

    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    nll = 0.0
    for pos, token_id in enumerate(cont_ids):
        idx = len(ctx_ids) - 1 + pos
        nll += -float(log_probs[0, idx, token_id].detach())
    byte_len = len(continuation.encode("utf-8"))
    if byte_len <= 0:
        raise ValueError("continuation has zero utf-8 bytes")
    return (nll / math.log(2.0)) / byte_len


def ensure_leading_space(text: str) -> str:
    return text if text.startswith((" ", "\n")) else f" {text}"


# ---- ARC ----

def _arc_format_example(question: str, choices: dict[str, str], answer_key: str | None = None) -> tuple[str, str]:
    lines = [f"Question: {question.strip()}"]
    for label in sorted(choices):
        lines.append(f"{label}. {choices[label].strip()}")
    context = "\n".join(lines) + "\nAnswer:"
    continuation = f" {answer_key}" if answer_key is not None else ""
    return context, continuation


def _load_arc(config_name: str) -> tuple[object, object]:
    eval_ds = load_dataset("allenai/ai2_arc", config_name, split="validation")
    shot_ds = load_dataset("allenai/ai2_arc", config_name, split="train")
    return eval_ds, shot_ds


def _format_arc_row(row: dict) -> tuple[str, list[str], int]:
    choices = {k: v for k, v in zip(row["choices"]["label"], row["choices"]["text"])}
    labels = sorted(choices)
    ctx, _ = _arc_format_example(row["question"], choices)
    endings = [f" {lab}" for lab in labels]
    gold = labels.index(row["answerKey"])
    return ctx, endings, gold


def _arc_shot_text(row: dict) -> str:
    choices = {k: v for k, v in zip(row["choices"]["label"], row["choices"]["text"])}
    ctx, cont = _arc_format_example(row["question"], choices, row["answerKey"])
    return ctx + cont


# ---- HellaSwag ----

def _load_hellaswag() -> tuple[object, object]:
    eval_ds = load_dataset("Rowan/hellaswag", split="validation")
    shot_ds = load_dataset("Rowan/hellaswag", split="train")
    return eval_ds, shot_ds


def _format_hellaswag_row(row: dict) -> tuple[str, list[str], int]:
    activity = (row.get("activity_label") or "").strip()
    ctx = (row.get("ctx") or "").strip()
    context = f"{activity}: {ctx}" if activity else ctx
    endings = [ensure_leading_space(str(e).strip()) for e in row["endings"]]
    gold = int(row["label"])
    return context, endings, gold


def _hellaswag_shot_text(row: dict) -> str:
    context, endings, gold = _format_hellaswag_row(row)
    return context + endings[gold]


TASKS: dict[str, TaskSpec] = {
    "ARC-Easy": TaskSpec(
        key="ARC-Easy",
        bpb_label="arc_easy_val_rc_5shot_bpb",
        family="arc_easy",
        loader=lambda: _load_arc("ARC-Easy"),
        format_row=_format_arc_row,
    ),
    "ARC-Challenge": TaskSpec(
        key="ARC-Challenge",
        bpb_label="arc_challenge_val_rc_5shot_bpb",
        family="arc_challenge",
        loader=lambda: _load_arc("ARC-Challenge"),
        format_row=_format_arc_row,
    ),
    "HellaSwag": TaskSpec(
        key="HellaSwag",
        bpb_label="hellaswag_val_rc_5shot_bpb",
        family="hellaswag",
        loader=_load_hellaswag,
        format_row=_format_hellaswag_row,
    ),
}

# Back-compat aliases used by training code
ARC_CONFIGS = {
    "ARC-Easy": TASKS["ARC-Easy"].bpb_label,
    "ARC-Challenge": TASKS["ARC-Challenge"].bpb_label,
}


def build_few_shot_prefix(shot_rows: list[dict], task: TaskSpec, n_shot: int) -> str:
    if task.key.startswith("ARC"):
        blocks = [_arc_shot_text(row) for row in shot_rows[:n_shot]]
    else:
        blocks = [_hellaswag_shot_text(row) for row in shot_rows[:n_shot]]
    return "\n\n".join(blocks)


def eval_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: TaskSpec,
    *,
    n_shot: int,
    device: torch.device,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> EvalMetrics:
    eval_ds, shot_ds = task.loader()
    shot_ds = shot_ds.shuffle(seed=seed)
    prefix = build_few_shot_prefix(list(shot_ds), task, n_shot)

    total_bpb = 0.0
    correct = 0
    count = 0
    for i, row in enumerate(eval_ds):
        if i % world_size != rank:
            continue
        context, endings, gold = task.format_row(row)
        full_context = prefix + ("\n\n" if prefix else "") + context
        gold_cont = endings[gold]
        total_bpb += continuation_bpb(model, tokenizer, full_context, gold_cont, device)
        scores = {
            j: continuation_bpb(model, tokenizer, full_context, ending, device)
            for j, ending in enumerate(endings)
        }
        predicted = min(scores, key=scores.get)
        correct += int(predicted == gold)
        count += 1

    if world_size > 1:
        stats = torch.tensor(
            [total_bpb, float(correct), float(count)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_bpb = float(stats[0].item())
        correct = int(stats[1].item())
        count = int(stats[2].item())

    return EvalMetrics(
        mean_bpb=total_bpb / max(count, 1),
        accuracy=correct / max(count, 1),
        n_examples=count,
    )


def eval_split(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config_name: str,
    split: str,
    *,
    n_shot: int,
    device: torch.device,
    seed: int,
) -> EvalMetrics:
    """Back-compat wrapper for ARC configs used by train_smollm2_135m_ddp.py."""
    del split  # always validation for these tasks
    if config_name not in TASKS:
        raise KeyError(f"unknown task {config_name}; known={sorted(TASKS)}")
    return eval_task(model, tokenizer, TASKS[config_name], n_shot=n_shot, device=device, seed=seed)


def run_suite(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task_keys: list[str],
    *,
    n_shot: int,
    device: torch.device,
    seed: int,
    run_name: str,
    checkpoint: str | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    bpb_labels: dict[str, float] = {}
    acc_labels: dict[str, float] = {}
    families: dict[str, float] = {}
    accuracy_families: dict[str, float] = {}

    for key in task_keys:
        task = TASKS[key]
        metrics = eval_task(
            model,
            tokenizer,
            task,
            n_shot=n_shot,
            device=device,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        acc_label = bpb_label_to_acc_label(task.bpb_label)
        bpb_labels[task.bpb_label] = metrics.mean_bpb
        acc_labels[acc_label] = metrics.accuracy
        families[task.family] = metrics.mean_bpb
        accuracy_families[task.family] = metrics.accuracy
        if rank == 0:
            print(
                f"{task.bpb_label}\t{metrics.mean_bpb:.6f}\t{acc_label}\t{metrics.accuracy:.4f}\t"
                f"({metrics.n_examples} examples, world_size={world_size})",
                flush=True,
            )

    payload = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "metric": "task_loss_bpb",
        "definition": "-log2 p(gold continuation | context) / utf8 bytes of continuation",
        "n_shot": n_shot,
        "tasks": task_keys,
        "world_size": world_size,
        "labels": bpb_labels,
        "accuracy_labels": acc_labels,
        "task_families": families,
        "accuracy_families": accuracy_families,
        "macro_mean": sum(families.values()) / max(len(families), 1),
        "macro_mean_accuracy": sum(accuracy_families.values()) / max(len(accuracy_families), 1),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ARC + HellaSwag task loss/accuracy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", default="smollm2-135m")
    parser.add_argument(
        "--tokenizer-id",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Tokenizer source (checkpoints often omit tokenizer files).",
    )
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=sorted(TASKS),
        choices=sorted(TASKS),
    )
    # legacy alias
    parser.add_argument("--configs", nargs="*", default=None, choices=sorted(ARC_CONFIGS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for task-loss eval")

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    tok_src = args.tokenizer_id
    ckpt_tok = args.checkpoint / "tokenizer.json"
    if ckpt_tok.exists() or (args.checkpoint / "tokenizer_config.json").exists():
        tok_src = str(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    task_keys = list(args.configs) if args.configs else list(args.tasks)
    payload = run_suite(
        model,
        tokenizer,
        task_keys,
        n_shot=args.n_shot,
        device=device,
        seed=args.seed,
        run_name=args.run_name,
        checkpoint=str(args.checkpoint),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
