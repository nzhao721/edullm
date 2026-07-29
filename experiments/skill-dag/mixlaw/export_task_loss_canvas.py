#!/usr/bin/env python3
"""Export mixlaw-task-loss-curves canvas data to static PNG images."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CANVAS = Path.home() / ".cursor/projects/c-alpha-ai-edullm/canvases/mixlaw-task-loss-curves.canvas.tsx"
OUT_DIR = ROOT / "task_loss_canvas_exports"

STEPS = [120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440]
CURVE_LABELS = [
    "arc_challenge_val_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
]
SHORT = {
    "arc_challenge_val_rc_5shot_bpb": "ARC challenge",
    "arc_easy_val_rc_5shot_bpb": "ARC easy",
    "mmlu_stem_val_rc_5shot_bpb": "MMLU STEM",
    "mmlu_humanities_val_rc_5shot_bpb": "MMLU humanities",
    "mmlu_social_sciences_val_rc_5shot_bpb": "MMLU social sci",
    "mmlu_other_val_rc_5shot_bpb": "MMLU other",
}


def load_mixes_from_canvas(path: Path) -> dict[str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    mixes: dict[str, list[dict]] = {}
    for match in re.finditer(r"(mix\d+):\s*(\[.*?\])(?=,\s*mix|\s*\};)", text, re.DOTALL):
        mixes[match.group(1)] = json.loads(match.group(2))
    if not mixes:
        raise RuntimeError(f"No MIXES data found in {path}")
    return mixes


def macro_mean(curve: list[dict]) -> list[float]:
    return [
        sum(p["task_loss_bpb"][label] for label in CURVE_LABELS) / len(CURVE_LABELS)
        for p in curve
    ]


def plot_macro(mixes: dict[str, list[dict]], out: Path) -> None:
    order = sorted(m for m in mixes if len(mixes[m]) == len(STEPS))
    cmap = plt.get_cmap("tab20", max(len(order), 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, mix in enumerate(order):
        ax.plot(STEPS, macro_mean(mixes[mix]), linewidth=1.2, color=cmap(i), label=mix, alpha=0.85)

    ax.set_title("Mixlaw in-run task loss — macro mean (6-label average)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Task loss (bpb, lower is better)")
    ax.set_ylim(1.4, 4.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=6, fontsize=7, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_ranking(mixes: dict[str, list[dict]], out: Path) -> None:
    order = sorted(m for m in mixes if len(mixes[m]) == len(STEPS))
    finals = sorted(((mix, macro_mean(mixes[mix])[-1]) for mix in order), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(12, 4))
    names = [f[0] for f in finals]
    values = [f[1] for f in finals]
    ax.bar(range(len(names)), values, color="#2563eb", width=0.75)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Final macro ranking at step 1440")
    ax.set_ylabel("Task loss (bpb)")
    ax.set_ylim(1.9, 2.4)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_by_label(mixes: dict[str, list[dict]], out: Path) -> None:
    order = sorted(m for m in mixes if len(mixes[m]) == len(STEPS))
    cmap = plt.get_cmap("tab20", max(len(order), 1))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    for ax, label in zip(axes, CURVE_LABELS, strict=True):
        for i, mix in enumerate(order):
            ys = [p["task_loss_bpb"][label] for p in mixes[mix]]
            ax.plot(STEPS, ys, linewidth=1.0, color=cmap(i), alpha=0.8)
        ax.set_title(SHORT[label])
        ax.set_ylabel("bpb")
        ax.set_ylim(1.4, 4.0)
        ax.grid(True, alpha=0.3)
    for ax in axes[-2:]:
        ax.set_xlabel("Training step")
    fig.suptitle("Per-label task loss — all mixes", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary_panel(mixes: dict[str, list[dict]], out: Path) -> None:
    """Single image with macro curves + final ranking (canvas top panels)."""
    order = sorted(m for m in mixes if len(mixes[m]) == len(STEPS))
    cmap = plt.get_cmap("tab20", max(len(order), 1))
    finals = sorted(((mix, macro_mean(mixes[mix])[-1]) for mix in order), key=lambda x: x[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    for i, mix in enumerate(order):
        ax1.plot(STEPS, macro_mean(mixes[mix]), linewidth=1.2, color=cmap(i), label=mix, alpha=0.85)
    ax1.set_title("Mixlaw in-run task loss curves (DataDecide 60M pilot)")
    ax1.set_ylabel("Macro mean bpb")
    ax1.set_ylim(1.4, 4.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=8, fontsize=6, loc="upper right")

    names = [f[0] for f in finals]
    values = [f[1] for f in finals]
    ax2.bar(range(len(names)), values, color="#2563eb", width=0.75)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Final macro ranking (step 1440)")
    ax2.set_ylabel("bpb")
    ax2.set_ylim(1.9, 2.4)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlabel("Mixture")

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not CANVAS.exists():
        raise SystemExit(f"Canvas not found: {CANVAS}")

    mixes = load_mixes_from_canvas(CANVAS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = OUT_DIR / "mixlaw_task_loss_summary.png"
    macro = OUT_DIR / "mixlaw_task_loss_macro.png"
    ranking = OUT_DIR / "mixlaw_task_loss_final_ranking.png"
    by_label = OUT_DIR / "mixlaw_task_loss_by_label.png"

    plot_summary_panel(mixes, summary)
    plot_macro(mixes, macro)
    plot_final_ranking(mixes, ranking)
    plot_by_label(mixes, by_label)

    print(f"wrote {summary}")
    print(f"wrote {macro}")
    print(f"wrote {ranking}")
    print(f"wrote {by_label}")


if __name__ == "__main__":
    main()
