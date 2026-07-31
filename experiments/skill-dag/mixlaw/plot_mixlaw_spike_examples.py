#!/usr/bin/env python3
"""Plot mixlaw in-run curves with step-1451 final anchor (spike visible)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "pilot_runs"
OUT = ROOT / "task_loss_canvas_exports"

CURVE_LABELS = [
    "arc_challenge_val_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
]
FAMILIES = [
    "arc_challenge",
    "arc_easy",
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_other",
]
LABEL_TO_FAM = {lab: fam for lab, fam in zip(CURVE_LABELS, FAMILIES)}
EXAMPLES = ("mix01", "mix07", "mix14")


def load_curve_with_anchor(mix: str) -> dict[str, list[tuple[int, float]]]:
    progress = PILOT / mix / "progress"
    points: dict[str, list[tuple[int, float]]] = {fam: [] for fam in FAMILIES}
    for line in (progress / "task_loss.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        for lab, val in row["task_loss_bpb"].items():
            if lab in LABEL_TO_FAM:
                points[LABEL_TO_FAM[lab]].append((step, float(val)))
    final = json.loads((progress / "task_loss_final.json").read_text(encoding="utf-8"))
    fam_src = final.get("task_families") or {}
    if fam_src:
        for fam in FAMILIES:
            points[fam].append((1451, float(fam_src[fam])))
    else:
        for lab, val in final["labels"].items():
            if lab in LABEL_TO_FAM:
                points[LABEL_TO_FAM[lab]].append((1451, float(val)))
    for fam in FAMILIES:
        points[fam].sort(key=lambda x: x[0])
    return points


def plot_examples() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(13, 3.2 * len(EXAMPLES)), sharex=True)
    if len(EXAMPLES) == 1:
        axes = np.array([axes])

    focus = ("arc_challenge", "mmlu_stem", "mmlu_other")
    for row, mix in enumerate(EXAMPLES):
        curves = load_curve_with_anchor(mix)
        for col, fam in enumerate(focus):
            ax = axes[row, col]
            xs, ys = zip(*curves[fam])
            ax.plot(xs[:-1], ys[:-1], "o-", ms=4, lw=1.5, color="#2563eb", label="in-run")
            ax.plot(xs[-2:], ys[-2:], "o-", ms=6, lw=2.0, color="#dc2626", label="1440→1451")
            ax.scatter([1451], [ys[-1]], s=80, marker="*", color="#dc2626", zorder=5)
            if row == 0:
                ax.set_title(fam.replace("_", " "))
            if col == 0:
                ax.set_ylabel(f"{mix}\nbpb")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(100, 1470)
            if row == len(EXAMPLES) - 1:
                ax.set_xlabel("training step")
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Mixlaw pilots: smooth in-run jsonl (blue) + final-anchor spike at 1451 (red)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out = OUT / "mixlaw_spike_examples.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_macro_panel() -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for mix in EXAMPLES:
        curves = load_curve_with_anchor(mix)
        macro: list[tuple[int, float]] = []
        by_step: dict[int, list[float]] = {}
        for fam in FAMILIES:
            for step, val in curves[fam]:
                by_step.setdefault(step, []).append(val)
        for step in sorted(by_step):
            macro.append((step, float(np.mean(by_step[step]))))
        xs, ys = zip(*macro)
        ax.plot(xs[:-1], ys[:-1], "o-", ms=3, lw=1.3, label=mix)
        ax.plot(xs[-2:], ys[-2:], "o-", ms=5, lw=2.0)
        ax.scatter([1451], [ys[-1]], s=70, marker="*", zorder=5)
    ax.set_xlabel("training step")
    ax.set_ylabel("macro mean (6 families)")
    ax.set_title("Mixlaw macro mean with 1440→1451 anchor spike")
    ax.set_xlim(100, 1470)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT / "mixlaw_spike_macro_examples.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = plot_examples()
    p2 = plot_macro_panel()
    print(f"wrote {p1}")
    print(f"wrote {p2}")
