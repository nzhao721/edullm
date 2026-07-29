#!/usr/bin/env python3
"""Plot task-loss curves for BLADE, CE-RegMix, REL-EMA, and RefHQ."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODELS = {
    "refhq": {"label": "RefHQ (HQ reference)", "color": "#9333ea", "linestyle": "--"},
    "blade": {"label": "BLADE", "color": "#2563eb", "linestyle": "-"},
    "ce-regmix": {"label": "CE (RegMix)", "color": "#dc2626", "linestyle": "-"},
    "rel-ema": {"label": "REL-EMA", "color": "#16a34a", "linestyle": "-"},
}


def load_series() -> dict[str, list[tuple[int, float, float]]]:
    rows = json.loads((ROOT / "summary.json").read_text())
    series: dict[str, list[tuple[int, float, float]]] = {k: [] for k in MODELS}
    for row in rows:
        model = row["model"]
        if model not in MODELS:
            continue
        series[model].append((row["step"], row["mmlu_avg"], row["core7_avg"]))
    for model in series:
        series[model].sort(key=lambda x: x[0])
    return series


def main() -> None:
    series = load_series()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    metrics = [
        ("mmlu_avg", axes[0]),
        ("core7_avg", axes[1]),
    ]
    idx = {"mmlu_avg": 1, "core7_avg": 2}

    # Plot RefHQ first so other lines draw on top where they overlap.
    plot_order = ["refhq", "blade", "ce-regmix", "rel-ema"]
    for model in plot_order:
        style = MODELS[model]
        points = series[model]
        steps = [p[0] for p in points]
        for key, ax in metrics:
            values = [p[idx[key]] for p in points]
            ax.plot(
                steps,
                values,
                marker="o",
                markersize=4,
                linewidth=2,
                linestyle=style.get("linestyle", "-"),
                label=style["label"],
                color=style["color"],
            )

    for ax in axes:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Task loss (BPB, lower is better)")
        ax.legend(loc="upper right", frameon=True)

    axes[0].set_title("MMLU")
    axes[1].set_title("Core-7 benchmarks")
    fig.suptitle(
        "Downstream task loss vs training step (OLMo2-370M)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    out_png = ROOT / "loss_curves.png"
    out_svg = ROOT / "loss_curves.svg"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
