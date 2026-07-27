#!/usr/bin/env python3
"""Summarize and plot training + per-checkpoint validation losses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_train_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    # Keep only monotonically increasing steps (ignore restarts appended to same file).
    best: dict[int, dict] = {}
    for row in rows:
        step = int(row["step"])
        if step not in best or row["elapsed_sec"] >= best[step]["elapsed_sec"]:
            best[step] = row
    return [best[k] for k in sorted(best)]


def load_checkpoint_losses(token_loss_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(token_loss_dir.glob("checkpoint_*.npz")):
        data = np.load(path)
        losses = data["losses"].astype(np.float64)
        rows.append(
            {
                "checkpoint_idx": int(data["checkpoint_idx"]),
                "global_step": int(data["global_step"]),
                "tokens_trained": int(data["tokens_trained"]),
                "path": str(path),
                "num_tokens": int(losses.size),
                "mean": float(losses.mean()),
                "median": float(np.median(losses)),
                "p90": float(np.percentile(losses, 90)),
                "p99": float(np.percentile(losses, 99)),
            }
        )
    return rows


def print_table(title: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no data)")
        return
    header = "  ".join(f"{label:>14}" for _, label in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{row[key]:>14}" for key, _ in columns))


def save_plots(out_dir: Path, run_name: str, train_rows: list[dict], ckpt_rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipped PNG plots (table + CSV still written).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if train_rows:
        tokens = [r["tokens_trained"] for r in train_rows]
        loss = [r["loss"] for r in train_rows]
        axes[0].plot(tokens, loss, linewidth=1.0)
        axes[0].set_xlabel("tokens trained")
        axes[0].set_ylabel("train loss")
        axes[0].set_title(f"{run_name}: training loss")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_title("no train_metrics.jsonl")

    if ckpt_rows:
        x = [r["tokens_trained"] for r in ckpt_rows]
        mean = [r["mean"] for r in ckpt_rows]
        median = [r["median"] for r in ckpt_rows]
        axes[1].plot(x, mean, marker="o", label="mean")
        axes[1].plot(x, median, marker=".", label="median")
        axes[1].set_xlabel("tokens trained")
        axes[1].set_ylabel("val token loss")
        axes[1].set_title(f"{run_name}: fixed-val checkpoint curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("no token_losses/*.npz")

    fig.tight_layout()
    plot_path = out_dir / "loss_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote plot: {plot_path}")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[col]) for col in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote CSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="e.g. outputs/pretrain_slimpajama")
    parser.add_argument("--no-plot", action="store_true", help="skip PNG generation")
    args = parser.parse_args()

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not out_dir.exists():
        raise SystemExit(f"Missing output dir: {out_dir}")

    run_name = out_dir.name
    manifest_path = out_dir / "run_manifest.json"
    summary_path = out_dir / "summary.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_name = manifest.get("run_name", run_name)

    print(f"Run: {run_name}")
    print(f"Dir: {out_dir}")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"Finished: {summary.get('finished_tokens_trained', '?')} tokens, "
            f"{summary.get('finished_steps', '?')} steps, "
            f"{summary.get('elapsed_sec', '?')} sec"
        )

    train_rows = load_train_metrics(out_dir / "train_metrics.jsonl")
    ckpt_rows = load_checkpoint_losses(out_dir / "token_losses")

    print_table(
        "Training loss (from train_metrics.jsonl)",
        train_rows[-10:],
        [
            ("step", "step"),
            ("tokens_trained", "tokens"),
            ("loss", "loss"),
            ("lr", "lr"),
        ],
    )
    print_table(
        "Checkpoint val loss (from token_losses/*.npz)",
        ckpt_rows,
        [
            ("checkpoint_idx", "ckpt"),
            ("tokens_trained", "tokens"),
            ("mean", "mean"),
            ("median", "median"),
            ("p90", "p90"),
        ],
    )

    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        analysis_dir / "train_metrics_clean.csv",
        train_rows,
        ["step", "tokens_trained", "loss", "lr", "elapsed_sec"],
    )
    write_csv(
        analysis_dir / "checkpoint_val_loss.csv",
        ckpt_rows,
        [
            "checkpoint_idx",
            "tokens_trained",
            "global_step",
            "num_tokens",
            "mean",
            "median",
            "p90",
            "p99",
        ],
    )

    if not args.no_plot:
        save_plots(analysis_dir, run_name, train_rows, ckpt_rows)


if __name__ == "__main__":
    main()
