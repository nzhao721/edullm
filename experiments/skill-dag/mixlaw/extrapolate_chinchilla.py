#!/usr/bin/env python3
"""Extrapolate in-run task-loss curves to the Chinchilla-optimal token budget.

Uses the step law L(s) = L_inf + A * s^(-alpha) from fit_mixing_law.py, fitted on
the 6 in-run curve labels (ARC + MMLU). Chinchilla-optimal here means
tokens/param = 20 on the published DataDecide non-embedding parameter count.

Step-law fits use **in-run eval points only** (``task_loss.jsonl``, typically steps
120–1440). The post-hoc ``task_loss_final.json`` at step 1451 is **not** appended:
it uses a different eval protocol (full vs ``eval_subset_batches``) and often
disagrees with the last in-run point (notably on ``mmlu_stem``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit_mixing_law import extrapolate, fit_step_law
from mixlaw_common import CURVE_TASK_LOSS_LABELS, DATADECIDE_MODEL_SIZE, task_family, token_budget

CHINCHILLA_TOKENS_PER_PARAM = 20.0
DATA_NAME = "mixlaw_data.json"
OUT_NAME = "mixlaw_chinchilla_extrapolated.json"


def _curve_points(run: dict) -> dict[str, list[tuple[int, float]]]:
    """Per-family (step, loss) points from in-run ``task_loss.jsonl`` only."""
    per_label: dict[str, list[tuple[int, float]]] = {}
    for point in run["curve"]:
        step = int(point["step"])
        for label, value in point["task_loss_bpb"].items():
            per_label.setdefault(label, []).append((step, float(value)))

    per_family: dict[str, list[tuple[int, float]]] = {}
    for label, points in per_label.items():
        fam = task_family(label)
        per_family.setdefault(fam, []).extend(points)

    for fam, points in per_family.items():
        by_step: dict[int, float] = {}
        for step, value in points:
            by_step[step] = value
        per_family[fam] = sorted(by_step.items())
    return per_family


def extrapolate_runs(data: dict, target_step: int, seed: int) -> dict:
    _, pilot_steps, pilot_tokens = token_budget(5.0)
    _, chinchilla_steps, chinchilla_tokens = token_budget(CHINCHILLA_TOKENS_PER_PARAM)
    if target_step != chinchilla_steps:
        chinchilla_steps = target_step

    curve_families = sorted({task_family(label) for label in CURVE_TASK_LOSS_LABELS})
    run_reports = []

    for run in data["runs"]:
        per_family = _curve_points(run)
        families_out: dict[str, dict] = {}
        measured_curve = []
        chinchilla_curve = []

        for fam in curve_families:
            points = per_family.get(fam, [])
            if points:
                measured = float(points[-1][1])
                measured_step = int(points[-1][0])
            else:
                measured = float(run["task_loss_families"][fam])
                measured_step = pilot_steps
            entry: dict = {
                "measured_step": measured_step,
                "measured": measured,
                "curve_points": [{"step": s, "loss": v} for s, v in points],
            }
            if len(points) >= 5:
                steps = np.array([p[0] for p in points], dtype=float)
                losses = np.array([p[1] for p in points], dtype=float)
                law = fit_step_law(steps, losses, seed=seed + run["id"])
                if law is not None:
                    entry["step_law"] = law
                    entry["chinchilla_step"] = chinchilla_steps
                    entry["chinchilla"] = float(extrapolate(law, chinchilla_steps))
                    entry["delta"] = entry["chinchilla"] - measured
                    chinchilla_curve.append(entry["chinchilla"])
                else:
                    entry["chinchilla"] = None
                    entry["note"] = "step-law fit failed"
            else:
                entry["chinchilla"] = None
                entry["note"] = f"need >=5 curve points, got {len(points)}"
            measured_curve.append(measured)
            families_out[fam] = entry

        run_reports.append(
            {
                "id": run["id"],
                "run_name": run["run_name"],
                "tag": run["tag"],
                "weights": run["weights"],
                "macro_curve_6_measured": float(np.mean(measured_curve)),
                "macro_curve_6_chinchilla": float(np.mean(chinchilla_curve))
                if len(chinchilla_curve) == len(curve_families)
                else None,
                "families": families_out,
            }
        )

    return {
        "tokens_per_param_chinchilla": CHINCHILLA_TOKENS_PER_PARAM,
        "tokens_per_param_pilot": 5.0,
        "datadecide_model_size": DATADECIDE_MODEL_SIZE,
        "pilot_steps": pilot_steps,
        "pilot_tokens": pilot_tokens,
        "chinchilla_steps": chinchilla_steps,
        "chinchilla_tokens": chinchilla_tokens,
        "step_ratio": chinchilla_steps / pilot_steps,
        "curve_families": curve_families,
        "n_curve_families": len(curve_families),
        "runs": run_reports,
    }


def print_comparison(report: dict) -> None:
    rows = []
    for run in report["runs"]:
        if run["macro_curve_6_chinchilla"] is None:
            continue
        rows.append(
            (
                run["macro_curve_6_measured"],
                run["macro_curve_6_chinchilla"],
                run["run_name"],
                run["tag"],
            )
        )
    rows.sort(key=lambda r: r[1])

    print(
        f"\nChinchilla extrapolation: step {report['chinchilla_steps']} "
        f"({report['chinchilla_tokens']:,} tokens, tpp={report['tokens_per_param_chinchilla']})"
    )
    print(f"Pilot measured at: step {report['pilot_steps']} ({report['pilot_tokens']:,} tokens, tpp=5)")
    print(f"Step ratio: {report['step_ratio']:.2f}x\n")

    print("=== Rankings (macro over 6 curve families) ===")
    print(f"{'rank':>4}  {'mix':>6}  {'tag':<12}  {'meas@pilot':>11}  {'extrap@chin':>11}  {'delta':>8}")
    for i, (m6, c6, name, tag) in enumerate(rows, 1):
        print(f"{i:4d}  {name:>6}  {tag:<12}  {m6:11.4f}  {c6:11.4f}  {c6-m6:+8.4f}")

    best_meas = min(rows, key=lambda r: r[0])
    best_chin = min(rows, key=lambda r: r[1])
    print(f"\nBest by 6-family pilot macro:   {best_meas[3]} ({best_meas[0]:.4f})")
    print(f"Best by 6-family Chinchilla:    {best_chin[3]} ({best_chin[1]:.4f})")
    if best_meas[3] != best_chin[3]:
        print("Ranking changed after extrapolation.")
    else:
        print("Same best mixture before and after extrapolation (on curve families).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path(DATA_NAME))
    ap.add_argument("--out", type=Path, default=Path(OUT_NAME))
    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="Override target step (default: Chinchilla tpp=20 budget)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    _, default_step, _ = token_budget(CHINCHILLA_TOKENS_PER_PARAM)
    target_step = args.step if args.step is not None else default_step

    report = extrapolate_runs(data, target_step, seed=args.seed)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print_comparison(report)


if __name__ == "__main__":
    main()
