#!/usr/bin/env python3
"""Plot Chinchilla-extrapolated probe task-loss curves and build offline A vs RegMix."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_MIXLAW = _SCRIPT.parent / "mixlaw"
for p in (_MIXLAW, _SCRIPT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extrapolate_chinchilla import extrapolate_runs  # noqa: E402
from fit_mixing_law import extrapolate  # noqa: E402
from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, DOMAINS, task_family  # noqa: E402
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    offline_A_from_extrapolated,
    regmix_family_losses_from_fit,
)

CHINCHILLA_STEP = 5806
PILOT_STEP = 1451

ONEHOT_PROBES: tuple[str, ...] = tuple(f"probe_{d}" for d in DOMAINS)
LOG_MAP = {
    "probe_dclm": "probe-1668442_1.out",
    "probe_arxiv": "probe-1668442_2.out",
    "probe_starcoder": "probe-1668442_3.out",
    "probe_pes2o": "probe-1668442_4.out",
    "probe_open-web-math": "probe-1668442_5.out",
    "probe_algebraic-stack": "probe-1668442_6.out",
    "probe_wiki": "probe-1668442_7.out",
}
STEP_RE = re.compile(r"\[step=(\d+)/\d+")
EVAL_RE = re.compile(r"eval/downstream_bpb/([a-z0-9_]+)_bpb=([0-9.]+)")


def parse_log_curves(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    curves: list[dict] = []
    current_step: int | None = None
    pending: dict[str, float] = {}

    def flush() -> None:
        nonlocal pending, current_step
        if current_step is not None and pending:
            curves.append({"step": current_step, "task_loss_bpb": dict(pending)})
        pending = {}

    for line in lines:
        m = STEP_RE.search(line)
        if m:
            flush()
            current_step = int(m.group(1))
            continue
        for label, val in EVAL_RE.findall(line):
            key = label if label.endswith("_bpb") else f"{label}_bpb"
            if key in CURVE_TASK_LOSS_LABELS:
                pending[key] = float(val)
    flush()
    return curves


def load_probe_run(runs_dir: Path, run_name: str, logs_dir: Path) -> dict:
    progress = runs_dir / run_name / "progress"
    final_path = progress / "task_loss_final.json"
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    labels = {k: float(v) for k, v in payload["labels"].items() if k in CURVE_TASK_LOSS_LABELS}
    fam_src = payload.get("task_families") or {}
    families = {k: float(v) for k, v in fam_src.items() if k in CURVE_FAMILIES}
    curve_path = progress / "task_loss.jsonl"
    if curve_path.is_file():
        curve = [json.loads(ln) for ln in curve_path.read_text().splitlines() if ln.strip()]
    else:
        curve = parse_log_curves(logs_dir / LOG_MAP[run_name])
    mix_id = ONEHOT_PROBES.index(run_name)
    return {
        "id": mix_id,
        "tag": run_name.replace("probe_", ""),
        "run_name": run_name,
        "weights": [],
        "task_loss_labels": labels,
        "task_loss_families": families,
        "macro_mean": float(payload.get("macro_mean", np.mean(list(families.values())))),
        "curve": curve,
    }


def family_series(curve: list[dict], family: str) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for pt in curve:
        losses = pt.get("task_loss_bpb") or {}
        for label, v in losses.items():
            if task_family(label) == family:
                steps.append(int(pt["step"]))
                vals.append(float(v))
                break
    return steps, vals


def plot_chinchilla_curves(report: dict, L_reg: dict[str, float], out_path: Path) -> None:
    cmap = plt.get_cmap("tab10")
    colors = {run["run_name"]: cmap(i) for i, run in enumerate(report["runs"])}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    chin_step = int(report["chinchilla_steps"])
    eval_steps = np.arange(120, PILOT_STEP + 1, 120)
    smooth = np.linspace(120, chin_step, 200)

    for ax, fam in zip(axes.flat, CURVE_FAMILIES):
        for run in report["runs"]:
            dom = run["run_name"].replace("probe_", "")
            entry = run["families"][fam]
            pts = entry.get("curve_points") or []
            if not pts:
                continue
            xs = [int(p["step"]) for p in pts]
            ys = [float(p["loss"]) for p in pts]
            color = colors[run["run_name"]]
            ax.plot(xs, ys, "o-", ms=3, lw=1.2, color=color, label=dom)
            law = entry.get("step_law")
            chin = entry.get("chinchilla")
            if law is not None and chin is not None:
                ext_y = [extrapolate(law, int(s)) for s in smooth]
                ax.plot(smooth, ext_y, "--", lw=1.0, color=color, alpha=0.75)
                ax.scatter([chin_step], [chin], marker="*", s=70, color=color, zorder=5)
        ax.axhline(L_reg[fam], color="0.35", ls=":", lw=1.2, alpha=0.9)
        ax.set_title(fam.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("task loss (bpb)")

    for ax in axes[1]:
        ax.set_xlabel("training step")
    axes[0, 0].set_xlim(0, chin_step * 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.suptitle(
        "Skill-It one-hot probes: measured evals (solid) + Chinchilla extrapolation "
        f"(dashed → step {chin_step}); dotted = RegMix reference",
        fontsize=11,
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_chinchilla_macro(report: dict, L_reg: dict[str, float], out_path: Path) -> None:
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 5))
    reg_macro = float(np.mean([L_reg[f] for f in CURVE_FAMILIES]))
    chin_step = int(report["chinchilla_steps"])
    smooth = np.linspace(120, chin_step, 200)

    for i, run in enumerate(report["runs"]):
        dom = run["run_name"].replace("probe_", "")
        color = cmap(i)
        macro_pts_x, macro_pts_y = [], []
        by_step: dict[int, list[float]] = {}
        for fam in CURVE_FAMILIES:
            for p in run["families"][fam].get("curve_points") or []:
                by_step.setdefault(int(p["step"]), []).append(float(p["loss"]))
        for step in sorted(by_step):
            macro_pts_x.append(step)
            macro_pts_y.append(float(np.mean(by_step[step])))
        if macro_pts_x:
            ax.plot(macro_pts_x, macro_pts_y, "o-", ms=3, lw=1.2, color=color, label=dom)
        chin_vals = [
            float(run["families"][fam]["chinchilla"])
            for fam in CURVE_FAMILIES
            if run["families"][fam].get("chinchilla") is not None
        ]
        if len(chin_vals) == len(CURVE_FAMILIES):
            chin_macro = float(np.mean(chin_vals))
            ext_y = []
            for s in smooth:
                vals = []
                for fam in CURVE_FAMILIES:
                    law = run["families"][fam].get("step_law")
                    if law is not None:
                        vals.append(extrapolate(law, int(s)))
                if vals:
                    ext_y.append(float(np.mean(vals)))
            if ext_y:
                ax.plot(smooth[: len(ext_y)], ext_y, "--", lw=1.0, color=color, alpha=0.75)
            ax.scatter([chin_step], [chin_macro], marker="*", s=70, color=color, zorder=5)

    ax.axhline(reg_macro, color="0.35", ls=":", lw=1.2, label="RegMix (chin)")
    ax.set_xlabel("training step")
    ax.set_ylabel("macro mean task loss (bpb)")
    ax.set_title(f"Macro mean over 6 families (Chinchilla @ step {chin_step})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def print_A(A: np.ndarray, L_reg: dict[str, float]) -> None:
    print("\nL_j(RegMix) @ Chinchilla:")
    for fam in CURVE_FAMILIES:
        print(f"  {fam:22s} {L_reg[fam]:.4f}")
    print("\nA_ij = max(0, L_j(RegMix) - L_j(i)) @ Chinchilla step 5806")
    hdr = " ".join(f"{f[:10]:>10}" for f in CURVE_FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, dom in enumerate(DOMAINS):
        row = " ".join(f"{A[i, j]:10.4f}" for j in range(len(CURVE_FAMILIES)))
        print(f"{dom:<18} {row}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--logs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "domain_order": list(DOMAINS),
        "curve_families": list(CURVE_FAMILIES),
        "runs": [load_probe_run(args.runs_dir, name, args.logs_dir) for name in ONEHOT_PROBES],
    }
    (args.out_dir / "probe_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    report = extrapolate_runs(data, CHINCHILLA_STEP, seed=args.seed)
    (args.out_dir / "probe_chinchilla_extrapolated.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    fit = load_fit_json(default_mixlaw_fit_path())
    L_reg = regmix_family_losses_from_fit(fit, domains=DOMAINS, families=CURVE_FAMILIES)
    A, detail = offline_A_from_extrapolated(
        report, L_reg, domains=DOMAINS, families=CURVE_FAMILIES, chinchilla_step=CHINCHILLA_STEP
    )
    detail["fit_json"] = str(default_mixlaw_fit_path())
    np.save(args.out_dir / "A_offline.npy", A)
    (args.out_dir / "A_offline.json").write_text(
        json.dumps({**detail, "shape": list(A.shape)}, indent=2) + "\n", encoding="utf-8"
    )

    plot_chinchilla_curves(report, L_reg, args.out_dir / "task_loss_chinchilla_by_family.png")
    plot_chinchilla_macro(report, L_reg, args.out_dir / "task_loss_chinchilla_macro.png")
    print_A(A, L_reg)


if __name__ == "__main__":
    main()
