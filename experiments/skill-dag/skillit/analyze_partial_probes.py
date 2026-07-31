#!/usr/bin/env python3
"""Plot task-loss curves and compute partial offline A from finished Skill-It probes."""
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
from mixlaw_common import CURVE_FAMILIES, CURVE_TASK_LOSS_LABELS, DOMAINS, task_family  # noqa: E402
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    offline_A_partial,
    regmix_family_losses_from_fit,
    regmix_family_losses_measured,
)

FINISHED = ("probe_dclm", "probe_arxiv", "probe_starcoder")
LOG_MAP = {
    "probe_dclm": "probe-1668442_1.out",
    "probe_arxiv": "probe-1668442_2.out",
    "probe_starcoder": "probe-1668442_3.out",
}
STEP_RE = re.compile(r"\[step=(\d+)/\d+")
EVAL_RE = re.compile(r"eval/downstream_bpb/([a-z0-9_]+)_bpb=([0-9.]+)")


def parse_log_curves(log_path: Path) -> list[dict]:
    """Recover in-run eval points from Slurm stdout (multiline eval blocks)."""
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


def load_probe_run(runs_dir: Path, run_name: str, logs_dir: Path | None) -> dict:
    progress = runs_dir / run_name / "progress"
    final_path = progress / "task_loss_final.json"
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    labels = {k: float(v) for k, v in payload["labels"].items() if k in CURVE_TASK_LOSS_LABELS}
    families = {k: float(v) for k, v in payload["task_families"].items() if k in CURVE_FAMILIES}
    curve_path = progress / "task_loss.jsonl"
    if curve_path.is_file():
        curve = [json.loads(ln) for ln in curve_path.read_text().splitlines() if ln.strip()]
    elif logs_dir is not None:
        curve = parse_log_curves(logs_dir / LOG_MAP.get(run_name, f"{run_name}.out"))
    else:
        curve = []
    mix_id = {
        "probe_dclm": 0,
        "probe_arxiv": 1,
        "probe_starcoder": 2,
        "probe_pes2o": 3,
        "probe_open-web-math": 4,
        "probe_algebraic-stack": 5,
        "probe_wiki": 6,
    }.get(run_name, -1)
    return {
        "id": mix_id,
        "tag": run_name,
        "run_name": run_name,
        "weights": [],
        "task_loss_labels": labels,
        "task_loss_families": families,
        "macro_mean": float(payload.get("macro_mean", np.mean(list(families.values())))),
        "curve": curve,
    }


def collect_partial(runs_dir: Path, logs_dir: Path | None) -> dict:
    runs = [load_probe_run(runs_dir, name, logs_dir) for name in FINISHED]
    return {
        "domain_order": list(DOMAINS),
        "curve_families": list(CURVE_FAMILIES),
        "runs": runs,
        "partial": True,
        "finished_probes": list(FINISHED),
    }


def partial_A_measured(data: dict) -> tuple[np.ndarray, dict]:
    """A_ij at measured step 1451 for available one-hot probes only."""
    by_name = {r["run_name"]: r for r in data["runs"]}
    L_reg = regmix_family_losses_measured()
    probe_losses = {
        run_name: by_name[run_name]["task_loss_families"]
        for run_name in by_name
        if run_name.startswith("probe_")
    }
    return offline_A_partial(
        probe_losses,
        L_reg,
        domains=DOMAINS,
        families=CURVE_FAMILIES,
        reference_label="regmix",
        formula="A_ij = max(0, L_j(regmix) - L_i_j) at measured step 1451; "
        "L_j(regmix) from mixlaw pilot mix01",
    )


def partial_A_chinchilla(data: dict, seed: int = 0) -> tuple[np.ndarray, dict, dict]:
    """Extrapolate available probes; NaN for domains without one-hot runs."""
    fit = load_fit_json(default_mixlaw_fit_path())
    L_reg = regmix_family_losses_from_fit(fit, domains=DOMAINS, families=CURVE_FAMILIES)
    report = extrapolate_runs(data, target_step=5806, seed=seed)
    by_name = {r["run_name"]: r for r in report["runs"]}
    probe_losses = {
        run_name: {
            fam: float(by_name[run_name]["families"][fam]["chinchilla"])
            for fam in CURVE_FAMILIES
        }
        for run_name in by_name
        if run_name.startswith("probe_")
    }
    A, detail = offline_A_partial(
        probe_losses,
        L_reg,
        domains=DOMAINS,
        families=CURVE_FAMILIES,
        reference_label="regmix",
        formula="A_ij = max(0, L_j(regmix) - L_i_j) at Chinchilla step 5806 (partial); "
        "L_j(regmix) from mixlaw_fit_chinchilla.json",
    )
    detail["chinchilla_losses"] = {"regmix": L_reg, **probe_losses}
    detail["fit_json"] = str(default_mixlaw_fit_path())
    return A, detail, report


def macro_curve(curve: list[dict]) -> tuple[list[int], list[float]]:
    steps, vals = [], []
    for pt in curve:
        losses = pt.get("task_loss_bpb") or {}
        fam_vals = []
        for label, v in losses.items():
            fam = task_family(label)
            if fam in CURVE_FAMILIES:
                fam_vals.append(float(v))
        if fam_vals:
            steps.append(int(pt["step"]))
            vals.append(float(np.mean(fam_vals)))
    return steps, vals


def plot_curves(data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")
    for idx, run in enumerate(data["runs"]):
        steps, macro = macro_curve(run["curve"])
        if steps:
            ax.plot(steps, macro, marker="o", ms=3, lw=1.5, label=run["run_name"], color=cmap(idx))
        ax.scatter(
            [1451],
            [run["macro_mean"]],
            s=60,
            marker="*",
            color=cmap(idx),
            zorder=5,
            label=f"{run['run_name']} final",
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("macro mean task loss (bpb)")
    ax.set_title("Skill-It probes — in-run curve (6-label macro mean)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "task_loss_curves_partial.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Per-family panel for uni vs one-hots at final only
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, fam in zip(axes.flat, CURVE_FAMILIES):
        xs, ys, labels = [], [], []
        for run in data["runs"]:
            xs.append(run["run_name"].replace("probe_", ""))
            ys.append(run["task_loss_families"][fam])
        ax.bar(xs, ys, color=[plt.get_cmap("tab10")(i) for i in range(len(xs))])
        ax.set_title(fam)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Final task loss by probe (step 1451)")
    fig.tight_layout()
    path2 = out_dir / "task_loss_final_by_family.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {path2}")


def print_matrix(A: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    hdr = " ".join(f"{f[:10]:>10}" for f in CURVE_FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        row = " ".join(f"{A[i, j]:10.4f}" if np.isfinite(A[i, j]) else f"{'pending':>10}" for j in range(len(CURVE_FAMILIES)))
        print(f"{d:<18} {row}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--logs-dir", type=Path, default=None, help="Slurm stdout logs for in-run curves")
    ap.add_argument("--out-dir", type=Path, default=_SCRIPT / "artifacts" / "partial_wave1")
    args = ap.parse_args()

    data = collect_partial(args.runs_dir, args.logs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "probe_data_partial.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    plot_curves(data, args.out_dir)

    A_meas, meas_detail = partial_A_measured(data)
    np.save(args.out_dir / "A_offline_partial_measured.npy", A_meas)
    (args.out_dir / "A_offline_partial_measured.json").write_text(
        json.dumps({**meas_detail, "A": A_meas.tolist(), "domain_order": DOMAINS, "family_order": CURVE_FAMILIES}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print_matrix(A_meas, "Partial A vs RegMix (measured @ step 1451)")

    try:
        A_chin, chin_detail, report = partial_A_chinchilla(data)
        np.save(args.out_dir / "A_offline_partial_chinchilla.npy", A_chin)
        (args.out_dir / "A_offline_partial_chinchilla.json").write_text(
            json.dumps(
                {**chin_detail, "A": [[None if not np.isfinite(x) else x for x in row] for row in A_chin.tolist()],
                 "domain_order": DOMAINS,
                 "family_order": CURVE_FAMILIES},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "probe_chinchilla_partial.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print_matrix(A_chin, "Partial A vs RegMix (Chinchilla extrapolated @ step 5806)")
    except Exception as exc:
        print(f"Chinchilla extrapolation skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
