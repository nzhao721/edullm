#!/usr/bin/env python3
"""Build offline Skill-It adjacency A from 8 DataDecide-60M probe runs.

Pipeline:
  1. Collect per-probe ``task_loss.jsonl`` + ``task_loss_final.json`` (same schema
     as mixlaw ``fit_mixing_law.py collect``).
  2. Chinchilla-extrapolate each curve family to step 5806 (tpp=20) via
     ``mixlaw/extrapolate_chinchilla.py`` logic.
  3. ``A_ij = max(0, L_uni_j - L_i_j)`` for domains i and families j.
  4. Write ``artifacts/A_offline.npy`` plus named JSON, then publish them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SKILLIT = Path(__file__).resolve().parent
_MIXLAW = _SKILLIT.parent / "mixlaw"
_TS_ROOT = _SKILLIT.parents[1] / "token-selection"
for p in (_MIXLAW, _SKILLIT, _TS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from extrapolate_chinchilla import extrapolate_runs  # noqa: E402
from mixlaw_common import (  # noqa: E402
    CURVE_FAMILIES,
    CURVE_TASK_LOSS_LABELS,
    DOMAINS,
    load_mixtures,
    macro_curve,
    task_family,
    token_budget,
)
from skillit_math import A_to_named_dict  # noqa: E402

UNI_RUN = "probe_uni"
CHINCHILLA_STEP = 5806  # token_budget(20) → 5806
ARTIFACTS_S3_URI = "s3://edullm-checkpoints/skillit/artifacts"


def collect_probe_runs(runs_dir: Path, mixtures_json: Path) -> dict:
    """Gather probe progress into a mixlaw_data-compatible payload."""
    mixtures = {m.id: m for m in load_mixtures(mixtures_json)}
    runs = []
    missing = []
    for mix_id, mix in sorted(mixtures.items()):
        progress = runs_dir / mix.run_name / "progress"
        final = progress / "task_loss_final.json"
        if not final.is_file():
            # Also accept flat layout: runs_dir/<run>/task_loss_final.json
            alt = runs_dir / mix.run_name / "task_loss_final.json"
            if alt.is_file():
                final = alt
                progress = runs_dir / mix.run_name
            else:
                missing.append(mix.run_name)
                continue

        payload = json.loads(final.read_text(encoding="utf-8"))
        label_src = payload.get("labels") or payload.get("task_loss_bpb") or {}
        labels = {
            k: float(v)
            for k, v in label_src.items()
            if k in CURVE_TASK_LOSS_LABELS
        }
        fam_src = payload.get("task_families") or {}
        if fam_src:
            families = {
                k: float(v) for k, v in fam_src.items() if k in CURVE_FAMILIES
            }
        else:
            families = {}
            for label, value in labels.items():
                fam = task_family(label)
                if fam in CURVE_FAMILIES:
                    families[fam] = float(value)
        if set(families) != set(CURVE_FAMILIES):
            raise SystemExit(
                f"{mix.run_name}: expected curve families {CURVE_FAMILIES}, "
                f"got {sorted(families)}"
            )

        curve = []
        curve_path = progress / "task_loss.jsonl"
        if curve_path.is_file():
            for line in curve_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    curve.append(json.loads(line))

        runs.append(
            {
                "id": mix_id,
                "tag": mix.tag,
                "run_name": mix.run_name,
                "weights": mix.weights,
                "task_loss_labels": labels,
                "task_loss_families": families,
                "macro_mean": macro_curve(families),
                "curve": curve,
            }
        )

    if missing:
        raise SystemExit(
            f"missing task_loss_final.json for probes: {missing}. "
            f"Expected under {runs_dir}/<probe_id>/progress/"
        )
    return {
        "domain_order": list(DOMAINS),
        "curve_families": list(CURVE_FAMILIES),
        "runs": runs,
    }


def build_A_from_extrapolated(report: dict) -> tuple[np.ndarray, dict]:
    """Construct A (7×6) and a detail dict from Chinchilla-extrapolated report."""
    by_name = {r["run_name"]: r for r in report["runs"]}
    if UNI_RUN not in by_name:
        raise SystemExit(f"missing uniform probe run {UNI_RUN!r}")
    uni = by_name[UNI_RUN]
    L_uni = {}
    for fam in CURVE_FAMILIES:
        entry = uni["families"][fam]
        if entry.get("chinchilla") is None:
            raise SystemExit(f"{UNI_RUN}::{fam}: Chinchilla loss is None ({entry.get('note')})")
        L_uni[fam] = float(entry["chinchilla"])

    A = np.zeros((len(DOMAINS), len(CURVE_FAMILIES)), dtype=np.float64)
    chin_losses: dict[str, dict[str, float]] = {UNI_RUN: dict(L_uni)}
    for i, dom in enumerate(DOMAINS):
        run_name = f"probe_{dom}"
        if run_name not in by_name:
            raise SystemExit(f"missing one-hot probe {run_name!r}")
        run = by_name[run_name]
        chin_losses[run_name] = {}
        for j, fam in enumerate(CURVE_FAMILIES):
            entry = run["families"][fam]
            if entry.get("chinchilla") is None:
                raise SystemExit(
                    f"{run_name}::{fam}: Chinchilla loss is None ({entry.get('note')})"
                )
            L_i = float(entry["chinchilla"])
            chin_losses[run_name][fam] = L_i
            A[i, j] = max(0.0, L_uni[fam] - L_i)

    detail = {
        "formula": "A_ij = max(0, L_uni_j - L_i_j) at Chinchilla step",
        "chinchilla_step": int(report.get("chinchilla_steps", CHINCHILLA_STEP)),
        "uniform_run": UNI_RUN,
        "chinchilla_losses": chin_losses,
        **A_to_named_dict(A, domains=DOMAINS, families=CURVE_FAMILIES),
    }
    return A, detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Root with <probe_id>/progress/task_loss_*.json (required unless --data)",
    )
    ap.add_argument(
        "--mixtures-json",
        type=Path,
        default=_SKILLIT / "probes.json",
        help="Probe mixture definitions (default: skillit/probes.json)",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional pre-collected mixlaw_data.json (skips collect)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_SKILLIT / "artifacts",
        help="Write A_offline.npy plus named JSON artifacts here",
    )
    ap.add_argument(
        "--s3-uri",
        default=ARTIFACTS_S3_URI,
        help=f"Artifact export prefix (default: {ARTIFACTS_S3_URI})",
    )
    ap.add_argument(
        "--no-s3-export",
        action="store_true",
        help="Keep artifacts local; equivalent to S3_EXPORT=0 for this invocation",
    )
    ap.add_argument("--step", type=int, default=CHINCHILLA_STEP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write-intermediate",
        action="store_true",
        help="Also write collected + extrapolated JSON under out-dir",
    )
    args = ap.parse_args()

    if args.data is not None:
        data = json.loads(args.data.read_text(encoding="utf-8"))
    else:
        if args.runs_dir is None:
            raise SystemExit("provide --runs-dir or --data")
        data = collect_probe_runs(args.runs_dir, args.mixtures_json)

    _, default_chin_step, _ = token_budget(20.0)
    target_step = int(args.step) if args.step is not None else default_chin_step
    report = extrapolate_runs(data, target_step, seed=args.seed)
    A, detail = build_A_from_extrapolated(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "A_offline.npy"
    json_path = args.out_dir / "adjacency.json"
    publish_json_path = args.out_dir / "A_offline.json"
    np.save(npy_path, A)
    payload = {
        **detail,
        "npy": str(npy_path.name),
        "shape": list(A.shape),
        "eta_default": 0.2,
        "w_default": 1.0,
    }
    json_text = json.dumps(payload, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    publish_json_path.write_text(json_text, encoding="utf-8")
    print(f"wrote {npy_path} shape={A.shape}")
    print(f"wrote {json_path}")
    print(f"wrote {publish_json_path}")

    if args.write_intermediate:
        (args.out_dir / "probe_data.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        (args.out_dir / "probe_chinchilla_extrapolated.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote intermediate JSONs under {args.out_dir}")

    if args.no_s3_export:
        print("S3 export disabled by --no-s3-export")
    else:
        from token_selection.olmo_ext.s3_export import sync_to_s3

        sync_to_s3(args.out_dir, args.s3_uri)

    # Quick summary: which domains help which families.
    print("\nA (rows=domains, cols=families); positive = domain beats uniform:")
    hdr = " ".join(f"{f[:8]:>8}" for f in CURVE_FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        row = " ".join(f"{A[i, j]:8.4f}" for j in range(len(CURVE_FAMILIES)))
        print(f"{d:<18} {row}")


if __name__ == "__main__":
    main()
