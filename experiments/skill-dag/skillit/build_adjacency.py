#!/usr/bin/env python3
"""Build offline Skill-It adjacency A from 7 DataDecide-60M one-hot probe runs.

Pipeline:
  1. Collect per-probe ``task_loss.jsonl`` + ``task_loss_final.json`` (same schema
     as mixlaw ``fit_mixing_law.py collect``). Step-law fits use **jsonl only** (steps
     120–1440); ``task_loss_final.json`` at 1451 is not appended (see mixlaw README).
  2. Chinchilla-extrapolate each curve family to step 5806 (tpp=20) via
     ``mixlaw/extrapolate_chinchilla.py`` logic.
  3. ``A_ij = max(0, L_j(r_RegMix) - L_i_j)`` for domains i and families j,
     with ``L_j(r_RegMix)`` from ``mixlaw_fit_chinchilla.json``.
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
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    offline_A_from_extrapolated,
    regmix_family_losses_from_fit,
)

CHINCHILLA_STEP = 5806  # token_budget(20) → 5806
ARTIFACTS_S3_URI = "s3://edullm-checkpoints/skillit/artifacts"
LEGACY_UNI_RUN = "probe_uni"


def _is_onehot_probe(run_name: str, tag: str) -> bool:
    if run_name == LEGACY_UNI_RUN or tag == "uniform":
        return False
    return run_name.startswith("probe_")


def collect_probe_runs(runs_dir: Path, mixtures_json: Path) -> dict:
    """Gather one-hot probe progress into a mixlaw_data-compatible payload."""
    mixtures = {m.id: m for m in load_mixtures(mixtures_json)}
    runs = []
    missing = []
    for mix_id, mix in sorted(mixtures.items()):
        if not _is_onehot_probe(mix.run_name, mix.tag):
            continue
        progress = runs_dir / mix.run_name / "progress"
        final = progress / "task_loss_final.json"
        if not final.is_file():
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


def build_A_from_extrapolated(
    report: dict,
    L_reg: dict[str, float],
    *,
    chinchilla_step: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Construct A (7×6) vs RegMix reference losses."""
    return offline_A_from_extrapolated(
        report,
        L_reg,
        domains=DOMAINS,
        families=CURVE_FAMILIES,
        reference_label="regmix",
        chinchilla_step=chinchilla_step or report.get("chinchilla_steps", CHINCHILLA_STEP),
    )


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
        "--fit-json",
        type=Path,
        default=None,
        help="Mixlaw fit for L_j(r_RegMix) (default: mixlaw/mixlaw_fit_chinchilla.json)",
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

    fit_path = args.fit_json or default_mixlaw_fit_path()
    fit = load_fit_json(fit_path)
    L_reg = regmix_family_losses_from_fit(fit, domains=DOMAINS, families=CURVE_FAMILIES)

    if args.data is not None:
        data = json.loads(args.data.read_text(encoding="utf-8"))
    else:
        if args.runs_dir is None:
            raise SystemExit("provide --runs-dir or --data")
        data = collect_probe_runs(args.runs_dir, args.mixtures_json)

    _, default_chin_step, _ = token_budget(20.0)
    target_step = int(args.step) if args.step is not None else default_chin_step
    report = extrapolate_runs(data, target_step, seed=args.seed)
    A, detail = build_A_from_extrapolated(report, L_reg, chinchilla_step=target_step)
    detail["fit_json"] = str(fit_path)

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

    print("\nA (rows=domains, cols=families); positive = domain beats RegMix @ Chinchilla:")
    hdr = " ".join(f"{f[:8]:>8}" for f in CURVE_FAMILIES)
    print(f"{'domain':<18} {hdr}")
    for i, d in enumerate(DOMAINS):
        row = " ".join(f"{A[i, j]:8.4f}" for j in range(len(CURVE_FAMILIES)))
        print(f"{d:<18} {row}")


if __name__ == "__main__":
    main()
