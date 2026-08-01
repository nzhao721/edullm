#!/usr/bin/env python3
"""Train shared YAML methods via ``train_olmo_template``.

Supported: ``full``, ``rel_ema``, ``rho_excess``, ``middle_ppl``,
``attention_topk``, ``learnability``. This entry point validates the config and
reports derived training parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.olmo_ext.scorers import MethodName
from token_selection.olmo_ext.train_module import has_olmo_core, make_ts_config
from token_selection.scripts import (
    derive_steps,
    load_config,
    resolve_output_dir,
    resolve_train_dataset,
)
from token_selection.scripts.experiment_contract import validate_scratch_config

_ALL: list[MethodName] = [
    "full",
    "rel_ema",
    "rho_excess",
    "middle_ppl",
    "attention_topk",
    "learnability",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_rho_10b.yaml")
    ap.add_argument(
        "--method",
        choices=_ALL,
        default=None,
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    methods = cfg.get("methods") or []
    if args.method:
        method: MethodName = args.method  # type: ignore[assignment]
    elif len(methods) == 1:
        method = str(methods[0])  # type: ignore[assignment]
    elif "learnability" in methods:
        method = "learnability"
    elif "attention_topk" in methods:
        method = "attention_topk"
    elif "middle_ppl" in methods:
        method = "middle_ppl"
    elif "rho_excess" in methods:
        method = "rho_excess"
    elif "rel_ema" in methods:
        method = "rel_ema"
    else:
        method = "rho_excess"
    validate_scratch_config(cfg, method=method)

    allowed = cfg.get("methods") or _ALL
    if method not in allowed:
        raise SystemExit(f"method {method!r} not in config methods {allowed}")

    # Fail closed: train corpus must resolve from published+validated edullm-data.
    # Does not consult FarmShare scratch, laptop-local caches, or edullm-datasets.
    try:
        train_data = resolve_train_dataset(cfg)
    except Exception as exc:
        raise SystemExit(f"train corpus resolution failed: {exc}") from exc

    total_steps, t0_steps = derive_steps(cfg)
    ts_cfg = make_ts_config(cfg, method=method, total_steps=total_steps, t0_steps=t0_steps)
    print(
        json.dumps(
            {
                "status": "olmo_core_required",
                "method": method,
                "has_olmo_core": has_olmo_core(),
                "total_steps": total_steps,
                "t0_steps": t0_steps,
                "ts_cfg": ts_cfg.__dict__,
                "train_data": {
                    "dataset_id": train_data["dataset_id"],
                    "version": train_data["version"],
                    "tokens_uri": train_data["tokens_uri"],
                    "dtype": train_data["dtype"],
                    "numpy_dtype": train_data["numpy_dtype"],
                    "rows": train_data["rows"],
                    "n_shards": len(train_data["paths"]),
                },
                "checkpoint_artifacts": "wandb",
                "hint": (
                    "Use token_selection.scripts.train_olmo_template --method "
                    f"{method} --olmo-root <OLMo-core> --launch "
                    "(stages tokens+order from edullm-data; uploads checkpoints, "
                    "progress, and evals to W&B). Resume with --resume and "
                    "WANDB_RESUME_ARTIFACT when local scratch is empty."
                ),
                "run_id": cfg.get("run_id"),
                "output_dir": str(out),
            },
            indent=2,
        )
    )
    if not has_olmo_core():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
