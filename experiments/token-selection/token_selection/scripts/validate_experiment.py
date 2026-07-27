#!/usr/bin/env python3
"""Fail-closed preflight for a token-selection experiment launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.scripts import load_config, resolve_output_dir, resolve_tokens_s3
from token_selection.scripts.experiment_contract import (
    validate_order_contract,
    validate_scratch_config,
    validate_token_budget,
    validate_token_manifest,
    verify_olmo_revision,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "token_selection/configs/run_rho_10b.yaml",
    )
    ap.add_argument(
        "--olmo-root",
        type=Path,
        default=None,
        help="Editable edu-llm/OLMo-core checkout to verify against the pinned revision.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    methods = cfg.get("methods") or []
    method = str(methods[0]) if len(methods) == 1 else None
    validate_scratch_config(cfg, method=method)

    # RHO production configs must point at a real local reference *before* launch.
    # validate_scratch_config only rejects null/REPLACE_ME; existence is checked here so
    # a typo cannot pass preflight and then strand a run_fingerprint after --launch.
    if method == "rho_excess":
        ref_raw = str(((cfg.get("reference") or {}).get("load_path")) or "")
        ref_path = Path(ref_raw)
        if not ref_path.is_absolute():
            ref_path = ROOT / ref_path
        if not ref_path.exists():
            raise SystemExit(
                f"rho_excess reference.load_path does not exist: {ref_raw!r} "
                f"(resolved {ref_path}). Sync the checkpoint locally first."
            )

    try:
        tokens_s3 = resolve_tokens_s3(cfg)
        token_manifest = validate_token_manifest(
            out / "tokens",
            expected_tokenizer=(cfg.get("data") or {}).get("tokenizer"),
        )
        budget = validate_token_budget(cfg, token_manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    order_manifest = out / "order" / "manifest.json"
    if not order_manifest.exists():
        raise SystemExit(
            f"Missing order contract: {order_manifest}; download tokens then run freeze_order"
        )
    order = json.loads(order_manifest.read_text(encoding="utf-8"))
    try:
        validate_order_contract(cfg, output_dir=out, contract=order["order_contract"])
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.olmo_root is not None:
        required = str((cfg.get("olmo_core") or {}).get("revision") or "")
        if not required:
            raise SystemExit("olmo_core.revision is required when --olmo-root is provided")
        try:
            verify_olmo_revision(args.olmo_root, required)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "status": "valid",
                "run_id": cfg.get("run_id"),
                "init_mode": (cfg.get("model") or {}).get("init_mode"),
                "tokens_s3": tokens_s3,
                "tokenizer": token_manifest.get("tokenizer"),
                "n_shards": len(token_manifest["shards"]),
                "n_tokens": token_manifest["n_tokens"],
                "token_budget": budget,
                "order_contract": order["order_contract"]["contract_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
