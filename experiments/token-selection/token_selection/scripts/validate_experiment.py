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

from token_selection.olmo_ext.refhq_materialize import reference_source_ok
from token_selection.scripts import load_config, resolve_output_dir, resolve_tokens_s3
from token_selection.scripts.edullm_data_tokens import (
    ensure_order_contract,
    ensure_train_tokens,
)
from token_selection.scripts.experiment_contract import (
    validate_order_contract,
    validate_scratch_config,
    validate_token_budget,
    validate_token_manifest,
    verify_olmo_revision,
)


def _reference_local_ok(cfg: dict, *, method: str | None) -> bool:
    """True when required local .pt paths already exist (S3 provenance may still be OK)."""
    if method is None:
        return True
    ref = cfg.get("reference") or {}
    if method in ("rho_excess", "rel_ema"):
        if method == "rel_ema":
            ema = cfg.get("ema") or {}
            seed = str(ema.get("seed_mode") or cfg.get("ema_seed_mode") or "zero").lower()
            if seed != "refhq":
                return True
        raw = str(ref.get("load_path") or "")
        if not raw or raw in {"null", "None", "REPLACE_ME"}:
            return False
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        return path.is_file()
    if method == "learnability":
        for key in ("early", "late"):
            block = ref.get(key) or {}
            raw = str(block.get("load_path") or "")
            if not raw or raw in {"null", "None", "REPLACE_ME"}:
                return False
            path = Path(raw)
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                return False
        return True
    return True


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
    ap.add_argument(
        "--stage",
        action="store_true",
        help=(
            "Stage train tokens + order contract from published edullm-data into "
            "output_dir (required on ephemeral scratch with no prior local corpus)."
        ),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    methods = cfg.get("methods") or []
    method = str(methods[0]) if len(methods) == 1 else None
    validate_scratch_config(cfg, method=method)

    # Refs: accept local .pt *or* S3 provenance (materialized at --launch). Do not
    # require a pre-existing scratch/laptop cache of RefHQ weights.
    if method in ("rho_excess", "rel_ema", "learnability"):
        if not reference_source_ok(cfg, method=method):
            raise SystemExit(
                f"{method} requires reference.load_path (local .pt) or S3 provenance "
                "(reference.s3_uri / early+late s3 fields); nothing to materialize at launch."
            )
        if not _reference_local_ok(cfg, method=method):
            print(
                json.dumps(
                    {
                        "warning": "reference_load_path_missing_locally",
                        "method": method,
                        "hint": (
                            "OK for ephemeral jobs: train_olmo_template --launch will "
                            "materialize from reference.s3_uri(s) into TOKEN_SELECTION_REF_CACHE."
                        ),
                    },
                    indent=2,
                ),
                flush=True,
            )

    if args.stage:
        try:
            ensure_train_tokens(cfg, out / "tokens")
            ensure_order_contract(cfg, out)
        except Exception as exc:
            raise SystemExit(f"train data staging failed: {exc}") from exc

    try:
        tokens_s3 = resolve_tokens_s3(cfg)
        token_manifest = validate_token_manifest(
            out / "tokens",
            expected_tokenizer=(cfg.get("data") or {}).get("tokenizer"),
        )
        budget = validate_token_budget(cfg, token_manifest)
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\n\nOn a clean/ephemeral machine, re-run with --stage to fetch "
            "tokens from s3://edullm-data/ via data.dataset_id (do not assume scratch "
            "corpora persist)."
        ) from exc

    order_manifest = out / "order" / "manifest.json"
    if not order_manifest.exists():
        raise SystemExit(
            f"Missing order contract: {order_manifest}. Re-run with --stage "
            "(or train_olmo_template --launch) so the contract is built from the "
            "staged edullm-data corpus."
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
                "staged": bool(args.stage),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
