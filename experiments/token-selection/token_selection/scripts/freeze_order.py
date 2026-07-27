#!/usr/bin/env python3
"""Record the deterministic order contract shared by experiment arms.

Public OLMo-core derives global indices from its loader seed and dataset
fingerprint; production uses the explicit contract written here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.scripts import load_config, resolve_output_dir  # noqa: E402
from token_selection.scripts.experiment_contract import (  # noqa: E402
    build_order_contract,
    validate_token_budget,
    validate_token_manifest,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_rho_10b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    seed = int(cfg.get("seed", 42))
    seq_len = int(cfg["data"]["sequence_length"])

    tokens_dir = out / "tokens"
    order_dir = out / "order"
    order_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = tokens_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing {manifest_path}. Download tokens with "
            "`sync_artifacts.py --direction download --what tokens`, then derive the "
            "manifest with `python -m token_selection.scripts.build_token_manifest`."
        )

    try:
        manifest = validate_token_manifest(
            tokens_dir, expected_tokenizer=(cfg.get("data") or {}).get("tokenizer")
        )
        budget = validate_token_budget(cfg, manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    n_tokens = int(manifest["n_tokens"])
    n_seqs = n_tokens // seq_len
    if n_seqs <= 0:
        raise SystemExit(f"Not enough tokens ({n_tokens}) for sequence_length={seq_len}")

    order_contract = build_order_contract(cfg, output_dir=out, token_manifest=manifest)
    meta = {
        "schema_version": 2,
        "seed": seed,
        "sequence_length": seq_len,
        "n_tokens": n_tokens,
        "n_sequences": n_seqs,
        "remainder_tokens": n_tokens - n_seqs * seq_len,
        "token_budget": budget,
        "order_contract": order_contract,
        "note": "Production OLMo-core uses the seeded global-index order in order_contract.",
    }
    (order_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Wrote deterministic order contract -> {order_dir}\n"
        f"  shards: {len(manifest['shards'])}  tokens: {n_tokens}\n"
        f"  trainable per epoch: {budget['usable_sequences_per_epoch']} sequences "
        f"({budget['usable_tokens_per_epoch']} tokens)\n"
        f"  budget: {budget['max_tokens']} tokens = {budget['epochs_consumed']} epochs"
    )


if __name__ == "__main__":
    main()
