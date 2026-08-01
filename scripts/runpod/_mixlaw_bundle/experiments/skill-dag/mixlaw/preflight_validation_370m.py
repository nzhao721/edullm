#!/usr/bin/env python3
"""Fail-closed dependency and OLMES preflight for MixLaw 370M training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_TS_ROOT = _REPO_ROOT / "experiments" / "token-selection"
for _path in (_HERE, _TS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mixlaw_runtime import (  # noqa: E402
    OLMES_BPB_LABELS,
    collect_dependency_versions,
    dependency_contract_errors,
)
from token_selection.olmo_ext.task_loss_hook import resolve_eval_script  # noqa: E402


def run_preflight(
    *,
    ladder_base_config: Path | None,
    eval_script: Path | None,
) -> dict[str, Any]:
    """Return metadata after proving the production eval stack is complete."""
    try:
        from olmo.eval.downstream import label_to_task_map
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot import OLMES label map: {exc}") from exc

    versions = collect_dependency_versions()
    available_labels = tuple(label_to_task_map)
    errors = dependency_contract_errors(
        versions,
        available_labels,
        ladder_base_config=ladder_base_config,
        eval_script=eval_script,
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": versions,
        "olmes_bpb_labels": list(OLMES_BPB_LABELS),
        "olmes_bpb_label_count": len(OLMES_BPB_LABELS),
        "ladder_base_config": str(Path(ladder_base_config).resolve()),
        "task_loss_eval_script": str(Path(eval_script).resolve()),
        "edullm_data_install_policy": "newest-release-or-github-main",
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder-base-config",
        type=Path,
        default=Path(os.environ["LADDER_BASE_CONFIG"])
        if os.environ.get("LADDER_BASE_CONFIG")
        else None,
    )
    parser.add_argument("--task-loss-eval-script", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    eval_script = resolve_eval_script(args.task_loss_eval_script)
    try:
        payload = run_preflight(
            ladder_base_config=args.ladder_base_config,
            eval_script=eval_script,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mixlaw-preflight] FAILED: {exc}", file=sys.stderr)
        return 2
    write_json_atomic(args.out, payload)
    print(
        "[mixlaw-preflight] ok "
        f"labels={payload['olmes_bpb_label_count']} "
        f"edullm-data={payload['dependencies']['edullm-data']} "
        f"metadata={args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
