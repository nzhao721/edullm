"""Local-scratch and W&B durability primitives for token-selection runs.

S3 is deliberately absent from this module. Training data and immutable
bootstrap references may be staged from S3 at run start; checkpoints, progress,
evals, durable markers, and resume state remain on runtime scratch and W&B.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

LAST_DURABLE_STEP_FILENAME = "last_durable_step.json"


class DurabilityError(RuntimeError):
    """Local or W&B durability state is invalid."""


def _read_path(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("last_durable_step"), int):
        raise DurabilityError(f"invalid durable-step metadata: {path}")
    if int(payload["last_durable_step"]) < 0:
        raise DurabilityError(f"invalid negative durable step: {path}")
    return payload


def read_last_durable_step(metadata_dir: str | Path) -> Optional[dict[str, Any]]:
    return _read_path(Path(metadata_dir) / LAST_DURABLE_STEP_FILENAME)


def write_last_durable_step(
    metadata_dir: str | Path,
    step: int,
    *,
    checkpoint_artifact: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically advance local metadata after required W&B uploads complete."""
    step = int(step)
    if step < 0:
        raise ValueError("last durable step must be non-negative")
    directory = Path(metadata_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / LAST_DURABLE_STEP_FILENAME
    current = _read_path(target)
    if current is not None and step < int(current["last_durable_step"]):
        raise DurabilityError(
            "refusing to move last durable step backward "
            f"({current['last_durable_step']} → {step})"
        )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "last_durable_step": step,
        "durability": "local_scratch+wandb",
    }
    if checkpoint_artifact:
        payload["checkpoint_artifact"] = str(checkpoint_artifact)
    if extra:
        for key, value in extra.items():
            if key not in {"schema_version", "last_durable_step", "durability"}:
                payload[str(key)] = value
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(directory)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
