"""Build-time validation for the MixLaw CUDA image."""
from __future__ import annotations

import ctypes
import importlib.metadata
from pathlib import Path

import datasets  # noqa: F401
import flash_attn  # noqa: F401
import sklearn  # noqa: F401
import torch
import torchmetrics  # noqa: F401
import wandb  # noqa: F401
from olmo.eval.downstream import label_to_task_map

import domain_stream  # noqa: F401
import mixlaw_runtime
import mixlaw_wandb  # noqa: F401
import platform_array_entrypoint  # noqa: F401
import platform_artifacts  # noqa: F401
import preflight_validation_370m  # noqa: F401
import stage_validation_pool_from_edullm_data  # noqa: F401
import train_mixlaw_validation_370m  # noqa: F401

assert torch.version.cuda == "12.8", torch.version.cuda
ctypes.CDLL("libcudart.so.12")

expected = {
    "ai2-olmo-core": "2.4.0",
    "ai2-olmo": "0.6.0",
    "edullm-data": "0.6.0",
    "wandb": "0.28.1",
    "torchmetrics": "1.9.0",
    "scikit-learn": "1.9.0",
}
installed = {name: importlib.metadata.version(name) for name in expected}
assert installed == expected, (installed, expected)
assert set(mixlaw_runtime.OLMES_BPB_LABELS).issubset(label_to_task_map)
assert len(mixlaw_runtime.OLMES_BPB_LABELS) == 20
assert Path(
    "/opt/edullm/experiments/skill-dag/mixlaw/ladder_base_config.yaml"
).is_file()
