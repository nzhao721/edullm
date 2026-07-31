"""Canonical S3 layout for token-selection experiment exports.

All arm checkpoints and results land under::

    s3://edullm-checkpoints/token-sel/<arm>/

with subdirectories ``checkpoints/``, ``task_loss_results/``, and optionally
``metrics/`` / ``progress/``.

Arm directory names match ``experiments/token-selection/<arm>/``.
"""

from __future__ import annotations

from typing import Final

CHECKPOINT_BUCKET: Final[str] = "edullm-checkpoints"
TOKEN_SEL_ROOT: Final[str] = "token-sel"

# Directory names under experiments/token-selection/ and under token-sel/ on S3.
ARM_DIRS: Final[tuple[str, ...]] = (
    "control",
    "blade",
    "rho-1",
    "rel-ema-exp",
    "rel-ema-refhq",
    "middle-ppl-token",
    "middle-ppl-doc",
    "attention",
    "learnability-token",
    "learnability-doc",
    "reference",
)


def arm_prefix(arm: str) -> str:
    """Return ``token-sel/<arm>`` (no leading/trailing slash)."""
    arm = str(arm).strip().strip("/")
    if not arm:
        raise ValueError("arm name must be non-empty")
    return f"{TOKEN_SEL_ROOT}/{arm}"


def arm_uri(arm: str, *parts: str) -> str:
    """Build ``s3://edullm-checkpoints/token-sel/<arm>[/parts…]``."""
    prefix = arm_prefix(arm)
    extra = "/".join(p.strip("/") for p in parts if str(p).strip())
    if extra:
        return f"s3://{CHECKPOINT_BUCKET}/{prefix}/{extra}"
    return f"s3://{CHECKPOINT_BUCKET}/{prefix}"


# Published corpora live in locked ``edullm-data`` (read via edullm_data.read).
# This constant is only the default *output* dataset_bucket key for older YAML;
# new configs should omit it and set ``data.dataset_id`` instead.
DATA_BUCKET: Final[str] = "edullm-data"
DATASET_BUCKET: Final[str] = DATA_BUCKET  # back-compat alias


def default_s3_block(arm: str) -> dict[str, str]:
    """YAML ``s3:`` block for spine configs.

    Checkpoints and experiment results publish under
    ``s3://edullm-checkpoints/token-sel/<arm>/``. Pre-tokenized train data is
    resolved from ``s3://edullm-data`` via ``data.dataset_id`` (see
    ``token_selection.scripts.train_data_resolve``).
    """
    return {
        "checkpoint_bucket": CHECKPOINT_BUCKET,
        "prefix": arm_prefix(arm),
        "profile": "sbsandbox",
    }


def arm_from_prefix(prefix: str) -> str:
    """Parse ``token-sel/<arm>[/…]`` → ``<arm>``."""
    prefix = str(prefix).strip().strip("/")
    root = f"{TOKEN_SEL_ROOT}/"
    if not prefix.startswith(root):
        raise ValueError(
            f"s3.prefix must start with {root!r} (got {prefix!r}); "
            "use token-sel/<arm> for checkpoint/result exports"
        )
    arm = prefix[len(root) :].split("/", 1)[0]
    if not arm:
        raise ValueError(f"s3.prefix missing arm directory: {prefix!r}")
    return arm
