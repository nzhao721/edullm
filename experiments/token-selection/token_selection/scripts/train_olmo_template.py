#!/usr/bin/env python3
"""OLMo-core scratch entry for ``rel_ema``, ``rho_excess``, ``middle_ppl``,
``attention_topk``, ``learnability``, and optional ``full``.

Requires edu-llm/OLMo-core installed. Builds trainer configs from the experiment
YAML and documents the torchrun launch. ``--launch`` fails closed if the requested
frozen-order controls cannot be represented by the pinned public APIs.

Ephemeral empty-scratch contract:

* Stage train shards from published ``s3://edullm-data/`` (never ``edullm-datasets``).
* Do not assume FarmShare/laptop corpora, local venvs, or prior save folders.
* Durable artifacts (permanent step dirs, ``run_fingerprint.json``, metrics,
  task_loss JSON) export to ``s3://edullm-checkpoints/token-sel/<arm>/``.
* ``--resume`` hydrates an empty save folder from that durable prefix first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.olmo_ext.checkpoint_ladder import (
    DEFAULT_CHECKPOINT_INTERVAL,
    checkpointer_kwargs_for_ladder,
    permanent_checkpoint_steps,
)
from token_selection.olmo_ext.scorers import MethodName
from token_selection.olmo_ext.train_module import make_ts_config
from token_selection.scripts import (
    derive_steps,
    load_config,
    resolve_output_dir,
    resolve_tokens_s3,
    resolve_train_dataset,
    s3_uri,
)
from token_selection.scripts.edullm_data_tokens import (
    ensure_order_contract,
    ensure_train_tokens,
)
from token_selection.scripts.experiment_contract import (
    manifest_train_paths,
    validate_order_contract,
    validate_scratch_config,
    validate_token_budget,
    validate_token_manifest,
    verify_olmo_revision,
)
_SELECTING = (
    "rel_ema",
    "rho_excess",
    "middle_ppl",
    "attention_topk",
    "learnability",
)

# On a shared multi-GPU host, launching without an explicit pin would default to
# physical GPU 0. Refuse that rather than compete with whatever owns the other devices.
_IDLE_MEMORY_MIB = 256


def resolve_attn_backend(cfg: Dict[str, Any] | None = None) -> Any:
    """Prefer FlashAttention-2 when available; else PyTorch SDPA.

    Resolution order:
      1. ``train.attn_backend`` in the experiment YAML (if set)
      2. env ``OLMO_ATTN_BACKEND`` (default ``auto``)

    Values: ``auto`` (flash if importable), ``flash_2``/``flash``, or ``torch``/``sdpa``.
    This is a wall-clock / memory kernel choice only — not part of the run fingerprint.
    """
    from olmo_core.nn.attention import AttentionBackendName  # type: ignore

    train = (cfg or {}).get("train") or {}
    raw = train.get("attn_backend")
    if raw is None or str(raw).strip() == "":
        prefer = os.environ.get("OLMO_ATTN_BACKEND", "auto").strip().lower()
    else:
        prefer = str(raw).strip().lower()

    if prefer in ("torch", "sdpa", "eager"):
        print(json.dumps({"attn_backend": "torch", "reason": "requested"}), flush=True)
        return AttentionBackendName.torch

    want_flash = prefer in ("auto", "flash_2", "flash", "flash2")
    if want_flash:
        try:
            import flash_attn  # noqa: F401

            backend = AttentionBackendName.flash_2
            get_cls = getattr(backend, "get_class", None)
            if callable(get_cls):
                get_cls().assert_supported()
            print(
                json.dumps(
                    {
                        "attn_backend": "flash_2",
                        "reason": "flash_attn available",
                        "prefer": prefer,
                    }
                ),
                flush=True,
            )
            return backend
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "attn_backend": "torch",
                        "reason": "flash_attn unavailable",
                        "prefer": prefer,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
            return AttentionBackendName.torch

    try:
        backend = AttentionBackendName(prefer)
        print(json.dumps({"attn_backend": prefer, "reason": "explicit"}), flush=True)
        return backend
    except Exception:
        print(
            json.dumps(
                {
                    "attn_backend": "torch",
                    "reason": "unknown prefer; fallback",
                    "prefer": prefer,
                }
            ),
            flush=True,
        )
        return AttentionBackendName.torch


def _parse_gpu_indices(raw: str) -> List[str]:
    """Parse a CUDA_VISIBLE_DEVICES-style pin into distinct integer GPU indices."""
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    if not parts:
        raise SystemExit("GPU pin is empty after parsing")
    indices: List[str] = []
    for part in parts:
        if not part.isdigit():
            raise SystemExit(
                f"train.cuda_visible_devices entries must be integer GPU indices, got {part!r} "
                f"in {raw!r}"
            )
        indices.append(part)
    if len(set(indices)) != len(indices):
        raise SystemExit(f"duplicate GPU indices in pin {raw!r}")
    return indices


def pin_cuda_visible_devices(cfg: Dict[str, Any]) -> str:
    """Force ``CUDA_VISIBLE_DEVICES`` to the configured physical GPU(s) and verify idle.

    Must run before any CUDA context is created. Accepts a single index (``\"6\"``) or a
    comma-separated list (``\"6,7\"``) for multi-GPU ``torchrun`` launches. Each listed
    physical device is probed via ``nvidia-smi -i`` and refused if it holds more than a
    trivial amount of memory — unless running under Slurm (``SLURM_JOB_ID``) or
    ``TOKEN_SELECTION_SKIP_IDLE_CHECK=1``, where the allocator owns the devices.
    Returns the pinned index string written to the env var.
    """
    raw = (cfg.get("train") or {}).get("cuda_visible_devices")
    if raw is None or str(raw).strip() == "":
        # Allow the outer launcher to own the pin (AWS scripts / Slurm).
        env_pin = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_pin is not None and str(env_pin).strip() != "":
            raw = env_pin
        else:
            raise SystemExit(
                "train.cuda_visible_devices is required for --launch (or set "
                "CUDA_VISIBLE_DEVICES in the environment). On a multi-GPU host omitting "
                "it would let torch grab GPU 0. Set it to an idle physical index "
                '(e.g. "0") or a comma list (e.g. "6,7").'
            )
    pinned = str(raw).strip()
    indices = _parse_gpu_indices(pinned)
    pinned = ",".join(indices)

    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and existing.strip() != pinned:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES is already {existing!r} but train.cuda_visible_devices "
            f"is {pinned!r}. Refusing to launch against a conflicting pin."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = pinned

    skip_idle = (
        os.environ.get("TOKEN_SELECTION_SKIP_IDLE_CHECK", "").strip().lower()
        in {"1", "true", "yes"}
        or bool(os.environ.get("SLURM_JOB_ID"))
    )
    probes = []
    for idx in indices:
        try:
            probe = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    idx,
                    "--query-gpu=index,uuid,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                f"Unable to query physical GPU {idx} via nvidia-smi; refusing to launch. ({exc})"
            ) from exc

        parts = [p.strip() for p in probe.split(",")]
        if len(parts) < 3 or parts[0] != idx:
            raise SystemExit(f"Unexpected nvidia-smi response for GPU {idx}: {probe!r}")
        used_mib = int(float(parts[2]))
        if not skip_idle and used_mib > _IDLE_MEMORY_MIB:
            raise SystemExit(
                f"Physical GPU {idx} is not idle ({used_mib} MiB used > {_IDLE_MEMORY_MIB} MiB). "
                "Refusing to launch so we do not share a device with another job."
            )
        probes.append(
            {
                "physical_gpu": idx,
                "uuid": parts[1],
                "memory_used_mib": used_mib,
                "idle_check_skipped": skip_idle,
            }
        )
    print(
        json.dumps(
            {
                "cuda_visible_devices": pinned,
                "num_gpus": len(indices),
                "devices": probes,
                "status": "pinned_idle",
            }
        ),
        flush=True,
    )
    return pinned


def _token_paths(tokens_dir: Path, *, expected_tokenizer: str) -> List[str]:
    """Training shards come from the manifest, not a glob (see manifest_train_paths)."""
    try:
        return manifest_train_paths(tokens_dir, expected_tokenizer=expected_tokenizer)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def prepare_train_data(cfg: Dict[str, Any], out: Path, *, stage: bool = True) -> Dict[str, Any]:
    """Resolve + optionally stage edullm-data train shards; ensure order contract.

    Never assumes FarmShare scratch / laptop-local tokens already exist. When
    ``stage`` is true (default on ``--launch``), fetches from validated
    ``s3://edullm-data/`` via ``ensure_train_tokens``. Plan-only mode still
    resolves the published corpus so configs cannot silently point at legacy
    ``edullm-datasets`` URIs.
    """
    try:
        remote = resolve_train_dataset(cfg)
    except Exception as exc:
        raise SystemExit(f"train corpus resolution failed: {exc}") from exc
    tokens_dir = out / "tokens"
    if stage:
        try:
            ensure_train_tokens(cfg, tokens_dir)
            ensure_order_contract(cfg, out)
        except Exception as exc:
            raise SystemExit(f"train data staging failed: {exc}") from exc
    return remote


def build_plan(
    cfg: Dict[str, Any],
    *,
    method: MethodName,
    out: Path,
) -> Dict[str, Any]:
    total_steps, t0_steps = derive_steps(cfg)
    ts_cfg = make_ts_config(cfg, method=method, total_steps=total_steps, t0_steps=t0_steps)
    tokens_dir = out / "tokens"
    order_dir = out / "order"
    tokenizer = str((cfg.get("data") or {}).get("tokenizer") or "")
    if not tokenizer:
        raise SystemExit("data.tokenizer is required; it fixes the model's vocabulary size")
    if not (tokens_dir / "manifest.json").exists():
        raise SystemExit(
            f"Missing {tokens_dir / 'manifest.json'}. On a clean machine run with "
            "--launch (auto-stages from data.dataset_id via edullm_data.read) or call "
            "token_selection.scripts.edullm_data_tokens.ensure_train_tokens first."
        )
    paths = _token_paths(tokens_dir, expected_tokenizer=tokenizer)
    try:
        token_budget = validate_token_budget(
            cfg, validate_token_manifest(tokens_dir, expected_tokenizer=tokenizer)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    order_manifest_path = order_dir / "manifest.json"
    if not order_manifest_path.exists():
        raise SystemExit(
            f"Missing order contract {order_manifest_path}. Re-run with --launch "
            "(auto-builds from the staged edullm-data corpus) or freeze_order.py."
        )
    order_manifest = json.loads(order_manifest_path.read_text(encoding="utf-8"))
    try:
        validate_order_contract(
            cfg, output_dir=out, contract=order_manifest["order_contract"]
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    arch = (cfg.get("model") or {}).get("arch")
    if not arch:
        raise SystemExit(
            "model.arch is required (e.g. olmo2_370M). Refusing to guess an architecture; "
            "a silent default could mismatch the externally trained baseline arm."
        )
    olmo_revision = str((cfg.get("olmo_core") or {}).get("revision") or "")
    warmup_steps = max(
        1, int((cfg.get("train") or {}).get("warmup_steps", round(0.01 * total_steps)))
    )
    uses_selection = method in _SELECTING
    ref = cfg.get("reference") or {}
    ref_path = str(ref.get("load_path") or "")
    early_path = str((ref.get("early") or {}).get("load_path") or "")
    late_path = str((ref.get("late") or {}).get("load_path") or "")
    ema_block = cfg.get("ema") or {}
    ema_seed_mode = str(
        ema_block.get("seed_mode") or cfg.get("ema_seed_mode") or "zero"
    ).lower()
    # RHO / middle_ppl freeze RefHQ; RefHQ-seeded REL loads the same exported model.pt into EMA only.
    needs_reference = method in ("rho_excess", "middle_ppl") or (
        method == "rel_ema" and ema_seed_mode == "refhq"
    )

    return {
        "run_id": cfg["run_id"],
        "method": method,
        "seed": int(cfg["seed"]),
        "init_mode": "scratch",
        "init_seed": int((cfg.get("model") or {}).get("init_seed", cfg["seed"])),
        "load_path": None,
        "reference_load_path": ref_path if needs_reference else "",
        "early_reference_load_path": early_path if method == "learnability" else "",
        "late_reference_load_path": late_path if method == "learnability" else "",
        "ema_seed_mode": ema_seed_mode if method == "rel_ema" else "zero",
        "model_name": cfg["model"]["name"],
        "model_arch": str(arch),
        "olmo_revision": olmo_revision,
        "tokenizer": cfg["data"]["tokenizer"],
        "sequence_length": int(cfg["data"]["sequence_length"]),
        "max_tokens": int(cfg["train"]["max_tokens"]),
        "global_batch_size": int(cfg["train"]["global_batch_size"]),
        "data_loader_seed": int((cfg.get("train") or {}).get("data_loader_seed", cfg["seed"])),
        "lr": float(cfg["train"]["lr"]),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "t0_steps": t0_steps if uses_selection else 0,
        "ts_cfg": ts_cfg.__dict__,
        "token_paths": paths,
        "token_budget": token_budget,
        "data_order": {
            "contract": order_manifest["order_contract"],
            "supported": True,
            "reason": (
                "NumpyDataLoaderConfig seed plus immutable token-manifest fingerprint "
                "is the public OLMo-core global-index order contract."
            ),
        },
        "save_folder": str(out / "checkpoints" / method),
        # Deliberately outside save_folder: the fresh-scratch guard rejects a non-empty
        # save folder, so anything the build writes there before the run is pinned would
        # block the relaunch after a failed build.
        "dataset_cache": str(out / "dataset_cache" / method),
        "metrics_dir": str(out / "metrics" / method),
        "s3_tokens": resolve_tokens_s3(cfg),
        "dataset_id": str((cfg.get("data") or {}).get("dataset_id") or ""),
        "s3_checkpoints": s3_uri(
            cfg, "checkpoints", method, bucket_key="checkpoint_bucket"
        ),
        "torchrun_example": (
            "CUDA_VISIBLE_DEVICES=<gpus> python -m torch.distributed.run --standalone "
            "--nproc_per_node=<N> -m token_selection.scripts.train_olmo_template "
            f"--config <this-arm-yaml> --method {method} "
            "--olmo-root /path/to/OLMo-core --launch"
        ),
        "cuda_visible_devices": str((cfg.get("train") or {}).get("cuda_visible_devices") or ""),
        "num_gpus": int((cfg.get("train") or {}).get("num_gpus") or 0) or None,
        "notes": [
            "TokenSelectTrainModule with NumpyDataLoaderConfig seed + order contract.",
            "Tokens resolve from data.dataset_id via edullm_data.read (s3://edullm-data/); "
            "--launch stages them onto the run dir on a clean machine.",
            "Scratch initialization never resumes an existing save folder or loads optimizer/trainer state.",
            "Durable saves: permanent step dirs + run_fingerprint.json + metrics export to "
            "s3://edullm-checkpoints/token-sel/<arm>/ (do not rely on scratch persistence).",
            "--resume fetches those durable artifacts when the local save folder is empty.",
            "Launch pins train.cuda_visible_devices (single index or comma list) and refuses busy devices.",
            "Global batch size is world-size invariant; set nproc_per_node to match the pin length.",
        ],
    }


def _assert_launch_capabilities(plan: Dict[str, Any]) -> None:
    """Refuse a scientifically invalid launch rather than silently weakening the plan."""
    blockers = []
    if not plan["data_order"]["supported"]:
        blockers.append(plan["data_order"]["reason"])
    if blockers:
        detail = "\n - ".join(blockers)
        raise SystemExit(
            "Refusing to launch: required Fair REL controls are not implemented by the "
            "pinned OLMo-core public APIs.\n - " + detail
        )


def _assert_fresh_save_folder(save_folder: Path) -> None:
    """A scratch arm must not accidentally continue or overwrite a prior arm."""
    if save_folder.exists() and any(save_folder.iterdir()):
        contents = sorted(p.name for p in save_folder.iterdir())[:5]
        raise SystemExit(
            f"Scratch run refuses non-empty save folder: {save_folder} (contains {contents}). "
            "Pass --resume to continue a matching run, choose a new run directory, or "
            "delete the folder if a previous launch failed before training started."
        )


def _run_fingerprint(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Identity that must be unchanged for a resume to be scientifically valid.

    It pins initialization (arch + seed + name + tokenizer + OLMo revision), the frozen
    data order, batching, the token budget, the learning rate, and the selection
    schedule (k / EMA alphas / warmup / reference path). A resume that changed any of
    these would silently continue a *different* experiment, so we refuse it. The checkpoint
    only carries EMA/optimizer state, not these knobs, which is exactly why they must be
    re-pinned here rather than trusted to the restored state.

    When ``reference_load_path`` is set (RHO or RefHQ-seeded REL), also pin
    ``reference_content_sha256`` so replacing the file at the same path cannot
    silently resume a different reference. Learnability pins early/late digests
    the same way. Hash keys are omitted when paths are empty so zero-init REL
    fingerprints stay backward-compatible.
    """
    ts_cfg = plan.get("ts_cfg") or {}
    ref_path = str(plan.get("reference_load_path") or "")
    early_path = str(plan.get("early_reference_load_path") or "")
    late_path = str(plan.get("late_reference_load_path") or "")
    ema_seed_mode = str(
        plan.get("ema_seed_mode") or ts_cfg.get("ema_seed_mode") or "zero"
    ).lower()
    fingerprint: Dict[str, Any] = {
        "run_id": str(plan["run_id"]),
        "method": str(plan["method"]),
        "seed": int(plan["seed"]),
        "init_seed": int(plan["init_seed"]),
        "model_name": str(plan["model_name"]),
        "model_arch": str(plan["model_arch"]),
        "olmo_revision": str(plan.get("olmo_revision", "")),
        "tokenizer": str(plan["tokenizer"]),
        "sequence_length": int(plan["sequence_length"]),
        "max_tokens": int(plan["max_tokens"]),
        "global_batch_size": int(plan["global_batch_size"]),
        "lr": float(plan["lr"]),
        "warmup_steps": int(plan["warmup_steps"]),
        "t0_steps": int(plan["t0_steps"]),
        "rel_k": float(ts_cfg.get("k", 0.0)),
        "rel_alpha_start": float(ts_cfg.get("alpha_start", 0.0)),
        "rel_alpha_end": float(ts_cfg.get("alpha_end", 0.0)),
        "alpha_schedule": str(ts_cfg.get("alpha_schedule") or "linear"),
        "alpha_tau": float(ts_cfg.get("alpha_tau") or 300.0),
        "ema_seed_mode": ema_seed_mode,
        "reference_load_path": ref_path,
        "early_reference_load_path": early_path,
        "late_reference_load_path": late_path,
        "order_contract_sha256": str(plan["data_order"]["contract"]["contract_sha256"]),
    }
    from token_selection.scripts.experiment_contract import sha256_file

    def _pin_ref(path_key: str, content_key: str, plan_key: str) -> None:
        path = str(fingerprint.get(path_key) or "")
        if not path:
            return
        content_sha = plan.get(plan_key)
        if not content_sha:
            p = Path(path)
            if not p.exists():
                raise SystemExit(
                    f"{path_key}={path!r} does not exist; cannot fingerprint the "
                    "frozen reference."
                )
            content_sha = sha256_file(p)
        fingerprint[content_key] = str(content_sha)

    _pin_ref("reference_load_path", "reference_content_sha256", "reference_content_sha256")
    _pin_ref(
        "early_reference_load_path",
        "early_reference_content_sha256",
        "early_reference_content_sha256",
    )
    _pin_ref(
        "late_reference_load_path",
        "late_reference_content_sha256",
        "late_reference_content_sha256",
    )
    return fingerprint


def _fingerprint_identity(fp: Mapping[str, Any]) -> Dict[str, Any]:
    """Scientific resume identity: drop host-local provenance fields.

    Reference load paths are allowed to move across machines as long as the
    corresponding ``*_content_sha256`` digests still match.
    """
    drop = {
        "reference_load_path",
        "early_reference_load_path",
        "late_reference_load_path",
    }
    return {k: v for k, v in fp.items() if k not in drop}


def _fingerprints_compatible(prior: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """True when ``current`` may scientifically continue ``prior``."""
    prior_id = _fingerprint_identity(prior)
    current_id = _fingerprint_identity(current)
    if prior_id == current_id:
        return True
    diffs = sorted(k for k in set(prior_id) | set(current_id) if prior_id.get(k) != current_id.get(k))
    # Allow extending the token budget after a completed (or interrupted) segment so
    # a 5B run can later continue to 10B from the same checkpoints.
    return (
        diffs == ["max_tokens"]
        and int(current_id["max_tokens"]) > int(prior_id["max_tokens"])
    )


def _fingerprint_path(plan: Dict[str, Any]) -> Path:
    return Path(plan["save_folder"]) / "run_fingerprint.json"


def _resolve_arm_name(cfg: Mapping[str, Any]) -> str:
    """Return the arm directory name used under ``token-sel/<arm>/`` on S3."""
    arm = str(cfg.get("arm") or "").strip()
    if arm:
        return arm
    from token_selection.olmo_ext.s3_layout import arm_from_prefix

    prefix = str((cfg.get("s3") or {}).get("prefix") or "").strip()
    if not prefix:
        raise SystemExit(
            "cfg.arm or s3.prefix (token-sel/<arm>) is required for durable "
            "checkpoint export / ephemeral resume fetch"
        )
    return arm_from_prefix(prefix)


def _ensure_resume_artifacts(
    plan: Dict[str, Any],
    cfg: Mapping[str, Any],
    method: MethodName,
) -> None:
    """Fetch durable checkpoints (+ metrics) from S3 when local save_folder is empty.

    Ephemeral FarmShare/AWS scratch does not retain a prior job's tokens, order,
    checkpoints, or venvs. ``--resume`` must not assume a local save folder; it
    pulls ``run_fingerprint.json`` and step dirs from ``edullm-checkpoints``.
    """
    from token_selection.olmo_ext.s3_export import (
        fetch_arm_method_checkpoints,
        fetch_arm_method_metrics,
    )

    save_folder = Path(plan["save_folder"])
    fingerprint_path = _fingerprint_path(plan)
    if fingerprint_path.is_file():
        return

    arm = _resolve_arm_name(cfg)
    remote = s3_uri(cfg, "checkpoints", method, bucket_key="checkpoint_bucket")
    print(
        json.dumps(
            {
                "event": "resume_fetch_checkpoints",
                "arm": arm,
                "method": method,
                "remote": remote,
                "local": str(save_folder),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        fetch_arm_method_checkpoints(
            arm, save_folder, method=method, raise_on_error=True
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"--resume: failed to fetch durable checkpoints from {remote}/: {exc}"
        ) from exc

    if not fingerprint_path.is_file():
        raise SystemExit(
            f"--resume set but no run_fingerprint.json under {save_folder} after "
            f"syncing {remote}/. Ephemeral scratch does not persist checkpoints; "
            "keep S3_EXPORT enabled during training so fingerprints + step dirs "
            "land under s3://edullm-checkpoints/token-sel/<arm>/checkpoints/, "
            "then resume on a clean machine. Launch without --resume to start fresh."
        )

    metrics_dir = Path(plan["metrics_dir"])
    fetch_arm_method_metrics(arm, metrics_dir, method=method)


def _export_durable_run_state(
    plan: Dict[str, Any],
    cfg: Mapping[str, Any],
    method: MethodName,
) -> None:
    """Push fingerprint / metrics / method checkpoints to S3 (upload-before-end).

    Mid-run permanent steps also export via ``TaskLossEvalCallback`` (even when
    task-loss eval is disabled). This path covers pre-fit fingerprint + post-fit
    catch-up so ephemeral scratch wipe cannot erase the only copy.
    """
    from token_selection.olmo_ext.s3_export import (
        export_arm_metrics_dir,
        export_arm_run_fingerprint,
        s3_export_enabled,
        sync_to_s3,
    )
    from token_selection.olmo_ext.s3_layout import arm_uri

    try:
        arm = _resolve_arm_name(cfg)
    except SystemExit:
        return
    if not s3_export_enabled():
        print(
            json.dumps(
                {
                    "event": "s3_export_disabled",
                    "warning": (
                        "S3_EXPORT=0 / SKIP_S3_UPLOAD=1 — checkpoints stay local-only "
                        "and will be lost when ephemeral scratch is wiped"
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    ok = True
    fp = _fingerprint_path(plan)
    if fp.is_file():
        ok = bool(export_arm_run_fingerprint(arm, fp, method=method)) and ok
    metrics_dir = Path(plan["metrics_dir"])
    if metrics_dir.exists() and any(metrics_dir.iterdir()):
        ok = bool(export_arm_metrics_dir(arm, metrics_dir, method=method)) and ok
    save_folder = Path(plan["save_folder"])
    if save_folder.exists() and any(save_folder.iterdir()):
        ok = bool(sync_to_s3(save_folder, arm_uri(arm, "checkpoints", str(method)))) and ok
    if not ok:
        raise SystemExit(
            "Durable S3 export failed for fingerprint/metrics/checkpoints under "
            f"token-sel/{arm}/. Fix AWS credentials / aws CLI on the train host, then retry. "
            "Set S3_EXPORT=0 only for intentional non-durable local smoke runs."
        )


def _prepare_run_dir(plan: Dict[str, Any], *, resume: bool) -> None:
    """Enforce fresh-scratch on first launch, or a fingerprint match on resume.

    On a fresh launch the identity is *checked* here but only *committed*
    (``_commit_run_fingerprint``) after the trainer builds successfully, so a build
    that dies (e.g. an OLMo-core API/env error) cannot strand a lone fingerprint that
    would then block a clean relaunch.

    Callers that pass ``resume=True`` must run ``_ensure_resume_artifacts`` first so
    ephemeral hosts can hydrate the save folder from ``edullm-checkpoints``.
    """
    save_folder = Path(plan["save_folder"])
    fingerprint_path = _fingerprint_path(plan)
    current = _run_fingerprint(plan)
    if resume:
        if not fingerprint_path.exists():
            raise SystemExit(
                f"--resume set but no run_fingerprint.json under {save_folder}; there is "
                "nothing to resume. On ephemeral scratch, durable checkpoints must exist "
                "under s3://edullm-checkpoints/ (fetched before this check). "
                "Launch without --resume to start the run."
            )
        prior = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if _fingerprints_compatible(prior, current):
            return
        diffs = sorted(k for k in set(prior) | set(current) if prior.get(k) != current.get(k))
        raise SystemExit(
            "Refusing to resume: the run identity changed since the checkpoint was "
            f"written (differing fields: {diffs}). Resuming would break the matched "
            "REL-vs-full comparison. Allowed on --resume: extending max_tokens upward, "
            "and relocating reference.load_path when reference_content_sha256 matches."
        )
    # Fresh scratch launch: the folder must be empty (the fingerprint is written later).
    _assert_fresh_save_folder(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)


def _commit_run_fingerprint(plan: Dict[str, Any], *, resume: bool) -> None:
    """Persist the run identity once the trainer is known to build.

    On resume, rewrite when ``max_tokens`` was extended or the reference path was
    relocated (same content hash) so a later resume still matches.
    """
    fingerprint_path = _fingerprint_path(plan)
    current = _run_fingerprint(plan)
    if resume:
        if fingerprint_path.exists():
            prior = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            if prior == current:
                return
            if _fingerprints_compatible(prior, current):
                fingerprint_path.write_text(
                    json.dumps(current, indent=2), encoding="utf-8"
                )
        return
    fingerprint_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def build_trainer(plan: Dict[str, Any], cfg: Dict[str, Any], method: MethodName, *, resume: bool = False):
    """Assemble the OLMo-core Trainer for one arm. Returns the built ``trainer``.

    Both methods take the same direct TokenSelectTrainModule construction path. The
    function first rejects unavailable frozen-order wiring. When ``resume``
    is set, the trainer is allowed to reload the latest checkpoint (model + optimizer +
    trainer/callback state, including REL's EMA) from the save folder; otherwise it is
    forbidden from resuming so a scratch arm cannot silently continue.
    """
    _assert_launch_capabilities(plan)
    try:
        import torch.distributed.checkpoint.state_dict as dist_cp_sd  # type: ignore
        from olmo_core.config import DType  # type: ignore
        from olmo_core.data import (  # type: ignore
            NumpyDataLoaderConfig,
            NumpyFSLDatasetConfig,
            TokenizerConfig,
        )
        from olmo_core.distributed.parallel import DataParallelType  # type: ignore
        from olmo_core.nn.transformer import TransformerConfig  # type: ignore
        from olmo_core.optim import AdamWConfig, CosWithWarmup  # type: ignore
        from olmo_core.train import Duration, LoadStrategy, TrainerConfig  # type: ignore
        from olmo_core.train.callbacks import CheckpointerCallback  # type: ignore
        from olmo_core.train.train_module.transformer import (  # type: ignore
            TransformerDataParallelConfig,
            TransformerTrainModuleConfig,
        )
    except ImportError as e:
        raise SystemExit(
            "olmo_core not installed. pip install -e /path/to/edu-llm/OLMo-core\n"
            f"Original error: {e}"
        ) from e

    from token_selection.olmo_ext.train_module import (
            RELCallback,
            TokenSelectConfig,
            TokenSelectTrainModule,
        )
    from token_selection.olmo_ext.callbacks import (
        RawComputeCallback,
        build_metrics_payload,
    )

    ts_cfg = TokenSelectConfig(
        **{k: v for k, v in plan["ts_cfg"].items() if k in TokenSelectConfig.__dataclass_fields__}
    )
    seq_len = int(plan["sequence_length"])
    gbs = int(plan["global_batch_size"])
    seed = int(plan["seed"])
    total_steps = int(plan["total_steps"])

    try:
        # --- data --------------------------------------------------------------------
        # allenai/dolma2-tokenizer is a tokenizer-only HF repo (no model config.json with
        # vocab_size). OLMo-core's from_hf looks for that first and fails closed; use the
        # built-in dolma2() config instead, which matches the corpus sidecars
        # (eos_token_id=100257, vocab_size=100278, pad_token_id=100277).
        tokenizer_id = str(plan["tokenizer"])
        if tokenizer_id in {"allenai/dolma2-tokenizer", "dolma2"} or tokenizer_id.endswith(
            "/dolma2-tokenizer"
        ):
            tokenizer = TokenizerConfig.dolma2()
        else:
            tokenizer = TokenizerConfig.from_hf(tokenizer_id)
        dataset_cfg = NumpyFSLDatasetConfig(
            paths=list(plan["token_paths"]),
            sequence_length=seq_len,
            tokenizer=tokenizer,
            work_dir=str(plan["dataset_cache"]),
        )
        loader_cfg = NumpyDataLoaderConfig(
            global_batch_size=gbs,
            seed=int(plan["data_loader_seed"]),  # identical across arms -> identical order
            num_workers=int(cfg.get("train", {}).get("num_workers", 4)),
        )

        # --- model + optim -----------------------------------------------------------
        arch = str(plan["model_arch"])
        model_builder = getattr(TransformerConfig, arch, None)
        if model_builder is None or not callable(model_builder):
            raise SystemExit(
                f"OLMo-core TransformerConfig has no builder {arch!r}; set model.arch "
                "to a valid olmo2_* size (e.g. olmo2_370M)."
            )
        attn_backend = resolve_attn_backend(cfg)
        model_cfg = model_builder(
            vocab_size=tokenizer.padded_vocab_size(),
            init_seed=int(plan["init_seed"]),
            attn_backend=attn_backend,
        )
        plan["attn_backend"] = getattr(attn_backend, "name", str(attn_backend))
        # Exact, world-size-independent param count so the FLOPs axis and the
        # cross-arm n_params equality check cannot drift between machines/GPU counts.
        n_params = int(model_cfg.num_params)
        optim_name = str(cfg.get("train", {}).get("optim_type", "adamw")).strip().lower()
        lr = float(plan["lr"])
        if optim_name in {"adamw", "adam_w"}:
            optim_cfg = AdamWConfig(lr=lr)
        elif optim_name in {"skip_step_adamw", "skipstepadamw"}:
            from olmo_core.optim import SkipStepAdamWConfig  # type: ignore

            # Match CE/BLADE/RefHQ stack (embeddings WD=0, SkipStepAdamW).
            from olmo_core.optim import OptimGroupOverride  # type: ignore

            optim_cfg = SkipStepAdamWConfig(
                lr=lr,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
                ],
            )
        else:
            raise SystemExit(
                f"train.optim_type={optim_name!r} unsupported; use adamw or skip_step_adamw"
            )
        warmup_steps = int(plan["warmup_steps"])
        alpha_f = float(cfg.get("train", {}).get("lr_alpha_f", 0.1))
        scheduler = CosWithWarmup(warmup_steps=warmup_steps, alpha_f=alpha_f)

        rank_mbz = int(cfg.get("train", {}).get("rank_microbatch_size", seq_len))
        compile_model = bool(cfg.get("train", {}).get("compile_model", False))
        train_cfg = cfg.get("train", {})
        # RefHQ / control controlled knobs — YAML spines declare these; do not leave
        # TransformerTrainModuleConfig defaults (None) for spine arms.
        max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        z_raw = train_cfg.get("z_loss_multiplier", 1e-5)
        z_loss_multiplier = None if z_raw is None else float(z_raw)
        # Default FSDP matches the REL arm. AWS multi-B200 RHO configs may set
        # train.dp_type=hsdp to match the CE/BLADE stack on the same host.
        dp_name_raw = str(cfg.get("train", {}).get("dp_type", "fsdp")).strip().lower()
        try:
            dp_name = DataParallelType[dp_name_raw]
        except KeyError as exc:
            allowed = ", ".join(sorted(t.name for t in DataParallelType))
            raise SystemExit(
                f"train.dp_type={dp_name_raw!r} is not a DataParallelType "
                f"(allowed: {allowed})"
            ) from exc
        if gbs % rank_mbz != 0:
            raise SystemExit(
                f"global_batch_size {gbs} is not divisible by rank_microbatch_size {rank_mbz}"
            )
        train_module_cfg = TransformerTrainModuleConfig(
            rank_microbatch_size=rank_mbz,
            max_sequence_length=seq_len,
            optim=optim_cfg,
            scheduler=scheduler,
            compile_model=compile_model,
            max_grad_norm=max_grad_norm,
            z_loss_multiplier=z_loss_multiplier,
            dp_config=TransformerDataParallelConfig(
                name=dp_name,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
            ),
        )
        if train_module_cfg.pp_config is not None:
            raise SystemExit(
                "TokenSelectTrainModule currently supports the non-pipeline "
                "TransformerTrainModule path only."
            )

        # --- trainer -----------------------------------------------------------------
        # Shared permanent ladder: step 0, every interval (skip last grid if within
        # one interval of final), plus true final. max_checkpoints=None; no ephemeral.
        interval = int(
            train_cfg.get("checkpoint_every_steps", DEFAULT_CHECKPOINT_INTERVAL)
        )
        total_steps = int(plan["total_steps"])
        ckpt_kwargs = checkpointer_kwargs_for_ladder(
            total_steps,
            interval,
            # Default False so the task_loss post-save hook sees a complete step dir.
            save_async=bool(train_cfg.get("save_async", False)),
        )
        # Explicit max_checkpoints=None wins over OLMo-core's default of 3.
        if "checkpoint_keep_last" in train_cfg:
            ckpt_kwargs["max_checkpoints"] = train_cfg.get("checkpoint_keep_last")
        # Opt-in ephemeral only when YAML sets it (plan default: none).
        ephemeral_interval = train_cfg.get("ephemeral_checkpoint_every_steps")
        if ephemeral_interval is not None:
            ckpt_kwargs["ephemeral_save_interval"] = int(ephemeral_interval)
        milestone_steps = train_cfg.get("checkpoint_milestone_steps")
        if milestone_steps is not None:
            ckpt_kwargs["fixed_steps"] = [int(step) for step in milestone_steps]
        if train_cfg.get("pre_train_checkpoint") is not None:
            ckpt_kwargs["pre_train_checkpoint"] = bool(
                train_cfg.get("pre_train_checkpoint")
            )
        plan["permanent_checkpoint_steps"] = permanent_checkpoint_steps(
            total_steps, interval
        )
        trainer_cfg = (
            TrainerConfig(
                save_folder=str(plan["save_folder"]),
                # Resume is gated by a run-fingerprint match in try_launch(). With resume
                # off we forbid save-folder-first resumption so a scratch arm cannot
                # silently continue a prior run. With resume on, fit() auto-loads the
                # latest save-folder checkpoint (trainer + optim + callback/EMA state).
                load_strategy=LoadStrategy.if_available if resume else LoadStrategy.never,
                load_trainer_state=bool(resume),
                load_optim_state=bool(resume),
                max_duration=Duration.tokens(int(plan["max_tokens"])),
            )
            .with_callback("checkpointer", CheckpointerCallback(**ckpt_kwargs))
        )
        # init_id binds everything that determines the initial weights + token space:
        # arch (not just param count), init seed, sequence length, tokenizer, and the
        # pinned OLMo-core revision. A baseline arm that differs in any of these produces
        # a different init_id and compare_runs fails closed instead of comparing apples to
        # oranges (n_params equality alone cannot catch same-size arch/tokenizer drift).
        init_id = hashlib.sha256(
            json.dumps(
                {
                    "model_name": plan["model_name"],
                    "model_arch": plan["model_arch"],
                    "init_seed": plan["init_seed"],
                    "sequence_length": seq_len,
                    "tokenizer": plan["tokenizer"],
                    "olmo_revision": plan.get("olmo_revision", ""),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        # spec_id binds the shared training spec (budget, batch, LR, warmup, tokenizer,
        # arch) so the externally trained `full` arm cannot silently use a different
        # recipe. warmup_steps is included because the LR-warmup schedule directly shapes
        # the loss curve and must match across arms.
        spec_id = hashlib.sha256(
            json.dumps(
                {
                    "max_tokens": int(plan["max_tokens"]),
                    "global_batch_size": gbs,
                    "sequence_length": seq_len,
                    "lr": float(plan["lr"]),
                    "warmup_steps": int(warmup_steps),
                    "model_arch": plan["model_arch"],
                    "tokenizer": plan["tokenizer"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        metrics_payload = build_metrics_payload(
            run_id=str(plan["run_id"]),
            method=method,
            seed=seed,
            ts_config=plan["ts_cfg"],
            t0_tokens=int(plan["t0_steps"]) * gbs if method in _SELECTING else 0,
            order_id=str(plan["data_order"]["contract"]["contract_sha256"]),
            init_id=init_id,
            spec_id=spec_id,
            n_params=n_params,
        )
        trainer_cfg = trainer_cfg.with_callback(
            "raw_metrics",
            RawComputeCallback(
                metrics_path=Path(plan["metrics_dir"]) / str(
                    (cfg.get("eval") or {}).get("metrics_filename", "metrics.json")
                ),
                payload=metrics_payload,
                resume=resume,
            ),
        )
        if method in _SELECTING:
            trainer_cfg = trainer_cfg.with_callback("rel", RELCallback())

        # Immediate full-suite task_loss_bpb on every permanent checkpoint.
        from token_selection.olmo_ext.task_loss_callback import TaskLossEvalCallback

        eval_cfg = (cfg.get("eval") or {}).get("task_loss") or {}
        task_loss_enabled = bool(eval_cfg.get("enabled", True))
        results_dir = Path(
            str(
                eval_cfg.get("results_dir")
                or (
                    Path(plan["metrics_dir"]).parent.parent
                    / "task_loss_results"
                    / str(plan["run_id"])
                )
            )
        )
        if not results_dir.is_absolute():
            results_dir = ROOT / results_dir
        trainer_cfg = trainer_cfg.with_callback(
            "task_loss_eval",
            TaskLossEvalCallback(
                total_steps=total_steps,
                save_folder=plan["save_folder"],
                run_id=str(plan["run_id"]),
                results_dir=results_dir,
                interval=interval,
                enabled=task_loss_enabled,
                command_template=eval_cfg.get("command_template"),
                eval_script=eval_cfg.get("eval_script"),
                arm=cfg.get("arm") or None,
                s3_prefix=str((cfg.get("s3") or {}).get("prefix") or "") or None,
                s3_export=bool(eval_cfg.get("s3_export", True)),
            ),
        )

        # W&B: train scalars via olmo_core WandBCallback; ckpt/eval artifacts via side channel.
        # Soft-skip without WANDB_API_KEY (SmolLM2); never weakens S3 fail-closed export.
        from token_selection.olmo_ext.wandb_logging import (
            apply_wandb_env_defaults,
            ensure_wandb_not_hard_disabled,
            make_wandb_artifacts_callback,
            wandb_callback_kwargs_from_env,
            wandb_enabled,
        )

        arm_name = str(cfg.get("arm") or plan.get("arm") or method)
        apply_wandb_env_defaults(
            project="token-selection",
            run_name=str(plan["run_id"]),
            group=arm_name,
        )
        ensure_wandb_not_hard_disabled()
        wb_cfg = (cfg.get("wandb") or {}) if isinstance(cfg.get("wandb"), dict) else {}
        wb_enabled = wandb_enabled(is_main=True)
        if "enabled" in wb_cfg:
            wb_enabled = bool(wb_cfg["enabled"]) and wb_enabled
        try:
            from olmo_core.train.callbacks import WandBCallback  # type: ignore

            wb_kwargs = wandb_callback_kwargs_from_env(
                run_name=str(plan["run_id"]),
                arm=arm_name,
                method=str(method),
                config={
                    "arm": arm_name,
                    "method": method,
                    "run_id": plan["run_id"],
                    "max_tokens": plan["max_tokens"],
                    "total_steps": total_steps,
                    "s3_prefix": (cfg.get("s3") or {}).get("prefix"),
                },
                enabled=wb_enabled,
            )
            if wb_cfg.get("project"):
                wb_kwargs["project"] = str(wb_cfg["project"])
            if wb_cfg.get("entity"):
                wb_kwargs["entity"] = str(wb_cfg["entity"])
            trainer_cfg = trainer_cfg.with_callback("wandb", WandBCallback(**wb_kwargs))
            trainer_cfg = trainer_cfg.with_callback(
                "wandb_artifacts",
                make_wandb_artifacts_callback(
                    results_dir=results_dir,
                    save_folder=plan["save_folder"],
                    total_steps=total_steps,
                    interval=interval,
                    tokens_per_step=int(plan.get("global_batch_size") or gbs),
                    upload_checkpoint_artifacts=bool(
                        wb_cfg.get("upload_checkpoint_artifacts", True)
                    ),
                ),
            )
        except ImportError:
            print(
                json.dumps(
                    {
                        "status": "wandb_callback_unavailable",
                        "detail": "olmo_core.WandBCallback missing; continuing without W&B",
                    }
                ),
                flush=True,
            )

        # --- build -------------------------------------------------------------------
        dataset = dataset_cfg.build()
        model = model_cfg.build(init_device="meta")
        module_kwargs: Dict[str, Any] = {
            "model": model,
            "optim": train_module_cfg.optim,
            "rank_microbatch_size": train_module_cfg.rank_microbatch_size,
            "max_sequence_length": train_module_cfg.max_sequence_length,
            "compile_model": train_module_cfg.compile_model,
            "float8_config": train_module_cfg.float8_config,
            "dp_config": train_module_cfg.dp_config,
            "tp_config": train_module_cfg.tp_config,
            "cp_config": train_module_cfg.cp_config,
            "ep_config": train_module_cfg.ep_config,
            "ac_config": train_module_cfg.ac_config,
            "z_loss_multiplier": train_module_cfg.z_loss_multiplier,
            "max_grad_norm": train_module_cfg.max_grad_norm,
            "scheduler": train_module_cfg.scheduler,
            "label_ignore_index": train_module_cfg.label_ignore_index,
            "ts_config": ts_cfg,
        }
        if train_module_cfg.autocast_precision is not None:
            module_kwargs["autocast_precision"] = train_module_cfg.autocast_precision.as_pt()
        # Always use sharded DCP (every rank writes). Never full_state_dict=True
        # with a rank-0-only torch.save — that drops HSDP shards (CE/BLADE bug).
        if train_module_cfg.state_dict_save_opts is not None:
            module_kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(
                **train_module_cfg.state_dict_save_opts
            )
        else:
            module_kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(
                full_state_dict=False,
                cpu_offload=True,
            )
        if train_module_cfg.state_dict_load_opts is not None:
            module_kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(
                **train_module_cfg.state_dict_load_opts
            )
        else:
            module_kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(
                full_state_dict=False,
            )
        if train_module_cfg.load_key_mapping is not None:
            module_kwargs["load_key_mapping"] = train_module_cfg.load_key_mapping
        train_module = TokenSelectTrainModule(**module_kwargs)
        data_loader = loader_cfg.build(dataset, dp_process_group=train_module.dp_process_group)

        trainer = trainer_cfg.build(train_module=train_module, data_loader=data_loader)
    except (AttributeError, TypeError) as e:
        raise SystemExit(
            "OLMo-core API mismatch while assembling the trainer. Reconcile the SEAM-marked "
            f"call sites in build_trainer() with the pinned fork.\nOriginal error: {type(e).__name__}: {e}"
        ) from e

    return trainer


def try_launch(plan: Dict[str, Any], cfg: Dict[str, Any], method: MethodName, *, resume: bool = False) -> None:
    """Initialize OLMo-core, build, and fit. Resume is gated by a run-fingerprint match."""
    # Pin before any CUDA context: must precede prepare_training_environment / build_trainer.
    pin_cuda_visible_devices(cfg)
    if resume:
        _ensure_resume_artifacts(plan, cfg, method)
    _prepare_run_dir(plan, resume=resume)
    try:
        from olmo_core.train import prepare_training_environment, teardown_training_environment  # type: ignore
        from olmo_core.utils import seed_all  # type: ignore
        import torch
    except ImportError as e:
        raise SystemExit(f"olmo_core not installed: {e}") from e

    visible = [
        p.strip()
        for p in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if p.strip() != ""
    ]
    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        if not visible:
            raise SystemExit("CUDA_VISIBLE_DEVICES empty after pin; refusing to launch")
        if n_visible != len(visible):
            raise SystemExit(
                f"Expected {len(visible)} visible CUDA device(s) after pinning "
                f"{','.join(visible)!r}, found {n_visible}. Check CUDA_VISIBLE_DEVICES."
            )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > n_visible:
            raise SystemExit(
                f"WORLD_SIZE={world_size} exceeds visible CUDA devices ({n_visible}). "
                "Set --nproc_per_node to the number of pinned GPUs."
            )
        gbs = int(plan["global_batch_size"])
        rank_mbz = int(cfg.get("train", {}).get("rank_microbatch_size", plan["sequence_length"]))
        if world_size * rank_mbz == 0 or gbs % (world_size * rank_mbz) != 0:
            raise SystemExit(
                f"global_batch_size {gbs} not divisible by world_size*rank_microbatch_size "
                f"({world_size}*{rank_mbz})"
            )

    prepare_training_environment(seed=int(plan["init_seed"]))
    # Match RefHQ / control: high TF32 matmul precision for bf16 train parity.
    torch.set_float32_matmul_precision("high")
    seed_all(int(plan["init_seed"]))
    try:
        trainer = build_trainer(plan, cfg, method, resume=resume)
        # Only now that the trainer is known to build do we pin the fresh-run identity,
        # so a failed build cannot leave a stale fingerprint that blocks relaunch.
        _commit_run_fingerprint(plan, resume=resume)
        # Durable copy: ephemeral scratch is not the source of truth for resume.
        _export_durable_run_state(plan, cfg, method)
        print(
            json.dumps(
                {
                    "status": "fitting",
                    "method": method,
                    "init_mode": plan["init_mode"],
                    "resume": resume,
                    "save_folder": plan["save_folder"],
                    "s3_checkpoints": plan.get("s3_checkpoints"),
                    "max_tokens": plan["max_tokens"],
                    "total_steps": plan["total_steps"],
                    "num_visible_gpus": len(visible),
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                },
                indent=2,
            )
        )
        trainer.fit()
        # Final metrics push after fit (checkpoints already exported per permanent save).
        _export_durable_run_state(plan, cfg, method)
    finally:
        teardown_training_environment()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "rho-1/configs/run_rho_10b.yaml",
    )
    ap.add_argument(
        "--method",
        choices=[
            "full",
            "rel_ema",
            "rho_excess",
            "middle_ppl",
            "attention_topk",
            "learnability",
        ],
        required=True,
    )
    ap.add_argument(
        "--olmo-root",
        type=Path,
        default=None,
        help="Optional pinned OLMo-core checkout to verify before launch.",
    )
    ap.add_argument(
        "--launch",
        action="store_true",
        help="Launch only when pinned public APIs satisfy frozen-order controls",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the latest checkpoint iff its run fingerprint "
            "(init/order/batching/budget) matches. On ephemeral scratch, fetches "
            "fingerprints + step dirs from s3://edullm-checkpoints/token-sel/<arm>/ "
            "when the local save folder is empty. Do not rely on scratch-only ckpts."
        ),
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    try:
        validate_scratch_config(cfg, method=args.method)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.launch and args.olmo_root is None:
        raise SystemExit(
            "--launch requires --olmo-root pointing at the pinned OLMo-core checkout so the "
            "revision can be verified. The installed package's revision is otherwise unchecked "
            "and both arms must train on the identical pinned framework."
        )
    if args.olmo_root is not None:
        revision = str((cfg.get("olmo_core") or {}).get("revision") or "")
        if not revision:
            raise SystemExit("olmo_core.revision must be set when --olmo-root is supplied")
        try:
            verify_olmo_revision(args.olmo_root, revision)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    method: MethodName = args.method  # type: ignore[assignment]
    allowed = cfg.get("methods") or [
        "full",
        "rel_ema",
        "rho_excess",
        "middle_ppl",
        "attention_topk",
        "learnability",
    ]
    if method not in allowed:
        raise SystemExit(f"method {method!r} not in config methods {allowed}")

    # Always resolve the published corpus (fail closed on legacy edullm-datasets).
    # On --launch, also stage shards + order contract onto the run dir (clean machine).
    remote = prepare_train_data(cfg, out, stage=bool(args.launch))
    print(
        json.dumps(
            {
                "event": "train_corpus_resolved",
                "dataset_id": remote["dataset_id"],
                "version": remote["version"],
                "tokens_uri": remote["tokens_uri"],
                "dtype": remote["dtype"],
                "rows": remote["rows"],
                "n_shards": len(remote["paths"]),
                "staged": bool(args.launch),
            },
            indent=2,
        ),
        flush=True,
    )

    # On --launch, materialize null RefHQ load_paths from YAML s3_uri(s) into the
    # shared local cache (idempotent; multi-rank safe via mkdir lock).
    if args.launch and method in ("rho_excess", "rel_ema", "learnability", "middle_ppl"):
        from token_selection.olmo_ext.refhq_materialize import ensure_reference_paths

        try:
            filled = ensure_reference_paths(cfg, method=method)
        except Exception as exc:  # noqa: BLE001 — surface as SystemExit before train
            raise SystemExit(f"RefHQ reference materialize failed: {exc}") from exc
        if filled:
            print(
                json.dumps({"event": "refhq_materialized", "paths": filled}, indent=2),
                flush=True,
            )
        try:
            validate_scratch_config(cfg, method=method)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if args.launch:
        plan = build_plan(cfg, method=method, out=out)
        try_launch(plan, cfg, method, resume=args.resume)
    else:
        # Plan-only: emit the resolved remote identity without requiring local shards.
        plan = {
            "run_id": cfg.get("run_id"),
            "method": method,
            "init_mode": "scratch",
            "dataset_id": remote["dataset_id"],
            "dataset_version": remote["version"],
            "s3_tokens": remote["tokens_uri"],
            "token_dtype": remote["dtype"],
            "token_rows": remote["rows"],
            "token_shards": remote["paths"],
            "tokenizer": (cfg.get("data") or {}).get("tokenizer"),
            "notes": [
                "Plan-only: corpus resolved from edullm-data; local shards not required.",
                "Pass --launch to stage tokens + order onto output_dir and fit.",
            ],
        }
        # If tokens were already staged (optional), enrich with a full build_plan.
        if (out / "tokens" / "manifest.json").exists() and (
            out / "order" / "manifest.json"
        ).exists():
            plan = build_plan(cfg, method=method, out=out)
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
