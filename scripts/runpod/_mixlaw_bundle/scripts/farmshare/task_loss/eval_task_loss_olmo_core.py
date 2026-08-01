#!/usr/bin/env python3
"""OLMo-ladder task-loss (RC 5-shot bpb) for olmo-core OLMo2-370M checkpoints."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.distributed as dist

# Compat for torch<2.7: olmo-core passes reason= to torch.compiler.disable.
_orig_torch_compiler_disable = torch.compiler.disable


def _torch_compiler_disable_compat(fn=None, recursive=True, **kwargs):
    kwargs.pop("reason", None)
    if fn is None:
        return lambda f: _orig_torch_compiler_disable(f, recursive=recursive)
    return _orig_torch_compiler_disable(fn, recursive=recursive)


torch.compiler.disable = _torch_compiler_disable_compat  # type: ignore[assignment]

from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig
from olmo.eval import build_evaluator
from olmo.tokenizer import Tokenizer
from olmo.torch_util import get_local_rank
from olmo.util import prepare_cli_environment

try:
    from olmo.util import add_cached_path_clients
except ImportError:  # ai2-olmo builds that dropped this helper
    def add_cached_path_clients() -> None:  # type: ignore[misc]
        return None

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import unshard_checkpoint

log = logging.getLogger("eval_task_loss_olmo_core")
_T = TypeVar("_T")


def _ensure_hf_auth() -> None:
    """Make Hub downloads authenticated on every rank (torchrun can drop env)."""
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        # Fall back to the on-disk token written by the RunPod launch script.
        for path in (
            Path.home() / ".cache" / "huggingface" / "token",
            Path("/root/.cache/huggingface/token"),
            Path("/workspace/hf-session.env"),
        ):
            try:
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8").strip()
                if path.name.endswith(".env"):
                    for line in text.splitlines():
                        if line.startswith("export HF_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip("'\"")
                            break
                else:
                    token = text
                if token:
                    break
            except OSError:
                continue
    if not token:
        log.warning("no HF_TOKEN available; Hub downloads may be rate-limited")
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        log.info("huggingface_hub login ok (token present)")
    except Exception as exc:  # noqa: BLE001
        log.warning("huggingface_hub login failed: %s", exc)


# Full OLMo-ladder RC 5-shot suite (20 labels). Do not substitute accuracy/CE.
TASK_LABELS = [
    "arc_challenge_val_rc_5shot_bpb",
    "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
]

EMBEDDING_SIZE = 100_352


def build_model() -> torch.nn.Module:
    try:
        backend = AttentionBackendName.torch
    except Exception:
        backend = None
    kwargs: dict[str, Any] = {"vocab_size": EMBEDDING_SIZE}
    if backend is not None:
        kwargs["attn_backend"] = backend
    cfg = TransformerConfig.olmo2_370M(**kwargs)
    model = cfg.build(init_device="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _patch_dtensor_unpickle() -> None:
    import torch.distributed.tensor as tdt
    if hasattr(tdt, "DTensor"):
        return
    _dt_mod = importlib.import_module("torch.distributed._tensor")
    _DT = getattr(_dt_mod, "DTensor")
    tdt.DTensor = _DT  # type: ignore[attr-defined]
    log.info("Patched torch.distributed.tensor.DTensor -> %s", _DT)


def _to_local_tensor(t: Any) -> torch.Tensor:
    if torch.is_tensor(t):
        full = getattr(t, "full_tensor", None)
        if callable(full):
            return full().detach().cpu()
        local = getattr(t, "to_local", None)
        if callable(local):
            return local().detach().cpu()
        return t.detach().cpu()
    raise TypeError(f"expected tensor, got {type(t)}")


def _extract_model_state(train_module_sd: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" in train_module_sd and isinstance(train_module_sd["model"], dict):
        nested = train_module_sd["model"]
        if nested and all(hasattr(v, "shape") or torch.is_tensor(v) for v in nested.values()):
            return {k: _to_local_tensor(v) for k, v in nested.items()}
    prefixed = {
        k[len("model.") :]: _to_local_tensor(v)
        for k, v in train_module_sd.items()
        if k.startswith("model.") and (torch.is_tensor(v) or hasattr(v, "shape"))
    }
    if prefixed:
        return prefixed
    if train_module_sd and all(
        torch.is_tensor(v) or hasattr(v, "shape") for v in train_module_sd.values()
    ):
        return {k: _to_local_tensor(v) for k, v in train_module_sd.items()}
    raise RuntimeError(
        "Could not locate model tensors in train_module state_dict; "
        f"top keys={sorted(train_module_sd.keys())[:30]}"
    )


def load_state_pt(checkpoint_dir: Path, model: torch.nn.Module) -> int:
    eval_pt = checkpoint_dir / "model_eval.pt"
    if eval_pt.is_file():
        payload = torch.load(eval_pt, map_location="cpu", weights_only=False)
        step = int(payload.get("step") or (checkpoint_dir / "step.txt").read_text().strip())
        model_sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        emb = model_sd.get("embeddings.weight") if isinstance(model_sd, dict) else None
        if emb is not None and tuple(emb.shape) != (EMBEDDING_SIZE, 1024):
            raise RuntimeError(
                f"Bad embeddings.weight shape {tuple(emb.shape)}; expected "
                f"({EMBEDDING_SIZE}, 1024). This usually means an HSDP shard-only "
                f"checkpoint (missing gather on save)."
            )
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing:
            log.warning("Missing %d keys (showing 8): %s", len(missing), missing[:8])
        if unexpected:
            log.warning("Unexpected %d keys (showing 8): %s", len(unexpected), unexpected[:8])
        if len(missing) > max(4, 0.05 * (len(model_sd) + len(missing))):
            raise RuntimeError(f"Too many missing keys ({len(missing)}); aborting")
        return step
    path = checkpoint_dir / "state.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    _patch_dtensor_unpickle()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step") or (checkpoint_dir / "step.txt").read_text().strip())
    tm_sd = ckpt["train_module"]
    model_sd = _extract_model_state(tm_sd)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        log.warning("Missing %d keys (showing 8): %s", len(missing), missing[:8])
    if unexpected:
        log.warning("Unexpected %d keys (showing 8): %s", len(unexpected), unexpected[:8])
    if len(missing) > max(4, 0.05 * (len(model_sd) + len(missing))):
        raise RuntimeError(f"Too many missing keys ({len(missing)}); aborting")
    return step


def materialize_distcp_model_eval(checkpoint_dir: Path) -> Path:
    """Unshard distcp → model_eval.pt without an active process group."""
    if dist.is_initialized():
        raise RuntimeError(
            "materialize_distcp_model_eval must run before dist.init_process_group "
            "(olmo-core unshard_checkpoint forbids a distributed context)"
        )
    out = checkpoint_dir / "model_eval.pt"
    if out.is_file():
        return out
    model_and_optim = checkpoint_dir / "model_and_optim"
    if not (model_and_optim / ".metadata").is_file():
        raise FileNotFoundError(f"missing distcp metadata under {model_and_optim}")
    step_txt = checkpoint_dir / "step.txt"
    if step_txt.is_file():
        step = int(step_txt.read_text().strip())
    else:
        step = int(checkpoint_dir.name.replace("step", "").split("-")[0])
    tmp = Path(tempfile.mkdtemp(prefix="olmo_core_unshard_"))
    try:
        unshard_checkpoint(
            dir=str(model_and_optim),
            target_dir=str(tmp),
            optim=False,
            save_overwrite=True,
        )
        candidates = [
            tmp / "model.pt",
            tmp / "model.pth",
            tmp / "model_and_optim" / "model.pt",
        ]
        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            pts = sorted(tmp.rglob("*.pt"))
            if not pts:
                raise RuntimeError(f"unshard produced no .pt under {tmp}")
            src = pts[0]
            log.warning("Using fallback unsharded file %s", src)
        model_sd = torch.load(src, map_location="cpu", weights_only=False)
        if isinstance(model_sd, dict) and "model" in model_sd and isinstance(model_sd["model"], dict):
            model_sd = model_sd["model"]
        # Drop non-tensor entries; keep plain CPU tensors.
        clean: dict[str, torch.Tensor] = {}
        for k, v in model_sd.items():
            if torch.is_tensor(v):
                clean[k] = v.detach().cpu()
        torch.save({"step": step, "model": clean}, out)
        log.info("Materialized %s (%d tensors, step=%s)", out, len(clean), step)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def load_distcp(checkpoint_dir: Path, model: torch.nn.Module) -> int:
    eval_pt = checkpoint_dir / "model_eval.pt"
    if not eval_pt.is_file():
        if dist.is_initialized():
            raise RuntimeError(
                f"missing {eval_pt}; run materialize_distcp_model_eval before "
                "dist.init_process_group, or pass a pre-built model_eval.pt"
            )
        materialize_distcp_model_eval(checkpoint_dir)
    return load_state_pt(checkpoint_dir, model)


def model_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        out = model(input_ids, return_logits=True)
    if hasattr(out, "logits"):
        return out.logits
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple) and torch.is_tensor(out[0]):
        return out[0]
    raise RuntimeError(f"Unexpected model output type: {type(out)}")


def _require_task_labels(labels: list[str]) -> None:
    """Fail early if installed ai2-olmo lacks the OLMES bpb labels we need."""
    from olmo.eval.downstream import label_to_task_map

    missing = [lab for lab in labels if lab not in label_to_task_map]
    if not missing:
        return
    sample = sorted(k for k in label_to_task_map if "arc_easy" in k)[:12]
    raise RuntimeError(
        "ai2-olmo label_to_task_map is missing required task-loss labels "
        f"(first missing={missing[0]!r}, missing={len(missing)}/{len(labels)}). "
        "Install ai2-olmo from GitHub main (PyPI 0.6.0 is too old). "
        f"arc_easy keys present={sample}"
    )


def evaluate_label(
    model: torch.nn.Module,
    cfg: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    label: str,
    device_eval_batch_size: int,
) -> tuple[float, int]:
    log.info("build_evaluator(%s) ...", label)
    evaluator = build_evaluator(
        cfg,
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            device_eval_batch_size=device_eval_batch_size,
            # Omit subset_num_batches (schema default) — some ai2-olmo builds
            # type it as plain int and reject an explicit None.
        ),
        tokenizer,
        device,
    )
    log.info("build_evaluator(%s) done; running batches ...", label)
    evaluator.reset_metrics()
    n_batches = 0
    loader_iter = iter(evaluator.eval_loader)
    while True:
        if n_batches == 0:
            log.info("%s: fetching first batch ...", label)
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        if n_batches == 0:
            ids = batch.get("input_ids")
            shape = tuple(ids.shape) if torch.is_tensor(ids) else None
            log.info("%s: first batch ok shape=%s; forward ...", label, shape)
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model_logits(model, batch["input_ids"])
        if n_batches == 0:
            log.info("%s: first forward ok logits=%s", label, tuple(logits.shape))
        evaluator.eval_metric.update(batch, logits)
        n_batches += 1
        if n_batches <= 3 or n_batches % 20 == 0:
            log.info("%s: completed %d batches", label, n_batches)
    log.info("%s: all batches done (%d); computing metrics ...", label, n_batches)
    metrics = evaluator.compute_metrics()
    log.info("%s: compute_metrics done", label)
    (value,) = list(metrics.values())
    return float(value), n_batches


# Train-only CPT fields that commonly break older ai2-olmo schemas during eval.
# Downstream task-loss only needs tokenizer + model pad/eos ids.
_EVAL_DROP_TOP_KEYS = frozenset(
    {
        "evaluators",
        "data",
        "optimizer",
        "scheduler",
        "wandb",
        "fsdp",
        "ddp",
        "compile",
        "sharded_checkpointer",
        "load_path",
        "load_path_sharded_checkpointer",
        "save_folder",
        "remote_save_folder",
        "activation_checkpointing",
        "fused_loss",
        "new_style_checkpoints",
        "try_load_latest_save",
        "force_save_unsharded",
        "no_pre_train_checkpoint",
        "reset_optimizer_state",
        "reset_trainer_state",
        "save_data_indices",
        "python_profiling",
        "torch_profiling",
        "module_outputs_save_steps",
        "hf_datasets_cache_dir",
        "speed_monitor",
        "gen1_gc_interval",
        "softmax_auxiliary_loss",
        "auxiliary_loss_multiplier",
        "distributed_strategy",
        "max_duration",
        "device_train_batch_size",
        "device_train_grad_accum",
        "fast_forward_batches",
        "restore_dataloader",
        "dry_run",
        "epoch",
        "time_limit",
        "early_stopping_factor",
        "stop_at",
        "stop_after",
        "canceled_check_interval",
        "extra_steps_after_cancel",
        "save_interval",
        "save_interval_unsharded",
        "save_interval_ephemeral",
        "save_num_checkpoints_to_keep",
        "save_num_unsharded_checkpoints_to_keep",
        "save_overwrite",
        "eval_interval",
        "eval_on_load",
        "console_log_interval",
        "max_grad_norm",
        "max_grad_norm_ratio",
        "precision",
    }
)


def _strip_nulls(obj: Any) -> Any:
    """Drop YAML nulls so OmegaConf can fall back to schema defaults."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


def _filter_to_schema(raw: Any, schema: Any) -> Any:
    """Keep only keys present in an OmegaConf structured schema."""
    from omegaconf import OmegaConf as om

    if not isinstance(raw, dict) or not om.is_dict(schema):
        return raw
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in schema:
            continue
        child = schema[key]
        if isinstance(value, dict) and om.is_config(child):
            out[key] = _filter_to_schema(value, child)
        else:
            out[key] = value
    return out


def _drop_dotted_key(raw: dict[str, Any], dotted: str) -> None:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return
    cur: Any = raw
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def _minimal_eval_train_config(raw: dict[str, Any] | None = None) -> TrainConfig:
    """Tokenizer-centric TrainConfig that works across ai2-olmo schema skew."""
    from olmo.config import ModelConfig, TokenizerConfig

    tok_id = "allenai/dolma2-tokenizer"
    eos_id, pad_id = 100257, 100277
    vocab_size, embedding_size = 100278, 100352
    if isinstance(raw, dict):
        tok = raw.get("tokenizer") or {}
        model = raw.get("model") or {}
        if isinstance(tok, dict) and tok.get("identifier"):
            tok_id = str(tok["identifier"])
        if isinstance(model, dict):
            if model.get("eos_token_id") is not None:
                eos_id = int(model["eos_token_id"])
            if model.get("pad_token_id") is not None:
                pad_id = int(model["pad_token_id"])
            if model.get("vocab_size") is not None:
                vocab_size = int(model["vocab_size"])
            if model.get("embedding_size") is not None:
                embedding_size = int(model["embedding_size"])
    cfg = TrainConfig(
        model=ModelConfig(
            d_model=1024,
            n_heads=16,
            n_layers=16,
            mlp_ratio=8,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        ),
        tokenizer=TokenizerConfig(identifier=tok_id),
        global_train_batch_size=8,
        device_train_microbatch_size=1,
        device_eval_batch_size=4,
        seed=6198,
    )
    cfg.evaluators = []
    return cfg


def _load_train_config_file(path: Path) -> TrainConfig:
    """Load ai2-olmo TrainConfig across schema skew with CPT/ladder YAMLs."""
    import re

    import yaml
    from omegaconf import OmegaConf as om
    from omegaconf.errors import OmegaConfBaseException

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected mapping in {path}, got {type(raw)}")
    raw = _strip_nulls(raw)
    for key in _EVAL_DROP_TOP_KEYS:
        raw.pop(key, None)
    try:
        TrainConfig._register_resolvers(validate_paths=False)
    except TypeError:
        TrainConfig._register_resolvers()
    schema = om.structured(TrainConfig)
    filtered = _filter_to_schema(raw, schema)

    # Drop enum/type mismatches iteratively (e.g. sharded_checkpointer=olmo_core).
    for _ in range(32):
        try:
            conf = om.merge(schema, om.create(filtered))
            cfg = om.to_object(conf)  # type: ignore[assignment]
            cfg.evaluators = []
            return cfg  # type: ignore[return-value]
        except OmegaConfBaseException as exc:
            msg = str(exc)
            m = re.search(r"full_key:\s*([^\s]+)", msg)
            if not m:
                log.warning("TrainConfig merge failed without key path (%s); using minimal", exc)
                return _minimal_eval_train_config(raw)
            dotted = m.group(1)
            log.warning("Dropping incompatible config key %s (%s)", dotted, exc)
            _drop_dotted_key(filtered, dotted)
    log.warning("Exhausted incompatible-key retries for %s; using minimal eval config", path)
    return _minimal_eval_train_config(raw)


def build_train_config(base_config: Path | None) -> TrainConfig:
    candidates = []
    if base_config is not None:
        candidates.append(base_config)
    env_cfg = os.environ.get("LADDER_BASE_CONFIG", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg))
    last_err: Exception | None = None
    for c in candidates:
        if c and c.is_file():
            try:
                return _load_train_config_file(c)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("Compatible TrainConfig load failed for %s: %s", c, exc)
                try:
                    import yaml

                    raw = yaml.safe_load(c.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        log.warning("Falling back to minimal eval TrainConfig from %s", c)
                        return _minimal_eval_train_config(raw)
                except Exception as minimal_exc:  # noqa: BLE001
                    log.warning("Minimal eval config from %s failed: %s", c, minimal_exc)
    # Last resort: hard-coded dolma2 tokenizer ids (OLMo2-370M).
    log.warning(
        "No usable ladder base config%s; using hard-coded dolma2 eval TrainConfig",
        f" (last error: {last_err})" if last_err else "",
    )
    return _minimal_eval_train_config(None)


def detect_checkpoint_format(checkpoint: Path, fmt: str = "auto") -> str:
    if fmt != "auto":
        return fmt
    if (checkpoint / "model_eval.pt").is_file() or (checkpoint / "state.pt").is_file():
        return "state_pt"
    if (checkpoint / "model_and_optim" / ".metadata").is_file():
        return "distcp"
    raise SystemExit(f"Cannot detect checkpoint format under {checkpoint}")


def _aggregate_task_loss_results(results: dict[str, float]) -> dict[str, float]:
    out = dict(results)
    mmlu_keys = [k for k in out if k.startswith("mmlu_") and k.endswith("_rc_5shot_bpb")]
    if mmlu_keys:
        out["mmlu_avg_rc_5shot_bpb"] = sum(out[k] for k in mmlu_keys) / len(mmlu_keys)
    core_keys = [
        "hellaswag_val_rc_5shot_bpb",
        "arc_challenge_test_rc_5shot_bpb",
        "arc_easy_test_rc_5shot_bpb",
        "piqa_val_rc_5shot_bpb",
        "csqa_val_rc_5shot_bpb",
        "socialiqa_val_rc_5shot_bpb",
        "openbookqa_test_rc_5shot_bpb",
        "boolq_val_rc_5shot_bpb",
        "winogrande_val_rc_5shot_bpb",
    ]
    present = [k for k in core_keys if k in out]
    if present:
        out["core_avg_rc_5shot_bpb"] = sum(out[k] for k in present) / len(present)
    raw = [float(out[k]) for k in TASK_LABELS if k in out]
    if len(raw) == len(TASK_LABELS):
        # Scientific contract: macro BPB is exactly the 20 raw ladder labels.
        # Derived MMLU/core aggregates must never receive extra macro weight.
        out["macro_mean_task_loss_bpb"] = sum(raw) / len(TASK_LABELS)
    return out


def run_task_loss_eval_distributed(
    checkpoint: Path,
    out: Path,
    run_name: str,
    *,
    base_config: Path | None = None,
    device_eval_batch_size: int = 4,
    fmt: str = "auto",
) -> dict[str, Any]:
    """Data-sharded OLMo-ladder task-loss on an **initialized** process group.

    Intended for in-training pause/free/eval/reload: callers must free the live
    FSDP train module first so each rank can hold a full eval replica. All ranks
    evaluate every label in lockstep while each evaluator loader shards examples.
    Does not init or destroy the process group. Rank 0 writes ``out`` JSON.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "run_task_loss_eval_distributed requires an initialized process group"
        )

    checkpoint = Path(checkpoint)
    out = Path(out)
    fmt = detect_checkpoint_format(checkpoint, fmt)
    if fmt == "distcp":
        # Cannot unshard after PG init; require a pre-materialized model_eval.pt.
        if not (checkpoint / "model_eval.pt").is_file():
            raise RuntimeError(
                f"distcp checkpoint {checkpoint} has no model_eval.pt; "
                "materialize before process-group init or pass state.pt"
            )
        fmt = "state_pt"

    prepare_cli_environment()
    add_cached_path_clients()
    _ensure_hf_auth()
    _require_task_labels(TASK_LABELS)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = get_local_rank()
    torch.cuda.set_device(f"cuda:{local_rank}")
    device = torch.device("cuda")

    log.info("Building olmo2_370M on %s (rank=%s/%s)", device, rank, world_size)
    model = build_model()
    step = load_state_pt(checkpoint, model)
    model.to(device)
    log.info("Loaded checkpoint step=%s from %s", step, checkpoint)

    cfg = build_train_config(base_config)
    cfg.device_eval_batch_size = device_eval_batch_size
    # Serialize Hub tokenizer fetch: 8 ranks hitting HF/cache locks at once
    # stalls with no logs (GPU util high / mem 0). Rank 0 warms cache first.
    if rank == 0:
        log.info("loading tokenizer on rank 0 (warms Hub cache)...")
        tokenizer = Tokenizer.from_train_config(cfg)
        log.info("tokenizer ready on rank 0")
    dist.barrier()
    if rank != 0:
        log.info("loading tokenizer on rank %s from cache...", rank)
        tokenizer = Tokenizer.from_train_config(cfg)
    dist.barrier()
    log.info("all ranks have tokenizer; warming eval datasets on rank 0...")
    # Same stall pattern as tokenizer: 8 ranks calling build_evaluator() hammer
    # HF datasets / cache locks with no logs. Rank 0 materializes every label first.
    if rank == 0:
        for warm_label in TASK_LABELS:
            log.info("warm dataset %s ...", warm_label)
            ev = build_evaluator(
                cfg,
                EvaluatorConfig(
                    label=warm_label,
                    type=EvaluatorType.downstream,
                    device_eval_batch_size=device_eval_batch_size,
                ),
                tokenizer,
                device,
            )
            # Touch the loader so HF parquet/json actually downloads.
            try:
                next(iter(ev.eval_loader))
            except StopIteration:
                pass
            del ev
        log.info("dataset warm done (%d labels)", len(TASK_LABELS))
    dist.barrier()
    log.info("all ranks: lockstep label eval (DistributedSampler + ICLMetric sync)")

    # OLMo downstream evaluators use DistributedSampler(world_size) and
    # ICLMetric(sync_on_compute=True). Every rank must build_evaluator, iterate
    # its shard, and call compute() together — rank-0-only deadlocks on metric
    # sync (seen as hang after "completed 1 batches" with idle GPUs).
    local_results: dict[str, float] = {}
    try:
        for label in TASK_LABELS:
            log.info("rank=%s evaluating %s ...", rank, label)
            bpb, n_batches = evaluate_label(
                model, cfg, tokenizer, device, label, device_eval_batch_size
            )
            local_results[label] = bpb
            if rank == 0:
                log.info(
                    "%s rank=0 %s: task_loss_bpb=%.6f (%d batches)",
                    run_name,
                    label,
                    bpb,
                    n_batches,
                )
                print(f"rank0\t{label}\t{bpb:.6f}", flush=True)

        gathered: list[Optional[dict[str, float]]] = [None] * world_size
        dist.all_gather_object(gathered, local_results)
        results: dict[str, float] = {}
        for part in gathered:
            if part:
                results.update(part)
        missing = [label for label in TASK_LABELS if label not in results]
        if missing:
            raise RuntimeError(
                "task-loss suite incomplete after all-rank evaluation: "
                f"missing {len(missing)}/{len(TASK_LABELS)} raw labels "
                f"(first={missing[0]!r})"
            )
        raw_results = {label: float(results[label]) for label in TASK_LABELS}
        results = _aggregate_task_loss_results(raw_results)
        macro_bpb = sum(raw_results.values()) / len(TASK_LABELS)
        payload: dict[str, Any] = {
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "format": fmt,
            "step": step,
            "world_size": world_size,
            "task_loss_bpb": results,
            "labels": raw_results,
            "macro_mean": macro_bpb,
            "raw_label_count": len(TASK_LABELS),
            "suite_complete": True,
        }
        if rank == 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload, indent=2), flush=True)
        dist.barrier()
        return payload
    finally:
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def pause_eval_reload_distributed(
    checkpoint: Path,
    out: Path,
    run_name: str,
    *,
    release_train_state: Optional[Callable[[], None]],
    reload_train_state: Callable[[], _T],
    base_config: Path | None = None,
    device_eval_batch_size: int = 4,
    fmt: str = "auto",
    strict: bool = True,
) -> tuple[_T, Optional[dict[str, Any]]]:
    """Release training state, run lockstep eval, then always rebuild/reload it.

    Every rank must call this entry point. ``release_train_state`` should break
    trainer/model references that keep FSDP allocations alive. The reload
    callback runs even when evaluation fails. In strict mode the eval exception
    is re-raised only after training state has been restored; smoke callers may
    opt into soft failure with ``strict=False``.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "pause_eval_reload_distributed requires an initialized process group"
        )

    dist.barrier()
    if release_train_state is not None:
        release_train_state()
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    dist.barrier()

    payload: Optional[dict[str, Any]] = None
    eval_error: Optional[Exception] = None
    try:
        payload = run_task_loss_eval_distributed(
            checkpoint,
            out,
            run_name,
            base_config=base_config,
            device_eval_batch_size=device_eval_batch_size,
            fmt=fmt,
        )
    except Exception as exc:  # reload is mandatory even when eval fails
        eval_error = exc
        log.exception("task-loss evaluation failed; restoring training state")
    finally:
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        dist.barrier()

    restored = reload_train_state()
    dist.barrier()
    if eval_error is not None:
        if strict:
            raise RuntimeError(
                f"task-loss evaluation failed for {checkpoint}; training state was restored"
            ) from eval_error
        log.error(
            "task-loss evaluation failed for %s; continuing because strict=False",
            checkpoint,
        )
    return restored, payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--format", choices=("state_pt", "distcp", "auto"), default="auto")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device-eval-batch-size", type=int, default=4)
    ap.add_argument("--base-config", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fmt = detect_checkpoint_format(args.checkpoint, args.format)

    # olmo-core unshard_checkpoint cannot run after process-group init.
    # With torchrun, only LOCAL_RANK 0 materializes; peers wait for the file.
    local_rank_env = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if fmt == "distcp" and not (args.checkpoint / "model_eval.pt").is_file():
        eval_pt = args.checkpoint / "model_eval.pt"
        if local_rank_env == 0:
            log.info("Materializing distcp → model_eval.pt (before dist init)")
            materialize_distcp_model_eval(args.checkpoint)
        else:
            for _ in range(3600):
                if eval_pt.is_file():
                    break
                time.sleep(1)
            else:
                raise SystemExit(f"timed out waiting for {eval_pt}")
        fmt = "state_pt"
    elif fmt == "distcp":
        fmt = "state_pt"

    local_rank = get_local_rank()
    torch.cuda.set_device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    run_task_loss_eval_distributed(
        args.checkpoint,
        args.out,
        args.run_name,
        base_config=args.base_config,
        device_eval_batch_size=args.device_eval_batch_size,
        fmt=fmt,
    )


if __name__ == "__main__":
    main()

