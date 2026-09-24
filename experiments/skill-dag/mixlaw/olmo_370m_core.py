#!/usr/bin/env python3
"""Shared OLMo2-370M model, checkpoint and bookkeeping code.

The MixLaw and Skill-It 370M trainers
(``train_mixlaw_validation_370m.py`` and ``../skillit/train_skillit_370m.py``)
import their recipe constants, ``build_train_module``, checkpoint
save/load/gather, resume staging (``stage_load_path``), the ``_Bookkeeping``
dataclass and the all-rank abort helper from here.

Every definition below is copied unchanged from the trainer both were
originally built on, ``experiments/curriculum/train_curriculum_regmix_370m.py``,
which is the path the reported runs imported them from. That file has since
been removed; only the code these two trainers call was kept.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
except Exception:  # pragma: no cover
    StateDictOptions = None  # type: ignore
    get_model_state_dict = None  # type: ignore
    get_optimizer_state_dict = None  # type: ignore
    set_model_state_dict = None  # type: ignore
    set_optimizer_state_dict = None  # type: ignore

log = logging.getLogger("olmo_370m_core")

SEQ_LEN = 2048


TOKENIZER_ID = "allenai/dolma2-tokenizer"


EMBEDDING_SIZE = 100_352


GLOBAL_BATCH_TOKENS = 4_194_304


MICROBATCH_TOKENS = 65_536


PEAK_LR = 4.0e-4


DEFAULT_SEED = 42


DEFAULT_LENGTH_TOKENS = 10_000_058_051  # → 2384 steps at GBS 4_194_304


CONFIG_NAME = "OLMo-2-370M-scratch"


# Canonical published corpora (edullm-data). Never use s3://edullm-datasets/.
DATA_BUCKET = "edullm-data"


LEGACY_DATA_BUCKET = "edullm-datasets"


def _broadcast_rank0_success(ok: bool) -> bool:
    """Rank 0 supplies ``ok``; all ranks return the broadcast value."""
    if not is_distributed():
        return ok
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.broadcast(flag, src=0)
    return bool(int(flag.item()))


def _abort_all_ranks(message: str, *, ok: bool) -> None:
    """If rank-0 reports failure, every rank raises SystemExit after broadcast."""
    if not _broadcast_rank0_success(ok):
        raise SystemExit(message)


def stage_load_path(
    load_path: str,
    *,
    save_folder: Path,
    wandb_run: object | None,
) -> Path:
    """Resolve a local or W&B checkpoint bootstrap into job scratch."""
    _refuse_legacy_uri(load_path)
    if load_path.startswith("s3://"):
        raise SystemExit(
            "S3 checkpoint resume is prohibited: S3 is input-data/bootstrap staging "
            "only and must not store run checkpoints; use a local path or "
            "wandb-artifact://entity/project/name:version"
        )
    prefix = "wandb-artifact://"
    if not load_path.startswith(prefix):
        return Path(load_path)
    if wandb_run is None:
        raise SystemExit("W&B artifact resume requires an active online W&B run")
    artifact_ref = load_path[len(prefix) :].strip("/")
    if artifact_ref.count("/") < 2 or ":" not in artifact_ref.rsplit("/", 1)[-1]:
        raise SystemExit(
            "W&B checkpoint reference must be "
            "wandb-artifact://entity/project/name:version"
        )
    artifact = wandb_run.use_artifact(artifact_ref, type="model")
    dest = Path(save_folder) / "_wandb_resume"
    log.info("Resuming: download W&B artifact %s → %s", artifact_ref, dest)
    downloaded = Path(artifact.download(root=str(dest)))
    direct = downloaded / "state.pt"
    if direct.is_file():
        return downloaded
    matches = list(downloaded.rglob("state.pt"))
    if len(matches) != 1:
        raise SystemExit(
            f"W&B resume artifact {artifact_ref!r} must contain exactly one state.pt; "
            f"found {len(matches)}"
        )
    return matches[0].parent


def _refuse_legacy_uri(uri: str) -> None:
    if LEGACY_DATA_BUCKET in uri:
        raise SystemExit(
            f"refusing legacy training URI (use s3://{DATA_BUCKET}/ via edullm_data): {uri}"
        )


@dataclass
class _Bookkeeping:
    """Minimal Trainer duck-type for TrainModule.optim_step / record_metric.

    Captures CE loss from ``record_ce_loss`` so the training loop can log
    ``train/loss`` to W&B.
    """

    global_step: int
    max_steps: int
    global_batch_size: int
    max_tokens: Optional[int] = None
    global_train_tokens_seen: int = 0
    dp_process_group: Any = None
    device: torch.device = torch.device("cuda")
    last_ce_loss: Optional[float] = None
    ce_loss_window_sum: float = 0.0
    ce_loss_window_n: int = 0

    def record_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_ce_loss(self, value: Any, *args: Any, **kwargs: Any) -> None:
        try:
            if torch.is_tensor(value):
                v = float(value.detach().float().mean().item())
            else:
                v = float(value)
        except Exception:
            return
        self.last_ce_loss = v
        self.ce_loss_window_sum += v
        self.ce_loss_window_n += 1

    def pop_ce_loss_avg(self) -> Optional[float]:
        if self.ce_loss_window_n <= 0:
            return self.last_ce_loss
        avg = self.ce_loss_window_sum / float(self.ce_loss_window_n)
        self.ce_loss_window_sum = 0.0
        self.ce_loss_window_n = 0
        return avg


def resolve_attn_backend() -> AttentionBackendName:
    prefer = os.environ.get("OLMO_ATTN_BACKEND", "torch").strip().lower()
    if prefer in ("torch", "sdpa", "eager"):
        return AttentionBackendName.torch
    if prefer in ("flash_2", "flash", "flash2", "auto"):
        try:
            import flash_attn  # noqa: F401

            backend = AttentionBackendName.flash_2
            backend.get_class().assert_supported()
            log.info("attn_backend=flash_2")
            return backend
        except Exception as e:
            log.warning("flash_attn unavailable (%s); using torch", e)
            return AttentionBackendName.torch
    try:
        return AttentionBackendName(prefer)
    except Exception:
        log.warning("Unknown OLMO_ATTN_BACKEND=%s; using torch", prefer)
        return AttentionBackendName.torch


def build_olmo2_config(*, fused_ce: bool) -> TransformerConfig:
    vocab_size = TokenizerConfig.dolma2().padded_vocab_size()
    if vocab_size != EMBEDDING_SIZE:
        raise SystemExit(
            f"dolma2 padded vocab {vocab_size} != expected EMBEDDING_SIZE {EMBEDDING_SIZE}"
        )
    cfg = TransformerConfig.olmo2_370M(
        vocab_size=vocab_size,
        attn_backend=resolve_attn_backend(),
    )
    if fused_ce:
        try:
            cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
            log.info("lm_head.loss_implementation=fused_linear (liger)")
        except Exception as e:
            log.warning("Could not set fused_linear (%s); using default CE", e)
    return cfg


def patch_liger_fused_ce_compat() -> bool:
    try:
        import importlib

        import liger_kernel  # noqa: F401

        cel = importlib.import_module("olmo_core.nn.functional.cross_entropy_loss")
    except Exception as e:
        log.warning("fused CE compat patch skipped (import): %s", e)
        return False

    apply_fn = getattr(cel, "_fused_linear_cross_entropy_loss", None)
    if apply_fn is None:
        return False
    if getattr(cel, "_edullm_fused_ce_patched", False):
        return True

    @torch._dynamo.disable()  # type: ignore[misc]
    def _fused_linear_cross_entropy_loss_compat(
        _input,
        weight,
        labels,
        *,
        bias=None,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
        ce_weight=None,
        label_smoothing: float = 0.0,
        softcap=None,
        accum_dtype=None,
    ):
        lse_scale = z_loss_multiplier if compute_z_loss else 0.0
        out = apply_fn(
            _input,
            weight,
            labels,
            bias,
            ce_weight,
            ignore_index,
            lse_scale,
            label_smoothing,
            reduction,
            softcap,
            compute_z_loss,
            accum_dtype,
        )
        if not isinstance(out, tuple):
            raise RuntimeError(f"unexpected fused CE return type: {type(out)}")
        ce_loss = out[0]
        z_loss = out[1] if len(out) > 1 else None
        if compute_z_loss:
            return ce_loss, z_loss
        return ce_loss, None

    cel.fused_linear_cross_entropy_loss = _fused_linear_cross_entropy_loss_compat  # type: ignore[attr-defined]
    cel._edullm_fused_ce_patched = True  # type: ignore[attr-defined]
    try:
        import olmo_core.nn.lm_head as lm_head

        lm_head.fused_linear_cross_entropy_loss = _fused_linear_cross_entropy_loss_compat  # type: ignore[attr-defined]
    except Exception as e:
        log.warning("could not patch lm_head fused CE binding (%s)", e)
    log.info("patched olmo_core fused_linear_cross_entropy_loss for liger>=0.8")
    return True


def try_enable_fused_ce() -> bool:
    try:
        import liger_kernel  # noqa: F401
    except Exception:
        log.warning("liger-kernel not installed; CE uses default LM-head path")
        return False
    if not patch_liger_fused_ce_compat():
        log.warning("liger present but fused CE compat patch failed; leaving default CE")
        return False
    return True


def build_train_module(
    *,
    lr: float,
    lr_warmup_steps: int,
    alpha_f: float,
    compile_model: bool,
    rank_microbatch_tokens: int,
) -> TransformerTrainModule:
    fused = try_enable_fused_ce()
    model_cfg = build_olmo2_config(fused_ce=fused)
    try:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps, alpha_f=alpha_f)
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps)
        if hasattr(scheduler, "alpha_f"):
            scheduler.alpha_f = alpha_f

    tm_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_tokens,
        max_sequence_length=SEQ_LEN,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    model = model_cfg.build(init_device="cuda")
    train_module = tm_cfg.build(model)
    log.info(
        "Built TransformerTrainModule (HSDP bf16, SkipStepAdamW, compile=%s, fused_ce=%s, alpha_f=%s)",
        compile_model,
        fused,
        alpha_f,
    )
    return train_module


def _cpu_plain_tensor(t: Any) -> torch.Tensor:
    if torch.is_tensor(t) and type(t).__name__ == "Tensor":
        return t.detach().cpu()
    full = getattr(t, "full_tensor", None)
    if callable(full):
        try:
            return full().detach().cpu()
        except Exception:
            pass
    local = getattr(t, "to_local", None)
    if callable(local):
        try:
            return local().detach().cpu()
        except Exception:
            pass
    if torch.is_tensor(t):
        return t.detach().cpu()
    raise TypeError(f"cannot convert {type(t)} to CPU tensor")


def _plainify_state_tree(obj: Any) -> Any:
    if torch.is_tensor(obj) or type(obj).__name__ == "DTensor":
        return _cpu_plain_tensor(obj)
    if isinstance(obj, dict):
        return {k: _plainify_state_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_plainify_state_tree(v) for v in obj]
        return type(obj)(seq) if not isinstance(obj, list) else seq
    return obj


def gather_train_module_state_dict(train_module: TransformerTrainModule) -> dict[str, Any]:
    if get_model_state_dict is None or StateDictOptions is None:
        return _plainify_state_tree(train_module.state_dict_to_save())

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(train_module.model, options=opts)
    optim_sd: Any = None
    if get_optimizer_state_dict is not None:
        try:
            optim_sd = get_optimizer_state_dict(
                train_module.model, train_module.optim, options=opts
            )
        except Exception as e:
            log.warning("full optimizer state gather failed (%s); saving model only", e)
    return {
        "model": _plainify_state_tree(model_sd),
        "optim": _plainify_state_tree(optim_sd) if optim_sd is not None else None,
    }


def save_checkpoint(
    path: Path,
    step: int,
    train_module: TransformerTrainModule,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    """All ranks gather; rank 0 atomically writes the local permanent checkpoint."""
    train_module_sd = gather_train_module_state_dict(train_module)
    ok = True
    err = "permanent checkpoint save failed"
    if get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=True)
            state = {
                "step": step,
                "train_module": train_module_sd,
                "args": vars(args),
                "meta": meta,
                "architecture": "olmo_core.TransformerConfig.olmo2_370M",
                "config_name": CONFIG_NAME,
                "train_stack": "TransformerTrainModule/HSDP/SkipStepAdamW (curriculum)",
                "method": (
                    "plain_ce" if args.pacing == "control" else f"curriculum:{args.pacing}"
                ),
                "arm": args.arm_id,
                "run_id": args.name,
                "ephemeral": False,
                "checkpoint_format": "full_state_dict_v1",
            }
            tmp = path / "state.pt.tmp"
            torch.save(state, tmp)
            tmp.replace(path / "state.pt")
            (path / "step.txt").write_text(str(step) + "\n")
            log.info("Saved permanent full checkpoint → %s (step=%s)", path, step)
        except Exception as exc:  # noqa: BLE001 — fail closed via broadcast
            ok = False
            err = f"permanent checkpoint save failed: {exc}"
            log.error("%s", err)
    _abort_all_ranks(err, ok=ok)


def load_checkpoint(path: Path, train_module: TransformerTrainModule) -> int:
    ckpt = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    tm_sd = ckpt["train_module"]
    fmt = ckpt.get("checkpoint_format")
    if (
        fmt == "full_state_dict_v1"
        and isinstance(tm_sd, dict)
        and "model" in tm_sd
        and set_model_state_dict is not None
        and StateDictOptions is not None
    ):
        opts = StateDictOptions(full_state_dict=True, strict=True)
        set_model_state_dict(train_module.model, tm_sd["model"], options=opts)
        if tm_sd.get("optim") is not None and set_optimizer_state_dict is not None:
            try:
                set_optimizer_state_dict(
                    train_module.model,
                    train_module.optim,
                    tm_sd["optim"],
                    options=opts,
                )
            except Exception as e:
                log.warning("optimizer restore failed (%s); continuing with model weights", e)
        else:
            try:
                train_module.load_state_dict({"model": tm_sd["model"], "optim": tm_sd.get("optim")})
            except Exception:
                pass
    else:
        train_module.load_state_dict(tm_sd)
    _move_optim_state_to_param_device(train_module.optim)
    step = int(ckpt["step"])
    log.info("Resumed from %s at step=%s format=%s", path, step, fmt or "legacy_sharded")
    return step


def _move_optim_state_to_param_device(optim: torch.optim.Optimizer) -> None:
    from olmo_core.distributed.utils import get_local_tensor as _glt

    moved = 0
    for group in optim.param_groups:
        for p in group["params"]:
            state = optim.state.get(p)
            if not state:
                continue
            try:
                device = _glt(p).device
            except Exception:
                device = p.device
            for k, v in list(state.items()):
                if torch.is_tensor(v) and v.device != device:
                    state[k] = v.to(device=device)
                    moved += 1
    if moved and get_rank() == 0:
        log.info("Moved %d optimizer state tensor(s) onto param devices after resume", moved)


def _checkpoint_step(path: Path) -> int:
    return int(path.name.replace("step", "").split("-")[0])


def find_latest_checkpoint(save_folder: Path) -> Optional[Path]:
    if not save_folder.is_dir():
        return None
    cands = [
        p
        for p in save_folder.iterdir()
        if p.is_dir() and p.name.startswith("step") and (p / "state.pt").is_file()
    ]
    if not cands:
        return None
    return max(cands, key=_checkpoint_step)
