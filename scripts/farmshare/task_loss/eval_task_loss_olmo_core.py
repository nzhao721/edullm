#!/usr/bin/env python3
"""OLMo-ladder task-loss (RC 5-shot bpb) for olmo-core OLMo2-370M checkpoints."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

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
from olmo.util import add_cached_path_clients, prepare_cli_environment

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.distributed.checkpoint import unshard_checkpoint

log = logging.getLogger("eval_task_loss_olmo_core")

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


def evaluate_label(
    model: torch.nn.Module,
    cfg: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    label: str,
    device_eval_batch_size: int,
) -> tuple[float, int]:
    evaluator = build_evaluator(
        cfg,
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            device_eval_batch_size=device_eval_batch_size,
            subset_num_batches=None,
        ),
        tokenizer,
        device,
    )
    evaluator.reset_metrics()
    n_batches = 0
    for batch in evaluator.eval_loader:
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        logits = model_logits(model, batch["input_ids"])
        evaluator.eval_metric.update(batch, logits)
        n_batches += 1
    metrics = evaluator.compute_metrics()
    (value,) = list(metrics.values())
    return float(value), n_batches


def build_train_config(base_config: Path | None) -> TrainConfig:
    candidates = []
    if base_config is not None:
        candidates.append(base_config)
    env_cfg = os.environ.get("LADDER_BASE_CONFIG", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg))
    for c in candidates:
        if c and c.is_file():
            cfg = TrainConfig.load(str(c), [])
            cfg.evaluators = []
            return cfg
    raise SystemExit(
        "Need an ai2-olmo config.yaml for tokenizer/eval loaders. "
        "Pass --base-config or set LADDER_BASE_CONFIG."
    )


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

    fmt = args.format
    if fmt == "auto":
        if (args.checkpoint / "model_eval.pt").is_file() or (args.checkpoint / "state.pt").is_file():
            fmt = "state_pt"
        elif (args.checkpoint / "model_and_optim" / ".metadata").is_file():
            fmt = "distcp"
        else:
            raise SystemExit(f"Cannot detect checkpoint format under {args.checkpoint}")

    # olmo-core unshard_checkpoint cannot run after process-group init.
    if fmt == "distcp" and not (args.checkpoint / "model_eval.pt").is_file():
        log.info("Materializing distcp → model_eval.pt (before dist init)")
        materialize_distcp_model_eval(args.checkpoint)
        fmt = "state_pt"
    elif fmt == "distcp":
        fmt = "state_pt"

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    prepare_cli_environment()
    add_cached_path_clients()
    device = torch.device("cuda")

    log.info("Building olmo2_370M on %s", device)
    model = build_model()
    step = load_state_pt(args.checkpoint, model)
    model.to(device)
    log.info("Loaded checkpoint step=%s from %s", step, args.checkpoint)

    cfg = build_train_config(args.base_config)
    cfg.device_eval_batch_size = args.device_eval_batch_size
    tokenizer = Tokenizer.from_train_config(cfg)

    results: dict[str, float] = {}
    for label in TASK_LABELS:
        bpb, n_batches = evaluate_label(
            model, cfg, tokenizer, device, label, args.device_eval_batch_size
        )
        results[label] = bpb
        log.info("%s %s: task_loss_bpb=%.6f (%d batches)", args.run_name, label, bpb, n_batches)
        print(f"{label}\t{bpb:.6f}", flush=True)

    mmlu_keys = [k for k in results if k.startswith("mmlu_") and k.endswith("_rc_5shot_bpb")]
    if mmlu_keys:
        results["mmlu_avg_rc_5shot_bpb"] = sum(results[k] for k in mmlu_keys) / len(mmlu_keys)
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
    present = [k for k in core_keys if k in results]
    if present:
        results["core_avg_rc_5shot_bpb"] = sum(results[k] for k in present) / len(present)
    if results:
        results["macro_mean_task_loss_bpb"] = sum(results[k] for k in TASK_LABELS if k in results) / max(
            1, sum(1 for k in TASK_LABELS if k in results)
        )

    payload = {
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "format": fmt,
        "step": step,
        "task_loss_bpb": results,
    }
    if dist.get_rank() == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

