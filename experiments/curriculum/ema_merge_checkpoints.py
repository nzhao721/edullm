#!/usr/bin/env python3
"""Post-hoc EMA merge of late curriculum checkpoints.

Merges model weights from steps ``2000, 2125, 2250, 2384`` with the convention::

    avg ← α · avg + (1 − α) · newest

Default ``α=0.8`` (so each update is ``0.8·avg + 0.2·newest``). Checkpoints are
processed oldest → newest. Writes ``step2384-ema/`` under the checkpoints root
and by default launches the shared 20-label task_loss eval on the merged
artifact (``--task-loss`` / ``--no-task-loss``).

Does **not** use online ``token_selection.olmo_ext.ema.EMAHistory`` (REL history).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch

log = logging.getLogger("ema_merge_checkpoints")

DEFAULT_EMA_STEPS: tuple[int, ...] = (2000, 2125, 2250, 2384)
DEFAULT_ALPHA = 0.8


def _is_tensor(obj: Any) -> bool:
    return torch.is_tensor(obj) or type(obj).__name__ in ("Tensor", "DTensor")


def ema_merge_state_dicts(
    state_dicts: Sequence[Mapping[str, Any]],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Recursively EMA-merge a sequence of nested state dicts (oldest first).

    Non-tensor leaves keep the **newest** value. Tensor leaves follow::

        avg ← alpha · avg + (1 - alpha) · newest
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    a = float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {a}")
    avg: Any = _clone_tree(state_dicts[0])
    for sd in state_dicts[1:]:
        avg = _ema_blend(avg, sd, alpha=a)
    return avg


def _clone_tree(obj: Any) -> Any:
    if _is_tensor(obj):
        t = obj.detach() if torch.is_tensor(obj) else obj
        return t.detach().cpu().clone() if torch.is_tensor(t) else t
    if isinstance(obj, Mapping):
        return {k: _clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tree(v) for v in obj)
    return obj


def _ema_blend(avg: Any, newest: Any, *, alpha: float) -> Any:
    if _is_tensor(avg) and _is_tensor(newest):
        a = avg.detach().float().cpu()
        b = newest.detach().float().cpu()
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError(f"tensor shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        out = alpha * a + (1.0 - alpha) * b
        return out.to(dtype=avg.dtype if torch.is_tensor(avg) else out.dtype)
    if isinstance(avg, Mapping) and isinstance(newest, Mapping):
        keys = set(avg) | set(newest)
        merged: Dict[str, Any] = {}
        for k in keys:
            if k not in avg:
                merged[k] = _clone_tree(newest[k])
            elif k not in newest:
                merged[k] = _clone_tree(avg[k])
            else:
                merged[k] = _ema_blend(avg[k], newest[k], alpha=alpha)
        return merged
    if isinstance(avg, list) and isinstance(newest, list) and len(avg) == len(newest):
        return [_ema_blend(x, y, alpha=alpha) for x, y in zip(avg, newest)]
    # Prefer newest for scalars / mismatched structures.
    return _clone_tree(newest)


def ema_weights_closed_form(
    n: int,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> List[float]:
    """Closed-form mixture weights for ``n`` checkpoints (oldest index 0).

    After recursive ``avg ← α·avg + (1-α)·newest`` over c0..c_{n-1}:

        w_i = α^{n-1-i} · (1-α)   for i > 0
        w_0 = α^{n-1}
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    a = float(alpha)
    one_m = 1.0 - a
    weights = [a ** (n - 1 - i) * (one_m if i > 0 else 1.0) for i in range(n)]
    # Numerical drift: normalize lightly when alpha in (0,1).
    s = sum(weights)
    if s <= 0:
        raise RuntimeError("degenerate EMA weights")
    return [w / s for w in weights]


def load_checkpoint_model(path: Path) -> tuple[Dict[str, Any], dict]:
    """Load ``state.pt`` and return ``(model_state_dict, full_ckpt)``."""
    state_path = path / "state.pt" if path.is_dir() else path
    if not state_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {state_path}")
    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    tm = ckpt.get("train_module")
    if isinstance(tm, dict) and "model" in tm:
        model_sd = tm["model"]
    elif isinstance(tm, dict):
        model_sd = tm
    else:
        raise ValueError(f"unrecognized train_module layout in {state_path}")
    if not isinstance(model_sd, dict):
        raise ValueError(f"model state_dict is not a dict in {state_path}")
    return model_sd, ckpt


def write_ema_checkpoint(
    out_dir: Path,
    *,
    model_sd: Mapping[str, Any],
    template_ckpt: Mapping[str, Any],
    steps: Sequence[int],
    alpha: float,
    arm_id: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tm_template = template_ckpt.get("train_module")
    if isinstance(tm_template, dict) and "model" in tm_template:
        train_module = {
            "model": dict(model_sd),
            "optim": None,  # EMA artifact is eval-only
        }
    else:
        train_module = dict(model_sd)

    state = {
        "step": int(steps[-1]) if steps else 0,
        "train_module": train_module,
        "args": template_ckpt.get("args"),
        "meta": {
            **(template_ckpt.get("meta") or {}),
            "ema_merge": {
                "steps": list(steps),
                "alpha": float(alpha),
                "convention": "avg <- alpha*avg + (1-alpha)*newest",
                "arm_id": arm_id,
            },
        },
        "architecture": template_ckpt.get("architecture"),
        "config_name": template_ckpt.get("config_name"),
        "train_stack": template_ckpt.get("train_stack"),
        "method": "posthoc_ema_merge",
        "arm": arm_id,
        "run_id": template_ckpt.get("run_id"),
        "checkpoint_format": template_ckpt.get("checkpoint_format", "full_state_dict_v1"),
    }
    tmp = out_dir / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(out_dir / "state.pt")
    (out_dir / "step.txt").write_text(str(state["step"]) + "\n", encoding="utf-8")
    (out_dir / "ema_manifest.json").write_text(
        json.dumps(state["meta"]["ema_merge"], indent=2) + "\n", encoding="utf-8"
    )
    return out_dir / "state.pt"


def resolve_step_dirs(checkpoints_root: Path, steps: Sequence[int]) -> List[Path]:
    dirs: List[Path] = []
    for s in steps:
        cand = checkpoints_root / f"step{int(s)}"
        if not (cand / "state.pt").is_file():
            raise FileNotFoundError(f"missing checkpoint for step {s}: {cand / 'state.pt'}")
        dirs.append(cand)
    return dirs


def maybe_task_loss(
    ckpt_dir: Path,
    *,
    results_dir: Path,
    eval_script: Optional[str],
    enabled: bool,
    run_name: str,
) -> None:
    if not enabled:
        return
    # Import shared hook from token-selection package.
    ts_root = Path(__file__).resolve().parents[1] / "token-selection"
    if str(ts_root) not in sys.path:
        sys.path.insert(0, str(ts_root))
    from token_selection.olmo_ext.task_loss_hook import trigger_task_loss_eval

    results_dir.mkdir(parents=True, exist_ok=True)
    trigger_task_loss_eval(
        ckpt_dir,
        run_name=run_name,
        out_path=results_dir / "step2384-ema_task_loss.json",
        eval_script=eval_script,
        enabled=True,
        async_=False,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoints-root",
        type=Path,
        required=True,
        help="Directory containing step{N}/state.pt "
        "(stage from s3://edullm-checkpoints/curriculum/<arm>/checkpoints into a work dir first)",
    )
    ap.add_argument("--arm-id", type=str, required=True)
    ap.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_EMA_STEPS),
        help=f"Steps to merge oldest→newest (default: {list(DEFAULT_EMA_STEPS)})",
    )
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <checkpoints-root>/step2384-ema",
    )
    ap.add_argument(
        "--task-loss-results-dir",
        type=Path,
        default=None,
        help="If set (and --task-loss), write EMA task_loss JSON here",
    )
    ap.add_argument(
        "--task-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run task_loss eval on the merged artifact (default: on; use --no-task-loss to skip)",
    )
    ap.add_argument("--task-loss-eval-script", type=str, default=None)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    steps = [int(s) for s in args.steps]
    dirs = resolve_step_dirs(args.checkpoints_root, steps)
    model_sds: List[Dict[str, Any]] = []
    template: Optional[dict] = None
    for d in dirs:
        sd, ckpt = load_checkpoint_model(d)
        model_sds.append(sd)
        template = ckpt
        log.info("loaded %s", d)
    assert template is not None
    merged = ema_merge_state_dicts(model_sds, alpha=float(args.alpha))
    out_dir = args.out_dir or (args.checkpoints_root / "step2384-ema")
    write_ema_checkpoint(
        out_dir,
        model_sd=merged,
        template_ckpt=template,
        steps=steps,
        alpha=float(args.alpha),
        arm_id=str(args.arm_id),
    )
    log.info("wrote EMA checkpoint → %s (alpha=%.3f steps=%s)", out_dir, args.alpha, steps)
    if args.task_loss:
        results = args.task_loss_results_dir or (args.checkpoints_root.parent / "task_loss_results")
        maybe_task_loss(
            out_dir,
            results_dir=Path(results),
            eval_script=args.task_loss_eval_script,
            enabled=True,
            run_name=f"{args.arm_id}-step2384-ema",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
