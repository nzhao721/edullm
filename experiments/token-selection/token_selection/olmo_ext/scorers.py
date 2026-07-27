"""Loss masks for full-token, REL, RHO-1 excess-loss, and middle-perplexity scoring.

Polarity matches OLMo-core ``label_mask``: ``True`` = token contributes to loss.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

MethodName = Literal["full", "rel_ema", "rho_excess", "middle_ppl"]


def _per_row_keep_counts(valid2d: Tensor, k: float) -> tuple[Tensor, Tensor]:
    """Return ``(n_valid, n_keep)`` per row for fraction ``k`` of valid positions."""
    n_valid = valid2d.sum(dim=1)
    n_keep = torch.clamp((n_valid.to(torch.float32) * k).round().long(), min=1)
    n_keep = torch.minimum(n_keep, n_valid)
    return n_valid, n_keep


def top_k_mask(scores: Tensor, k: float, *, valid: Optional[Tensor] = None) -> Tensor:
    """Keep the top ``k`` fraction of positions by ``scores`` (higher = more important).

    Selection is *per sequence*: for 2-D ``[B, T]`` scores, each row keeps its own
    top-``k`` fraction of *valid* positions independently, matching ssToken's per-sample
    top-rho selection (Eq. 10). A 1-D tensor is treated as a single sequence, so the
    old global behaviour is preserved for that case. Leading dims of an N-D tensor are
    treated as independent rows and selection runs along the last (sequence) dim.
    """
    if scores.numel() == 0:
        return scores.bool()

    k = float(min(max(k, 1e-8), 1.0))
    orig_shape = scores.shape
    scores2d = scores.reshape(1, -1) if scores.dim() == 1 else scores.reshape(-1, orig_shape[-1])
    rows, cols = scores2d.shape

    if valid is None:
        valid2d = torch.ones_like(scores2d, dtype=torch.bool)
    else:
        valid2d = valid.reshape(scores2d.shape).to(dtype=torch.bool)

    _, n_keep = _per_row_keep_counts(valid2d, k)

    masked = scores2d.masked_fill(~valid2d, float("-inf"))
    order = masked.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(cols, device=scores.device).expand(rows, cols)
    ranks.scatter_(1, order, ar)
    keep = (ranks < n_keep.unsqueeze(1)) & valid2d
    return keep.reshape(orig_shape)


def middle_k_mask(scores: Tensor, k: float, *, valid: Optional[Tensor] = None) -> Tensor:
    """Keep the middle ``k`` fraction of positions by ``scores`` (e.g. CE / log-PPL).

    Per sequence, drops the lowest and highest ``(1-k)/2`` of *valid* positions
    (easiest and hardest when ``scores`` is loss). Keep count matches :func:`top_k_mask`
    (``round(n_valid * k)``, at least one when any valid). A 1-D tensor is one sequence;
    leading dims of an N-D tensor are independent rows along the last dim.
    """
    if scores.numel() == 0:
        return scores.bool()

    k = float(min(max(k, 1e-8), 1.0))
    orig_shape = scores.shape
    scores2d = scores.reshape(1, -1) if scores.dim() == 1 else scores.reshape(-1, orig_shape[-1])
    rows, cols = scores2d.shape

    if valid is None:
        valid2d = torch.ones_like(scores2d, dtype=torch.bool)
    else:
        valid2d = valid.reshape(scores2d.shape).to(dtype=torch.bool)

    n_valid, n_keep = _per_row_keep_counts(valid2d, k)
    n_drop = n_valid - n_keep
    drop_low = n_drop // 2  # easiest (lowest scores)

    # Ascending rank among valid: 0 = easiest. Invalids sort to +inf so they land last.
    masked = scores2d.masked_fill(~valid2d, float("inf"))
    order = masked.argsort(dim=1, descending=False)
    ranks = torch.empty_like(order)
    ar = torch.arange(cols, device=scores.device).expand(rows, cols)
    ranks.scatter_(1, order, ar)
    lo = drop_low.unsqueeze(1)
    hi = (drop_low + n_keep).unsqueeze(1)
    keep = (ranks >= lo) & (ranks < hi) & valid2d
    return keep.reshape(orig_shape)


def normalize_rel_per_row(
    rel: Tensor,
    *,
    valid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Per-sequence min-max normalize REL into ``[0, 1]`` (ssToken Eq. 8).

    Monotonic within each row, so for *pure* REL top-k this does not change the
    selection; it exists to make REL commensurable when blending with other signals
    (e.g. an attention score) in future conditions. Invalid positions map to 0.
    """
    orig_shape = rel.shape
    r = rel.reshape(1, -1) if rel.dim() == 1 else rel.reshape(-1, orig_shape[-1])
    if valid is None:
        v = torch.ones_like(r, dtype=torch.bool)
    else:
        v = valid.reshape(r.shape).to(dtype=torch.bool)

    row_max = r.masked_fill(~v, float("-inf")).max(dim=1, keepdim=True).values
    row_min = r.masked_fill(~v, float("inf")).min(dim=1, keepdim=True).values
    denom = (row_max - row_min).clamp_min(eps)
    out = (r - row_min) / denom
    out = out.masked_fill(~v, 0.0)
    return out.reshape(orig_shape)


def warmup_mask(shape_ref: Tensor, *, valid: Optional[Tensor] = None) -> Tensor:
    """All valid positions contribute to loss (full baseline + selection warmup)."""
    if valid is not None:
        return valid.to(dtype=torch.bool)
    return torch.ones_like(shape_ref, dtype=torch.bool)


def full_mask(shape_ref: Tensor, *, valid: Optional[Tensor] = None) -> Tensor:
    """Full-token baseline: every valid position contributes to loss."""
    return warmup_mask(shape_ref, valid=valid)


def rel_ema_mask(
    current_loss: Tensor,
    history_loss: Tensor,
    k: float,
    *,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Keep tokens with highest ``REL = L_hist − L_curr``."""
    return top_k_mask(history_loss - current_loss, k, valid=valid)


def rho_excess_mask(
    current_loss: Tensor,
    reference_loss: Tensor,
    k: float,
    *,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Keep tokens with highest RHO-1 excess loss ``L_curr − L_ref``."""
    return top_k_mask(current_loss - reference_loss, k, valid=valid)


def middle_ppl_mask(
    current_loss: Tensor,
    k: float,
    *,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Keep the middle ``k`` by current-model CE (``L_curr`` ≈ log-perplexity)."""
    return middle_k_mask(current_loss, k, valid=valid)


def build_mask(
    *,
    method: MethodName = "rel_ema",
    k: float,
    current_loss: Optional[Tensor] = None,
    history_loss: Optional[Tensor] = None,
    reference_loss: Optional[Tensor] = None,
    shape_ref: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    warmup: bool = False,
) -> Tensor:
    """Build loss mask for ``full``, ``rel_ema``, ``rho_excess``, or ``middle_ppl``."""
    if method == "full" or warmup:
        ref = shape_ref if shape_ref is not None else current_loss
        if ref is None:
            raise ValueError("full/warmup mask requires shape_ref or current_loss")
        return full_mask(ref, valid=valid)

    if method == "rel_ema":
        if current_loss is None or history_loss is None:
            raise ValueError("REL mask requires current_loss and history_loss")
        return rel_ema_mask(current_loss, history_loss, k, valid=valid)

    if method == "rho_excess":
        if current_loss is None or reference_loss is None:
            raise ValueError("RHO excess mask requires current_loss and reference_loss")
        return rho_excess_mask(current_loss, reference_loss, k, valid=valid)

    if method == "middle_ppl":
        if current_loss is None:
            raise ValueError("middle_ppl mask requires current_loss")
        return middle_ppl_mask(current_loss, k, valid=valid)

    raise ValueError(
        f"Unknown method {method!r}; expected 'full', 'rel_ema', 'rho_excess', or 'middle_ppl'"
    )
