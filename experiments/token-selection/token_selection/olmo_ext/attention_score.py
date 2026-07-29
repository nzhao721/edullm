"""Last-layer attention-received scores (ssToken-adapted for causal LM).

FlashAttention does not expose the full attention matrix. Following ssToken, we
capture the last block's attention *input* during the train forward (cheap hook),
then recompute Q/K for that layer only and form causal attention weights to
obtain per-key received mass.

Score for token ``i`` (per head, then mean over heads)::

    score[i] = sum_{j >= i} A[j, i]

where ``A[j, i]`` is the attention weight from query ``j`` onto key ``i`` under
the causal mask. Column-sum of the causal matrix equals mass from the query at
``i`` and all later queries onto key ``i``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def attention_received_from_qk(
    q: Tensor,
    k: Tensor,
    *,
    scale: Optional[float] = None,
    query_chunk: int = 256,
) -> Tensor:
    """Causal attention-received scores from Q/K tensors.

    Args:
        q: ``[B, T, H, D]`` query projections (after RoPE / QK-norm).
        k: ``[B, T, H_kv, D]`` key projections. If ``H_kv < H`` (GQA), keys are
            repeated to match ``H``.
        scale: Softmax scale; defaults to ``D ** -0.5``.
        query_chunk: Chunk size along the query axis to bound ``O(T^2)`` memory.

    Returns:
        Float tensor ``[B, T]`` — mean over heads of ``sum_{j>=i} A[j, i]``.
    """
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(f"expected q/k shaped [B,T,H,D]; got {tuple(q.shape)} / {tuple(k.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k batch/seq/head_dim mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")

    bsz, seq_len, n_heads, head_dim = q.shape
    n_kv = k.shape[2]
    if n_kv != n_heads:
        if n_heads % n_kv != 0:
            raise ValueError(f"n_heads={n_heads} not divisible by n_kv_heads={n_kv}")
        # Repeat KV heads to match Q heads (GQA).
        k = k.repeat_interleave(n_heads // n_kv, dim=2)

    scale_f = float(head_dim**-0.5 if scale is None else scale)
    # Accumulate in float32 for stable softmax column sums.
    received = torch.zeros(bsz, n_heads, seq_len, device=q.device, dtype=torch.float32)
    q32 = q.to(dtype=torch.float32)
    k32 = k.to(dtype=torch.float32)
    chunk = max(1, int(query_chunk))

    for j0 in range(0, seq_len, chunk):
        j1 = min(seq_len, j0 + chunk)
        q_chunk = q32[:, j0:j1]  # [B, C, H, D]
        # logits[b,h,c,t] = q[b,j0+c,h] · k[b,t,h] * scale
        logits = torch.einsum("bchd,bthd->bhct", q_chunk, k32) * scale_f
        # Causal: query at absolute index j cannot attend to keys > j.
        abs_j = torch.arange(j0, j1, device=q.device)
        key_idx = torch.arange(seq_len, device=q.device)
        # mask[c, t] True where key > query → fill -inf
        invalid = key_idx.unsqueeze(0) > abs_j.unsqueeze(1)  # [C, T]
        logits = logits.masked_fill(invalid.view(1, 1, j1 - j0, seq_len), float("-inf"))
        attn = torch.softmax(logits, dim=-1)  # over keys
        received += attn.sum(dim=-2)  # sum over this query chunk → [B, H, T]

    return received.mean(dim=1)  # [B, T]


def unwrap_transformer(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP (``.module``) and ``torch.compile`` (``_orig_mod``) wrappers.

    Attention hooks must register on the real Transformer that owns ``.blocks``.
    """
    seen: set[int] = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
            model = model.module  # type: ignore[assignment]
            continue
        if hasattr(model, "_orig_mod") and isinstance(getattr(model, "_orig_mod"), nn.Module):
            model = model._orig_mod  # type: ignore[assignment]
            continue
        break
    return model


def _last_transformer_block(model: nn.Module) -> nn.Module:
    model = unwrap_transformer(model)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError(
            "attention_topk requires a Transformer with `.blocks`; "
            f"got {type(model).__name__} after unwrap"
        )
    values: Sequence[nn.Module]
    if isinstance(blocks, nn.ModuleDict):
        # Keys are "0".."N-1"; take highest numeric key.
        keys = sorted(blocks.keys(), key=lambda s: int(s))
        values = [blocks[k] for k in keys]
    elif isinstance(blocks, (nn.ModuleList, list, tuple)):
        values = list(blocks)
    elif hasattr(blocks, "values"):
        values = list(blocks.values())
    else:
        raise RuntimeError(f"unsupported blocks container: {type(blocks).__name__}")
    if not values:
        raise RuntimeError("model.blocks is empty")
    return values[-1]


def _attention_module(block: nn.Module) -> nn.Module:
    attn = getattr(block, "attention", None)
    if attn is None:
        raise RuntimeError(
            f"last block {type(block).__name__} has no `.attention` for recompute"
        )
    return attn


def _project_qk(
    attn: nn.Module,
    x: Tensor,
    *,
    rope_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    """Mirror OLMo-core ``Attention`` Q/K path up to (but not including) SDPA."""
    w_q = getattr(attn, "w_q", None)
    w_k = getattr(attn, "w_k", None)
    if w_q is None or w_k is None:
        raise RuntimeError(
            f"attention module {type(attn).__name__} lacks w_q/w_k; "
            "fused-QKV attention is not supported for attention_topk recompute"
        )

    q = w_q(x)
    k = w_k(x)
    clip = getattr(attn, "clip_qkv", None)
    if clip is not None:
        q = q.clamp(min=-clip, max=clip)
        k = k.clamp(min=-clip, max=clip)

    head_dim = int(getattr(attn, "head_dim"))
    use_head_qk_norm = bool(getattr(attn, "use_head_qk_norm", False))
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)

    if not use_head_qk_norm:
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)

    bsz, seq_len, _ = x.shape
    q = q.view(bsz, seq_len, -1, head_dim)
    k = k.view(bsz, seq_len, -1, head_dim)

    if use_head_qk_norm:
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)

    rope = getattr(attn, "rope", None)
    if rope is not None:
        apply = getattr(attn, "_apply_rope", None)
        if apply is None:
            raise RuntimeError("Attention has rope but no _apply_rope helper")
        kw = dict(rope_kwargs or {})
        # Signature: (q, k, start_pos, pos_sin, pos_cos, freqs_cis, cu_doc_lens)
        q, k = apply(
            q,
            k,
            kw.get("start_pos"),
            kw.get("pos_sin"),
            kw.get("pos_cos"),
            kw.get("freqs_cis"),
            kw.get("cu_doc_lens"),
        )
    return q, k


@dataclass
class _AttnCapture:
    x: Optional[Tensor] = None
    rope_kwargs: Optional[Dict[str, Any]] = None
    attn_module: Optional[nn.Module] = None


@contextlib.contextmanager
def capture_last_layer_attention_input(model: nn.Module) -> Iterator[_AttnCapture]:
    """Hook the last block's ``attention`` forward to store its input activations."""
    block = _last_transformer_block(model)
    attn = _attention_module(block)
    state = _AttnCapture(attn_module=attn)

    def _hook(_module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        if not args:
            raise RuntimeError("attention forward received no positional input")
        # Detach: scores must not feed the train graph (selection is stop-grad).
        state.x = args[0].detach()
        # Keep only RoPE / doc-layout kwargs needed for Q/K recompute.
        keep = (
            "pos_sin",
            "pos_cos",
            "freqs_cis",
            "cu_doc_lens",
            "cu_doc_lens_q",
            "cu_doc_lens_k",
            "max_doc_len",
            "max_doc_len_q",
            "max_doc_len_k",
            "local_k_slice",
            "cache_leftpad",
        )
        state.rope_kwargs = {k: kwargs[k] for k in keep if k in kwargs and kwargs[k] is not None}

    handle = attn.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        yield state
    finally:
        handle.remove()


@torch.no_grad()
def scores_from_capture(
    capture: _AttnCapture,
    *,
    query_chunk: int = 256,
) -> Tensor:
    """Recompute last-layer attention-received scores from a filled capture."""
    if capture.x is None or capture.attn_module is None:
        raise RuntimeError(
            "last-layer attention input was not captured; the train forward may "
            "have skipped the final block"
        )
    q, k = _project_qk(capture.attn_module, capture.x, rope_kwargs=capture.rope_kwargs)
    scale = getattr(capture.attn_module, "softmax_scale", None)
    return attention_received_from_qk(q, k, scale=scale, query_chunk=query_chunk)


@torch.no_grad()
def last_layer_attention_received(
    model: nn.Module,
    input_ids: Tensor,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    query_chunk: int = 256,
    forward_fn=None,
) -> Tuple[Tensor, Tensor]:
    """Run a forward under capture and return ``(logits_or_unused, attn_scores)``.

    Prefer :func:`capture_last_layer_attention_input` around the *training* forward
    so scoring reuses that pass. This helper is for smoke / offline use.
    """
    with capture_last_layer_attention_input(model) as cap:
        if forward_fn is not None:
            out = forward_fn(input_ids, **(model_kwargs or {}))
        else:
            out = model(input_ids, **(model_kwargs or {}))
    scores = scores_from_capture(cap, query_chunk=query_chunk)
    return out, scores
