"""Tests for attention-received scoring and attention_topk masks."""

from __future__ import annotations

import torch
import torch.nn as nn

from token_selection.olmo_ext.attention_score import (
    attention_received_from_qk,
    unwrap_transformer,
)
from token_selection.olmo_ext.scorers import attention_topk_mask, build_mask, top_k_mask


def test_attention_received_causal_column_sum():
    # One head, identical Q/K → uniform softmax over the causal prefix.
    # received[i] = sum_{j>=i} 1/(j+1); monotonically non-increasing in i.
    bsz, seq, heads, dim = 1, 4, 1, 8
    q = torch.ones(bsz, seq, heads, dim)
    k = torch.ones(bsz, seq, heads, dim)
    scores = attention_received_from_qk(q, k, query_chunk=2)
    assert scores.shape == (1, 4)
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2] >= scores[0, 3]
    expected = torch.tensor(
        [
            1.0 + 0.5 + 1.0 / 3.0 + 0.25,
            0.5 + 1.0 / 3.0 + 0.25,
            1.0 / 3.0 + 0.25,
            0.25,
        ]
    )
    assert torch.allclose(scores[0], expected, atol=1e-5)


def test_attention_received_gqa_repeat():
    bsz, seq, n_heads, n_kv, dim = 2, 8, 4, 2, 4
    q = torch.randn(bsz, seq, n_heads, dim)
    k = torch.randn(bsz, seq, n_kv, dim)
    scores = attention_received_from_qk(q, k, query_chunk=3)
    assert scores.shape == (bsz, seq)
    assert torch.isfinite(scores).all()


def test_attention_topk_mask_keeps_high_received():
    scores = torch.tensor([[0.1, 0.2, 0.9, 0.8, 0.3]])
    mask = attention_topk_mask(scores, 0.6)
    # 5 tokens, k=0.6 -> keep 3 highest: indices 2,3, and next (1 or 4).
    assert int(mask.sum()) == 3
    assert mask[0, 2] and mask[0, 3]
    assert torch.equal(mask, top_k_mask(scores, 0.6))


def test_build_mask_attention_topk_and_warmup():
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = build_mask(method="attention_topk", k=0.5, attention_score=scores)
    assert mask.tolist() == [False, False, True, True]
    warm = build_mask(
        method="attention_topk",
        k=0.5,
        attention_score=scores,
        shape_ref=scores,
        warmup=True,
    )
    assert warm.all()


def test_attention_topk_requires_score():
    try:
        build_mask(method="attention_topk", k=0.6)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "attention_score" in str(e)


def test_unwrap_transformer_ddp_and_compile():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4)])

    inner = Tiny()

    class FakeDDP(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

    class FakeCompiled(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self._orig_mod = module

    wrapped = FakeCompiled(FakeDDP(inner))
    assert unwrap_transformer(wrapped) is inner
    assert unwrap_transformer(inner) is inner
