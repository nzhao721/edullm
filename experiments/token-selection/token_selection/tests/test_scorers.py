"""Tests for full + REL scorers / masks."""

from __future__ import annotations

import torch

from token_selection.olmo_ext.scorers import (
    build_mask,
    full_mask,
    middle_k_mask,
    normalize_rel_per_row,
    top_k_mask,
    warmup_mask,
)


def test_top_k_keep_ratio():
    scores = torch.arange(10, dtype=torch.float32)
    mask = top_k_mask(scores, 0.6)
    assert int(mask.sum()) == 6
    assert mask[-6:].all()
    assert not mask[:4].any()


def test_top_k_respects_valid():
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0])
    valid = torch.tensor([False, True, True, True])
    mask = top_k_mask(scores, 2 / 3, valid=valid)
    assert not mask[0]
    assert int(mask.sum()) == 2


def test_top_k_per_sequence_independent():
    # Row 0 has globally-low scores, row 1 globally-high. Per-sequence selection must
    # keep each row's own top-k rather than letting row 1 starve row 0 (the old global
    # behaviour would keep all of row 1 and none of row 0).
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])
    mask = top_k_mask(scores, 0.5)
    assert mask.tolist() == [[False, False, True, True], [False, False, True, True]]
    assert int(mask[0].sum()) == 2  # low-scoring row still keeps its share
    assert int(mask[1].sum()) == 2


def test_top_k_flat_easy_row_keeps_share():
    # A uniformly "easy" (all-equal) sequence must still contribute ~k of its tokens,
    # not zero.
    scores = torch.tensor([[5.0, 5.0, 5.0, 5.0], [0.1, 0.2, 0.3, 0.4]])
    mask = top_k_mask(scores, 0.5)
    assert int(mask[0].sum()) == 2
    assert int(mask[1].sum()) == 2


def test_top_k_per_row_valid_counts():
    scores = torch.tensor([[9.0, 8.0, 7.0, 6.0], [1.0, 2.0, 3.0, 4.0]])
    valid = torch.tensor([[False, True, True, True], [True, True, True, True]])
    mask = top_k_mask(scores, 0.5, valid=valid)
    # Row 0: 3 valid -> round(1.5)=2 kept; position 0 never kept.
    assert not mask[0, 0]
    assert int(mask[0].sum()) == 2
    # Row 1: 4 valid -> 2 kept (the two highest: indices 2,3).
    assert mask[1].tolist() == [False, False, True, True]


def test_top_k_row_all_invalid_or_single():
    scores = torch.tensor([[3.0, 1.0, 2.0], [5.0, 4.0, 6.0]])
    valid = torch.tensor([[False, False, False], [False, True, False]])
    mask = top_k_mask(scores, 0.6, valid=valid)
    assert int(mask[0].sum()) == 0  # empty row keeps nothing
    assert mask[1].tolist() == [False, True, False]  # single valid -> keep it


def test_rel_mask_is_per_sequence():
    curr = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    hist = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    # REL = hist - curr -> row0: [3,1,-1,-3] keep top2 = idx0,1; row1: [-3,-1,1,3] keep idx2,3.
    mask = build_mask(method="rel_ema", k=0.5, current_loss=curr, history_loss=hist)
    assert mask.tolist() == [[True, True, False, False], [False, False, True, True]]


def test_normalize_rel_per_row_monotonic_no_op_for_selection():
    rel = torch.tensor([[3.0, 1.0, -1.0, -3.0], [-3.0, -1.0, 1.0, 3.0]])
    norm = normalize_rel_per_row(rel)
    assert torch.allclose(norm.max(dim=1).values, torch.ones(2))
    assert torch.allclose(norm.min(dim=1).values, torch.zeros(2))
    # Per-row top-k identical whether or not we normalize first (monotone within a row).
    assert torch.equal(top_k_mask(rel, 0.5), top_k_mask(norm, 0.5))


def test_normalize_rel_respects_valid():
    rel = torch.tensor([[10.0, 2.0, 4.0, 8.0]])
    valid = torch.tensor([[False, True, True, True]])
    norm = normalize_rel_per_row(rel, valid=valid)
    assert norm[0, 0] == 0.0  # invalid position zeroed
    # min over valid (2.0) -> 0, max over valid (8.0) -> 1.
    assert torch.isclose(norm[0, 1], torch.tensor(0.0))
    assert torch.isclose(norm[0, 3], torch.tensor(1.0))


def test_full_mask_keeps_all_valid():
    x = torch.zeros(4)
    assert full_mask(x).all()
    valid = torch.tensor([False, True, True, True])
    m = build_mask(method="full", k=0.5, shape_ref=x, valid=valid)
    assert m.tolist() == [False, True, True, True]


def test_warmup_mask():
    x = torch.zeros(100)
    assert warmup_mask(x).all()


def test_rel_mask():
    curr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    hist = torch.tensor([4.0, 3.0, 2.0, 1.0])
    rel = build_mask(method="rel_ema", k=0.5, current_loss=curr, history_loss=hist)
    assert rel.tolist() == [True, True, False, False]


def test_rel_warmup_falls_back_to_full():
    curr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    hist = torch.zeros(4)
    m = build_mask(method="rel_ema", k=0.5, current_loss=curr, history_loss=hist, warmup=True)
    assert m.all()


def test_rho_excess_mask_polarity():
    # High current loss and low ref loss => high excess => keep.
    curr = torch.tensor([4.0, 3.0, 2.0, 1.0])
    ref = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mask = build_mask(method="rho_excess", k=0.5, current_loss=curr, reference_loss=ref)
    assert mask.tolist() == [True, True, False, False]


def test_rho_warmup_falls_back_to_full():
    curr = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ref = torch.zeros(4)
    m = build_mask(
        method="rho_excess", k=0.5, current_loss=curr, reference_loss=ref, warmup=True
    )
    assert m.all()


def test_middle_k_drops_easiest_and_hardest():
    # Ascending CE: drop lowest 2 and highest 2 when k=0.6 on 10 tokens -> keep 6.
    scores = torch.arange(10, dtype=torch.float32)
    mask = middle_k_mask(scores, 0.6)
    assert int(mask.sum()) == 6
    assert mask.tolist() == [False, False, True, True, True, True, True, True, False, False]


def test_middle_k_per_sequence_independent():
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]])
    mask = middle_k_mask(scores, 0.5)
    # Each row: 4 valid -> keep 2 middle. Row0 ascending ranks 1..2; row1 scores desc
    # so middle by value is indices with ranks 1..2 after ascending sort = 20,30.
    assert mask[0].tolist() == [False, True, True, False]
    assert mask[1].tolist() == [False, True, True, False]


def test_middle_k_respects_valid():
    scores = torch.tensor([10.0, 1.0, 2.0, 3.0, 4.0])
    valid = torch.tensor([False, True, True, True, True])
    mask = middle_k_mask(scores, 0.5, valid=valid)
    assert not mask[0]
    # 4 valid -> keep 2 middle by ascending CE among valid: values 2,3
    assert mask.tolist() == [False, False, True, True, False]


def test_middle_ppl_mask_and_warmup():
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = build_mask(method="middle_ppl", k=0.6, reference_loss=ref)
    # 5 tokens, k=0.6 -> keep 3 middle: drop easiest + hardest
    assert mask.tolist() == [False, True, True, True, False]
    warm = build_mask(
        method="middle_ppl", k=0.6, reference_loss=ref, warmup=True
    )
    assert warm.all()


def test_learnability_mask_polarity():
    """Larger L_early − L_late = larger improvement → kept (top-k)."""
    # Token 0 improved a lot (early hard, late easy); token 3 got worse.
    early = torch.tensor([4.0, 3.0, 2.0, 1.0])
    late = torch.tensor([1.0, 1.5, 1.5, 3.0])
    # scores = [3.0, 1.5, 0.5, -2.0] → top 50% keeps indices 0,1
    mask = build_mask(method="learnability", k=0.5, early_loss=early, late_loss=late)
    assert mask.tolist() == [True, True, False, False]


def test_learnability_warmup_falls_back_to_full():
    early = torch.tensor([4.0, 3.0, 2.0, 1.0])
    late = torch.zeros(4)
    m = build_mask(
        method="learnability",
        k=0.5,
        early_loss=early,
        late_loss=late,
        shape_ref=early,
        warmup=True,
    )
    assert m.all()


def test_random_mask_keep_ratio_and_seed():
    shape = torch.zeros(2, 8)
    valid = torch.ones_like(shape, dtype=torch.bool)
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    mask_a = build_mask(method="random", k=0.6, shape_ref=shape, valid=valid, generator=gen_a)
    mask_b = build_mask(method="random", k=0.6, shape_ref=shape, valid=valid, generator=gen_b)
    assert torch.equal(mask_a, mask_b)
    assert int(mask_a[0].sum()) == 5  # round(8 * 0.6)
    assert int(mask_a[1].sum()) == 5


def test_random_mask_per_sequence():
    shape = torch.zeros(2, 4)
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    gen = torch.Generator().manual_seed(0)
    mask = build_mask(method="random", k=0.5, shape_ref=shape, valid=valid, generator=gen)
    assert not mask[0, 3]
    assert int(mask[0].sum()) == 2
    assert int(mask[1].sum()) == 2


def test_learnability_requires_both_losses():
    import pytest

    with pytest.raises(ValueError, match="early_loss"):
        build_mask(method="learnability", k=0.5, late_loss=torch.zeros(2))
