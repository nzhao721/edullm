"""Tests for EMAHistory."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from token_selection.olmo_ext.ema import EMAHistory, alpha_at_step


def test_alpha_warmup_then_decay():
    assert alpha_at_step(0, t0=10, total_steps=110, alpha_start=0.9999, alpha_end=0.999) == 0.9999
    assert alpha_at_step(9, t0=10, total_steps=110, alpha_start=0.9999, alpha_end=0.999) == 0.9999
    mid = alpha_at_step(60, t0=10, total_steps=110, alpha_start=0.9999, alpha_end=0.999)
    assert 0.999 < mid < 0.9999
    end = alpha_at_step(110, t0=10, total_steps=110, alpha_start=0.9999, alpha_end=0.999)
    assert abs(end - 0.999) < 1e-12


class M(nn.Module):
    def __init__(self, value=(1.0, 1.0)):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(list(value)))


def test_history_is_a_convex_combination_of_observed_weights_only():
    """The initialization must not leak into the history, at any step."""
    m = M((99.0, 99.0))  # an initialization far from every weight the model will observe
    ema = EMAHistory.from_module(m, alpha=0.5)
    assert not ema.has_history

    m.w.data.fill_(2.0)
    ema.update_module(m, alpha=0.5)
    # One observation, so the debiased history is exactly that observation.
    assert torch.allclose(ema.history("w"), torch.tensor([2.0, 2.0]))

    m.w.data.fill_(4.0)
    ema.update_module(m, alpha=0.5)
    # Weights 0.25 on theta_1 and 0.5 on theta_2, normalized by 0.75: 2/3 + 8/3.
    assert torch.allclose(ema.history("w"), torch.tensor([10 / 3, 10 / 3]))
    # Every history value stays inside the range of the observed weights.
    assert 2.0 <= float(ema.history("w")[0]) <= 4.0


def test_uncorrected_shadow_would_be_dominated_by_the_init():
    """Guard the reason for debiasing: a long alpha barely moves off the init."""
    m = M((0.0, 0.0))
    ema = EMAHistory.from_module(m, alpha=0.99)
    m.w.data.fill_(1.0)
    for _ in range(48):
        ema.update_module(m)
    # Raw accumulator has only reached ~38% of the observed value ...
    assert abs(float(ema.shadow["w"][0]) - (1 - 0.99**48)) < 1e-6
    # ... while the debiased history is already exactly the observed weights.
    assert torch.allclose(ema.history("w"), torch.tensor([1.0, 1.0]))


def test_ema_copy_to():
    m = M((1.0,))
    ema = EMAHistory.from_module(m, alpha=0.9)
    m.w.data.fill_(5.0)
    ema.update_module(m)
    hist = M((0.0,))
    ema.copy_to(hist)
    assert torch.allclose(hist.w, ema.history("w"))


def test_reading_history_before_any_update_is_refused():
    ema = EMAHistory.from_module(M(), alpha=0.5)
    with pytest.raises(RuntimeError, match="before any optimizer step"):
        ema.history("w")


def test_swap_to_is_a_noop_before_any_update():
    m = M((7.0, 7.0))
    ema = EMAHistory.from_module(m, alpha=0.5)
    with ema.swap_to(m):
        # Scoring against the live weights gives REL = 0, not REL against a random init.
        assert torch.allclose(m.w, torch.tensor([7.0, 7.0]))
    assert torch.allclose(m.w, torch.tensor([7.0, 7.0]))


def test_ema_swap_to_uses_history_then_restores():
    m = M()
    ema = EMAHistory.from_module(m, alpha=0.5)
    m.w.data.fill_(3.0)
    ema.update_module(m, alpha=0.5)
    m.w.data.fill_(5.0)
    ema.update_module(m, alpha=0.5)  # history = (0.25*3 + 0.5*5) / 0.75
    expected = torch.full((2,), 13 / 3)

    m.w.data.fill_(7.0)  # live weights now 7
    with ema.swap_to(m):
        assert torch.allclose(m.w, expected)  # history weights active
    assert torch.allclose(m.w, torch.tensor([7.0, 7.0]))  # live weights restored

    # Restore must hold even if the body raises.
    try:
        with ema.swap_to(m):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert torch.allclose(m.w, torch.tensor([7.0, 7.0]))


def test_state_round_trip_preserves_the_correction():
    m = M((1.0, 1.0))
    ema = EMAHistory.from_module(m, alpha=0.5)
    m.w.data.fill_(3.0)
    ema.update_module(m)

    restored = EMAHistory.from_module(M((0.0, 0.0)), alpha=0.5)
    restored.load_state_dict(ema.state_dict())
    assert restored.correction == ema.correction
    assert torch.allclose(restored.history("w"), ema.history("w"))


def test_legacy_ema_state_is_refused():
    ema = EMAHistory.from_module(M(), alpha=0.5)
    with pytest.raises(ValueError, match="unsupported EMA state version"):
        ema.load_state_dict({"w": torch.tensor([1.0, 1.0])})


class _FakeDTensor:
    """Duck-typed DTensor: ``to_local()`` returns the live shard storage."""

    def __init__(self, storage: torch.Tensor):
        self._storage = storage

    def to_local(self) -> torch.Tensor:
        return self._storage

    @property
    def requires_grad(self) -> bool:
        return bool(self._storage.requires_grad)


class DTensorParamModule(nn.Module):
    def __init__(self, value=(1.0, 1.0)):
        super().__init__()
        self._w = nn.Parameter(torch.tensor(list(value)))

    def named_parameters(self, prefix="", recurse=True):  # noqa: ARG002
        # Wrap the Parameter (not .data): .data has requires_grad=False, which would
        # drop the param from EMAHistory.from_module's filter.
        yield "w", _FakeDTensor(self._w)


def test_swap_to_works_when_parameters_look_like_dtensors():
    """FSDP2 params are DTensors; cloning them yields plain local Tensors.

    The B200 smoke died on ``p.copy_(saved)`` mixing the two. Operating through
    ``to_local()`` keeps every copy on rank-local storage.
    """
    m = DTensorParamModule((1.0, 1.0))
    ema = EMAHistory.from_module(m, alpha=0.5)
    assert "w" in ema.shadow
    m._w.data.fill_(3.0)
    ema.update_module(m, alpha=0.5)
    m._w.data.fill_(5.0)
    ema.update_module(m, alpha=0.5)
    expected = torch.full((2,), 13 / 3)

    m._w.data.fill_(7.0)
    with ema.swap_to(m):
        assert torch.allclose(m._w.data, expected)
    assert torch.allclose(m._w.data, torch.tensor([7.0, 7.0]))
