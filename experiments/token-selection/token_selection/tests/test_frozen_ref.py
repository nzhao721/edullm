"""Tests for FrozenReference shadow + swap."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from token_selection.olmo_ext.frozen_ref import FrozenReference
from token_selection.olmo_ext.train_module import (
    TokenSelectConfig,
    TokenSelectLoop,
    TokenSelectState,
    load_reference_state_dict,
)


class M(nn.Module):
    def __init__(self, value=(1.0, 2.0)):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(list(value)))


def test_frozen_ref_swap_restores_live_weights():
    live = M((7.0, 7.0))
    ref_src = M((3.0, 4.0))
    frozen = FrozenReference.from_module(ref_src)
    with frozen.swap_to(live):
        assert torch.allclose(live.w, torch.tensor([3.0, 4.0]))
    assert torch.allclose(live.w, torch.tensor([7.0, 7.0]))


def test_load_reference_state_dict_from_pt(tmp_path):
    path = tmp_path / "ref.pt"
    torch.save({"w": torch.tensor([9.0, 8.0])}, path)
    loaded = load_reference_state_dict(path)
    assert torch.equal(loaded["w"], torch.tensor([9.0, 8.0]))

    wrapped = tmp_path / "wrapped.pt"
    torch.save({"model": {"w": torch.tensor([1.0, 2.0])}}, wrapped)
    loaded2 = load_reference_state_dict(wrapped)
    assert torch.equal(loaded2["w"], torch.tensor([1.0, 2.0]))


def test_load_reference_refuses_s3():
    with pytest.raises(ValueError, match="remote"):
        load_reference_state_dict("s3://bucket/ref.pt")


def test_rho_state_roundtrip(tmp_path):
    cfg = TokenSelectConfig(method="rho_excess", t0_steps=0, total_steps=4)
    model = M()
    frozen = FrozenReference.from_module(M((5.0, 6.0)))
    state = TokenSelectState(cfg, model, frozen_ref=frozen)
    state.after_optim_step(model)
    ckpt = state.state_dict()

    restored_model = M()
    restored = TokenSelectState(
        cfg, restored_model, frozen_ref=FrozenReference.from_module(M((0.0, 0.0)))
    )
    restored.load_state_dict(ckpt)
    assert restored.step == 1
    assert restored.frozen_ref is not None
    assert torch.allclose(restored.frozen_ref.shadow["w"], torch.tensor([5.0, 6.0]))


def test_rho_loop_selects_and_scores():
    torch.manual_seed(0)
    train = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16, bias=False))
    # Flatten to TinyLM-like: wrap
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.out = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return self.out(self.embed(x))

    torch.manual_seed(0)
    model = Tiny()
    torch.manual_seed(1)
    ref = FrozenReference.from_module(Tiny())
    cfg = TokenSelectConfig(method="rho_excess", k=0.5, t0_steps=0, total_steps=2)
    loop = TokenSelectLoop(model, cfg, frozen_ref=ref)
    ids = torch.randint(0, 16, (2, 8))
    out = loop.train_step(ids)
    assert out["compute"]["fwd_passes_history"] == 1
    assert out["mean_score_kept"] is not None
    assert 0.0 < out["selected_frac"] <= 1.0


def test_frozen_ref_not_updated_by_optim_or_train_steps():
    """RHO reference is a weight shadow: optim steps must never blend it toward θ."""

    class Tiny(nn.Module):
        def __init__(self, seed: int):
            super().__init__()
            torch.manual_seed(seed)
            self.embed = nn.Embedding(16, 8)
            self.out = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return self.out(self.embed(x))

    torch.manual_seed(0)
    model = Tiny(0)
    torch.manual_seed(1)
    frozen = FrozenReference.from_module(Tiny(1))
    snap = {k: v.clone() for k, v in frozen.shadow.items()}
    cfg = TokenSelectConfig(method="rho_excess", k=0.5, t0_steps=0, total_steps=4)
    loop = TokenSelectLoop(model, cfg, frozen_ref=frozen)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    kept: list[float] = []
    dropped: list[float] = []
    for _ in range(4):
        ids = torch.randint(0, 16, (2, 8))
        out = loop.train_step(ids)
        out["loss"].backward()
        opt.step()
        loop.optim_step_done()
        if out["mean_score_kept"] is not None:
            kept.append(float(out["mean_score_kept"]))
        if out["mean_score_dropped"] is not None:
            dropped.append(float(out["mean_score_dropped"]))
    assert all(torch.equal(snap[k], frozen.shadow[k]) for k in snap)
    assert kept and dropped
    assert (sum(kept) / len(kept)) > (sum(dropped) / len(dropped))


def test_learnability_loop_dual_ref_scoring():
    """Two frozen refs; score path runs two history forwards; selection active."""

    class Tiny(nn.Module):
        def __init__(self, seed: int):
            super().__init__()
            torch.manual_seed(seed)
            self.embed = nn.Embedding(16, 8)
            self.out = nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return self.out(self.embed(x))

    torch.manual_seed(0)
    model = Tiny(0)
    early = FrozenReference.from_module(Tiny(1))
    late = FrozenReference.from_module(Tiny(2))
    early_snap = {k: v.clone() for k, v in early.shadow.items()}
    late_snap = {k: v.clone() for k, v in late.shadow.items()}
    cfg = TokenSelectConfig(method="learnability", k=0.5, t0_steps=0, total_steps=2)
    loop = TokenSelectLoop(
        model, cfg, frozen_ref_early=early, frozen_ref_late=late
    )
    ids = torch.randint(0, 16, (2, 8))
    out = loop.train_step(ids)
    assert out["method"] == "learnability"
    assert out["compute"]["fwd_passes_history"] == 2
    assert out["mean_score_kept"] is not None
    assert 0.0 < out["selected_frac"] <= 1.0
    assert all(torch.equal(early_snap[k], early.shadow[k]) for k in early_snap)
    assert all(torch.equal(late_snap[k], late.shadow[k]) for k in late_snap)


def test_average_reference_state_dicts():
    from token_selection.olmo_ext.train_module import average_reference_state_dicts

    a = {"w": torch.tensor([0.0, 2.0])}
    b = {"w": torch.tensor([2.0, 4.0])}
    c = {"w": torch.tensor([4.0, 6.0])}
    avg = average_reference_state_dicts([a, b, c])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 4.0]))


def test_load_weights_rejects_shape_mismatch():
    live = M((1.0, 2.0))
    frozen = FrozenReference.from_module(live)
    with pytest.raises(ValueError, match="shape"):
        frozen.load_weights({"w": torch.tensor([1.0, 2.0, 3.0])})


def test_from_state_dict_keeps_full_weights_against_dtensor_module():
    """Checkpoints stay full-size; DTensor module only validates global shape."""

    class _FakeDTensor:
        def __init__(self, local: torch.Tensor, *, global_rows: int):
            self._local = local
            self.shape = torch.Size((global_rows, local.shape[1]))
            self.device_mesh = object()
            self.placements = (object(),)
            self.device = local.device
            self.dtype = local.dtype

        def to_local(self) -> torch.Tensor:
            return self._local

        def full_tensor(self) -> torch.Tensor:
            # Pretend the full weight is zeros with global rows.
            return torch.zeros(self.shape, dtype=self._local.dtype)

    class Sharded(nn.Module):
        def __init__(self):
            super().__init__()
            self._w = torch.zeros(2, 4)

        def named_parameters(self, prefix="", recurse=True):  # noqa: ARG002
            yield "w", _FakeDTensor(self._w, global_rows=8)

    full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    frozen = FrozenReference.from_state_dict(Sharded(), {"w": full})
    assert torch.equal(frozen.shadow["w"], full)


def test_swap_to_restores_after_allgather_shape_change(monkeypatch):
    """FSDP may all-gather during the scoring forward; restore must still work."""

    class _FakeDTensor:
        def __init__(self, local: torch.Tensor, *, global_rows: int, gathered: bool = False):
            self._local = local
            self._global_rows = global_rows
            self.gathered = gathered
            self.device_mesh = object()
            self.placements = (object(),)
            self.device = local.device
            self.dtype = local.dtype

        @property
        def shape(self):
            return torch.Size((self._global_rows, self._local.shape[-1]))

        def to_local(self) -> torch.Tensor:
            return self._local

        def full_tensor(self) -> torch.Tensor:
            if self.gathered or self._local.shape[0] == self._global_rows:
                return self._local.detach().clone()
            out = torch.zeros(self.shape, dtype=self._local.dtype)
            out[: self._local.shape[0]].copy_(self._local)
            return out

    live_local = torch.ones(2, 4)
    param = _FakeDTensor(live_local, global_rows=8, gathered=False)

    class Mod(nn.Module):
        def named_parameters(self, prefix="", recurse=True):  # noqa: ARG002
            yield "w", param

    ref_full = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    frozen = FrozenReference.from_state_dict(Mod(), {"w": ref_full})

    def _fake_distribute(src, mesh, placements):  # noqa: ARG001
        class _Out:
            def to_local(self_inner):
                return src[:2].detach().clone()

        return _Out()

    import sys
    from types import ModuleType

    fake_mod = ModuleType("torch.distributed.tensor")
    fake_mod.distribute_tensor = _fake_distribute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", fake_mod)

    expected_restore = param.full_tensor()
    with frozen.swap_to(Mod()):
        # Simulate FSDP all-gather mid-forward: local storage becomes full garbage.
        param.gathered = True
        param._local = torch.randn(8, 4)

    assert torch.equal(param._local, expected_restore)
