"""Integration-ish tests for TokenSelectLoop (full + REL+EMA)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from token_selection.olmo_ext.frozen_ref import FrozenReference
from token_selection.olmo_ext.train_module import (
    RELCallback,
    TokenSelectConfig,
    TokenSelectLoop,
    TokenSelectState,
    has_olmo_core,
    masked_ce_from_token_ce,
    per_token_ce,
)


def _shifted_labels_ce(logits, input_ids, label_mask, ignore_index=-100):
    """Independent reference: build shifted labels with an ignore index, then one CE.

    This is the formulation the train module used before the per-token CE was folded, and
    it must stay numerically equivalent -- the folded version exists to avoid a second
    log-softmax the size of the logits, not to change the objective.
    """
    labels = input_ids.clone().masked_fill(~label_mask.to(torch.bool), ignore_index)
    labels = F.pad(labels[:, 1:], (0, 1), value=ignore_index)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss, int((labels != ignore_index).sum())


def test_folded_ce_matches_shifted_labels_in_value_and_gradient():
    torch.manual_seed(0)
    input_ids = torch.randint(1, 32, (3, 12))
    label_mask = torch.rand(3, 12) < 0.6
    label_mask[:, 0] = False  # nothing predicts the first token

    base = torch.randn(3, 12, 32, dtype=torch.float64)
    folded_logits = base.clone().requires_grad_(True)
    reference_logits = base.clone().requires_grad_(True)

    folded_loss, folded_n = masked_ce_from_token_ce(
        per_token_ce(folded_logits, input_ids), label_mask
    )
    reference_loss, reference_n = _shifted_labels_ce(reference_logits, input_ids, label_mask)

    assert folded_n == reference_n == int(label_mask.sum())
    assert torch.allclose(folded_loss.double(), reference_loss, rtol=1e-9, atol=1e-9)

    folded_loss.backward()
    reference_loss.backward()
    assert torch.allclose(folded_logits.grad, reference_logits.grad, rtol=1e-9, atol=1e-9)


def test_folded_ce_ignores_positions_outside_the_mask():
    torch.manual_seed(1)
    input_ids = torch.randint(1, 32, (2, 8))
    logits = torch.randn(2, 8, 32, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 3] = True

    loss, n = masked_ce_from_token_ce(per_token_ce(logits, input_ids), mask)
    assert n == 2
    loss.backward()
    # Only the position that predicts target 3 (logits index 2) may receive gradient.
    touched = {int(i) for i in logits.grad.abs().sum(dim=-1).nonzero()[:, 1].unique()}
    assert touched == {2}


def test_masked_z_loss_matches_olmo_core_formula():
    from token_selection.olmo_ext.train_module import masked_z_from_token_z, per_token_z_loss

    torch.manual_seed(2)
    logits = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 0] = False
    lam = 1e-5
    token_z = per_token_z_loss(logits, z_loss_multiplier=lam)
    z_sum = masked_z_from_token_z(token_z, mask)
    # Independent: z on shifted logits, mask on target positions 1..T-1.
    shift = logits[:, :-1, :].float()
    z_sq = shift.logsumexp(-1).pow(2)
    expected = (lam * z_sq * mask[:, 1:]).sum()
    assert torch.allclose(z_sum, expected, rtol=1e-9, atol=1e-9)
    z_sum.backward()
    assert logits.grad is not None
    assert logits.grad[:, -1].abs().sum() == 0  # last logit row unused (no next target)



class Tiny(nn.Module):
    def __init__(self, v: int = 32, d: int = 16):
        super().__init__()
        self.embed = nn.Embedding(v, d)
        self.out = nn.Linear(d, v, bias=False)

    def forward(self, x):
        return self.out(self.embed(x))


def test_rel_warmup_then_select():
    torch.manual_seed(0)
    model = Tiny()
    cfg = TokenSelectConfig(method="rel_ema", k=0.6, t0_steps=2, total_steps=6, seed=0)
    loop = TokenSelectLoop(model, cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    fracs = []
    for _ in range(6):
        x = torch.randint(1, 32, (2, 16))
        out = loop.train_step(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        loop.optim_step_done()
        fracs.append(out["selected_frac"])
        assert out["loss"].isfinite()
        assert out["scoring_tokens"] >= 0
    assert fracs[0] > 0.9
    assert fracs[1] > 0.9
    assert 0.4 < fracs[-1] < 0.8


def test_random_keep_fraction():
    torch.manual_seed(0)
    model = Tiny()
    cfg = TokenSelectConfig(method="random", k=0.6, t0_steps=0, total_steps=4, seed=7)
    loop = TokenSelectLoop(model, cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    fracs = []
    for _ in range(4):
        x = torch.randint(1, 32, (2, 16))
        out = loop.train_step(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        loop.optim_step_done()
        assert out["scoring_tokens"] == 0
        assert out["method"] == "random"
        fracs.append(out["selected_frac"])
    assert all(0.45 < f < 0.75 for f in fracs)


def test_full_no_scoring_overhead():
    torch.manual_seed(0)
    model = Tiny()
    cfg = TokenSelectConfig(method="full", k=0.6, t0_steps=0, total_steps=4, seed=0)
    loop = TokenSelectLoop(model, cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(4):
        x = torch.randint(1, 32, (2, 16))
        out = loop.train_step(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        loop.optim_step_done()
        assert out["scoring_tokens"] == 0
        assert out["selected_frac"] > 0.9
        assert out["method"] == "full"


def test_fold_scoring_uses_two_forwards():
    torch.manual_seed(0)
    model = Tiny()
    cfg = TokenSelectConfig(method="rel_ema", k=0.6, t0_steps=1, total_steps=6, seed=0)
    loop = TokenSelectLoop(model, cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randint(1, 32, (2, 16))

    warm = loop.train_step(x)
    assert warm["warmup"] is True
    assert warm["compute"]["forward_tokens_history"] == 0
    assert warm["compute"]["forward_tokens_current"] == 0
    opt.zero_grad()
    warm["loss"].backward()
    opt.step()
    loop.optim_step_done()

    out = loop.train_step(x)
    assert out["warmup"] is False
    c = out["compute"]
    # Folded: current score comes from the training forward -> only history is extra.
    assert c["forward_tokens_train"] == x.numel()
    assert c["forward_tokens_history"] == x.numel()
    assert c["forward_tokens_current"] == 0
    assert c["fwd_passes_current"] == 0
    assert out["scoring_tokens"] == x.numel()  # history only, not 2x


def test_step_and_ema_advance_only_on_optim_step():
    torch.manual_seed(0)
    model = Tiny()
    cfg = TokenSelectConfig(method="rel_ema", k=0.6, t0_steps=0, total_steps=6, seed=0)
    loop = TokenSelectLoop(model, cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    st = loop.state
    shadow0 = st.ema.shadow["out.weight"].clone()

    x = torch.randint(1, 32, (2, 16))
    step_before = st.step
    out = loop.train_step(x)
    # The scoring/mask step must not advance step or EMA; that is the per-optim-step job.
    assert st.step == step_before
    assert torch.allclose(st.ema.shadow["out.weight"], shadow0)

    opt.zero_grad()
    out["loss"].backward()
    opt.step()
    loop.optim_step_done()
    assert st.step == step_before + 1
    assert not torch.allclose(st.ema.shadow["out.weight"], shadow0)


def test_state_without_history_module_uses_swap():
    model = Tiny()
    cfg = TokenSelectConfig(method="rel_ema", k=0.6, t0_steps=0, total_steps=4)
    st = TokenSelectState(cfg, model, build_history_module=False)
    assert st.ema is not None
    assert st.history_model is None  # FSDP path keeps no second full module
    x = torch.randint(1, 32, (2, 8))
    with st.ema.swap_to(model):
        logits = model(x)
    assert logits.shape[-1] == 32


def test_middle_ppl_warmup_then_select_with_frozen_ref():
    torch.manual_seed(0)
    model = Tiny()
    frozen = FrozenReference.from_module(model)
    cfg = TokenSelectConfig(method="middle_ppl", k=0.6, t0_steps=2, total_steps=6, seed=0)
    loop = TokenSelectLoop(model, cfg, frozen_ref=frozen)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    fracs = []
    scoring_tokens_hist = []
    for _ in range(6):
        x = torch.randint(1, 32, (2, 16))
        out = loop.train_step(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        loop.optim_step_done()
        fracs.append(out["selected_frac"])
        scoring_tokens_hist.append(out["scoring_tokens"])
        assert out["loss"].isfinite()
        assert out["method"] == "middle_ppl"
    assert scoring_tokens_hist[0] == 0
    assert scoring_tokens_hist[1] == 0
    assert scoring_tokens_hist[-1] > 0
    assert fracs[0] > 0.9
    assert fracs[1] > 0.9
    assert 0.4 < fracs[-1] < 0.8


def test_relcallback_importable_without_olmo():
    assert has_olmo_core() is False
    cb = RELCallback()
    assert hasattr(cb, "post_step")
