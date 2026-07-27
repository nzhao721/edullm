"""Checkpoint-state tests that run without OLMo-core installed."""

from __future__ import annotations

import torch
import torch.nn as nn

from token_selection.olmo_ext.train_module import (
    RELCallback,
    TokenSelectConfig,
    TokenSelectState,
)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))


def test_token_select_state_restores_ema_and_step() -> None:
    cfg = TokenSelectConfig(method="rel_ema", t0_steps=1, total_steps=10, alpha_start=0.5)
    model = Tiny()
    state = TokenSelectState(cfg, model, build_history_module=False)

    model.weight.data.add_(4.0)
    state.after_optim_step(model)
    checkpoint = state.state_dict()

    restored_model = Tiny()
    restored = TokenSelectState(cfg, restored_model, build_history_module=False)
    restored.load_state_dict(checkpoint)

    assert restored.step == 1
    assert restored.tokens_seen == 0
    assert restored.ema is not None
    assert state.ema is not None
    assert torch.equal(restored.ema.shadow["weight"], state.ema.shadow["weight"])
    assert restored.ema.alpha == restored.current_alpha()


def test_rel_callback_persists_and_advances_once_per_trainer_step() -> None:
    cfg = TokenSelectConfig(method="rel_ema", t0_steps=0, total_steps=10, alpha_start=0.5)
    model = Tiny()

    class FakeTrainModule:
        def __init__(self, model: Tiny) -> None:
            self.model = model
            self.state = TokenSelectState(cfg, model, build_history_module=False)

        def on_optim_step_end(self) -> None:
            self.state.after_optim_step(self.model)

        def token_selection_state_dict(self):
            return self.state.state_dict()

        def load_token_selection_state(self, state):
            self.state.load_state_dict(state)

    class FakeTrainer:
        def __init__(self, train_module: FakeTrainModule, global_step: int) -> None:
            self.train_module = train_module
            self.global_step = global_step

    train_module = FakeTrainModule(model)
    callback = RELCallback()
    callback.trainer = FakeTrainer(train_module, global_step=1)
    model.weight.data.add_(2.0)

    # post_train_batch is used before OLMo-core's checkpointer, and post_step is
    # intentionally idempotent for that same completed optimizer step.
    callback.post_train_batch()
    callback.post_step()
    assert train_module.state.step == 1
    checkpoint = callback.state_dict()

    restored_module = FakeTrainModule(Tiny())
    restored_callback = RELCallback()
    restored_callback.trainer = FakeTrainer(restored_module, global_step=1)
    restored_callback.load_state_dict(checkpoint)

    assert restored_module.state.step == 1
    assert restored_module.state.ema is not None
    assert train_module.state.ema is not None
    assert torch.equal(
        restored_module.state.ema.shadow["weight"], train_module.state.ema.shadow["weight"]
    )
