"""OLMo-core integration checks, skipped when OLMo-core is unavailable locally."""

from __future__ import annotations

import pytest


olmo_core = pytest.importorskip("olmo_core", reason="cluster-only OLMo-core integration")


def test_pinned_callback_and_transformer_module_seams_are_present():
    from olmo_core.train.callbacks import Callback
    from olmo_core.train.train_module.transformer.train_module import TransformerTrainModule

    from token_selection.olmo_ext.callbacks import RawComputeCallback
    from token_selection.olmo_ext.train_module import RELCallback, TokenSelectTrainModule

    assert issubclass(RELCallback, Callback)
    assert issubclass(RawComputeCallback, Callback)
    assert issubclass(TokenSelectTrainModule, TransformerTrainModule)
    assert hasattr(RELCallback, "post_train_batch")
    assert hasattr(RELCallback, "post_step")
    assert hasattr(TokenSelectTrainModule, "consume_token_selection_compute_delta")
