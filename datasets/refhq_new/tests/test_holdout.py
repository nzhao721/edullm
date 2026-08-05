"""Unit tests for document holdout math and conversation flattening."""

from __future__ import annotations

import pytest

from normalize_filter_source import flatten_conversation
from refhq_new_sources import HOLDOUT_FRACTION, holdout_counts


def test_holdout_fraction_constant() -> None:
    assert HOLDOUT_FRACTION == 0.0015


def test_holdout_counts_typical() -> None:
    # 10_000 * 0.0015 = 15
    n_train, n_val = holdout_counts(10_000, 0.0015)
    assert n_train + n_val == 10_000
    assert n_val == 15
    assert n_train == 9_985


def test_holdout_counts_rounds() -> None:
    # 1_000 * 0.0015 = 1.5 → round to 2
    n_train, n_val = holdout_counts(1_000, 0.0015)
    assert n_val == 2
    assert n_train == 998


def test_holdout_counts_small_pool_keeps_train() -> None:
    n_train, n_val = holdout_counts(1, 0.0015)
    assert n_train == 1
    assert n_val == 0


def test_holdout_counts_empty() -> None:
    assert holdout_counts(0) == (0, 0)


def test_holdout_counts_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        holdout_counts(100, 0.0)
    with pytest.raises(ValueError):
        holdout_counts(100, 0.5)


def test_flatten_messages_role_content() -> None:
    text = flatten_conversation(
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
    )
    assert text == "User: Hi\n\nAssistant: Hello"


def test_flatten_openhermes_from_value() -> None:
    text = flatten_conversation(
        {
            "conversations": [
                {"from": "human", "value": "2+2?"},
                {"from": "gpt", "value": "4"},
            ]
        }
    )
    assert text == "User: 2+2?\n\nAssistant: 4"


def test_flatten_plain_text_when_no_messages() -> None:
    assert flatten_conversation({"text": "already flat"}) == "already flat"
