"""Attention backend resolution prefers flash_2, falls back to torch."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from token_selection.scripts.train_olmo_template import resolve_attn_backend


def _install_fake_attn(monkeypatch, *, flash_supported: bool = True):
    class AttentionBackendName:
        torch = SimpleNamespace(name="torch")

        class _Flash:
            name = "flash_2"

            @staticmethod
            def get_class():
                if flash_supported:
                    return SimpleNamespace(assert_supported=lambda: None)
                raise RuntimeError("flash unsupported")

        flash_2 = _Flash()

        def __new__(cls, prefer):
            raise ValueError(prefer)

    monkeypatch.setitem(sys.modules, "olmo_core", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "olmo_core.nn", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "olmo_core.nn.attention",
        SimpleNamespace(AttentionBackendName=AttentionBackendName),
    )


def test_resolve_attn_respects_torch(monkeypatch):
    _install_fake_attn(monkeypatch)
    monkeypatch.setenv("OLMO_ATTN_BACKEND", "flash_2")
    backend = resolve_attn_backend({"train": {"attn_backend": "torch"}})
    assert backend.name == "torch"


def test_resolve_attn_auto_uses_flash_when_available(monkeypatch):
    _install_fake_attn(monkeypatch, flash_supported=True)
    monkeypatch.setenv("OLMO_ATTN_BACKEND", "auto")
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(__version__="2.7"))
    backend = resolve_attn_backend({"train": {}})
    assert backend.name == "flash_2"


def test_resolve_attn_auto_falls_back_without_flash(monkeypatch):
    _install_fake_attn(monkeypatch, flash_supported=False)
    monkeypatch.setenv("OLMO_ATTN_BACKEND", "auto")
    # Missing flash_attn -> ImportError path.
    sys.modules.pop("flash_attn", None)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        None,  # forces ImportError on import in CPython
    )
    backend = resolve_attn_backend({"train": {"attn_backend": "auto"}})
    assert backend.name == "torch"
