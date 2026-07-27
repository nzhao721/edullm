"""Bias-corrected exponential moving-average history weights for REL scoring.

FSDP / distributed note: under FSDP2, ``named_parameters()`` yields ``DTensor`` shards.
Plain ``Tensor``/``DTensor`` mixes are illegal (``aten.copy_`` raises), and
``p.detach().clone()`` on a ``DTensor`` often returns a *local* ``Tensor``, which is
exactly what blew up the first REL-active step on the B200. The accumulator therefore
stores local shards only, and every read/write goes through :func:`_local_tensor` so
``update`` / ``swap_to`` / ``copy_to`` stay on the same rank-local storage the optimizer
touches. For the history forward, prefer :meth:`EMAHistory.swap_to` (swaps history
weights into the *training* model for one no-grad pass, then restores) so no second
full-size module copy is needed under FSDP.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import Tensor, nn


def _local_tensor(t: Tensor) -> Tensor:
    """Rank-local storage for a parameter or EMA buffer (identity for plain Tensors)."""
    to_local = getattr(t, "to_local", None)
    if callable(to_local):
        return to_local()
    return t


def _copy_into_param_(dst: Tensor, src: Tensor) -> None:
    """In-place copy that never mixes a plain Tensor with a DTensor destination."""
    _local_tensor(dst).copy_(_local_tensor(src))


def alpha_at_step(
    step: int,
    *,
    t0: int,
    total_steps: int,
    alpha_start: float,
    alpha_end: float,
) -> float:
    """Piecewise α schedule: constant during warmup, then linear decay to alpha_end.

    During steps ``[0, t0)`` return ``alpha_start`` (history is updated but REL is off).
    During ``[t0, total_steps]`` linearly interpolate from ``alpha_start`` → ``alpha_end``.
    If ``total_steps <= t0``, stay at ``alpha_start``.
    """
    if total_steps <= t0 or step < t0:
        return float(alpha_start)
    denom = max(total_steps - t0, 1)
    frac = min(max((step - t0) / denom, 0.0), 1.0)
    return float(alpha_start + frac * (alpha_end - alpha_start))


class EMAHistory:
    """Bias-corrected moving average of *observed* weights, for REL history scoring.

    The accumulator starts at zero and is normalized on read::

        s_t = α_t·s_{t-1} + (1 − α_t)·θ_t        s_0 = 0
        c_t = α_t·c_{t-1} + (1 − α_t)            c_0 = 0
        θ_hist = s_t / c_t

    ``c_t`` is exactly the total weight accumulated in ``s_t``, so ``s_t / c_t`` is a
    convex combination of the observed weights θ_1..θ_t and the initialization θ_0 never
    appears in it.

    The textbook alternative (seed the shadow with θ_0 and blend in place) leaves θ_0 as
    the single heaviest ingredient for the first ~1/(1−α) optimizer steps. That is
    unusable for from-scratch pretraining: a weight-space blend of a random init and a
    trained model is not an older model but an off-manifold point whose per-token loss is
    nearly flat, which collapses ``REL = L_hist − L_curr`` into ``constant − L_curr`` and
    makes top-k keep the *easiest* tokens instead of the most learnable ones. Warmup does
    not rescue it, because warmup delays when the history is read, not what is in it.

    History tensors live on the same device/dtype/layout as the source parameters and are
    updated in-place under ``torch.no_grad()``.
    """

    STATE_VERSION = 2

    def __init__(self, named_params: Iterable[Tuple[str, Tensor]], *, alpha: float = 0.999):
        self.alpha = float(alpha)
        self._correction = 0.0
        # Local shards only: under FSDP2, cloning a DTensor parameter yields a plain
        # Tensor of the shard, which is what we want to accumulate into.
        self._shadow: Dict[str, Tensor] = {}
        for name, p in named_params:
            self._shadow[name] = _local_tensor(p).detach().clone().zero_()

    @classmethod
    def from_module(cls, module: nn.Module, *, alpha: float = 0.999) -> "EMAHistory":
        return cls(((n, p) for n, p in module.named_parameters() if p.requires_grad), alpha=alpha)

    @property
    def shadow(self) -> Mapping[str, Tensor]:
        """Raw unnormalized accumulator. Use :meth:`history` for usable weights."""
        return self._shadow

    @property
    def correction(self) -> float:
        """Total weight accumulated so far; the divisor that debiases the accumulator."""
        return self._correction

    @property
    def has_history(self) -> bool:
        """False until the first update, when there is nothing to score against yet."""
        return self._correction > 0.0

    def _require_history(self) -> float:
        if not self.has_history:
            raise RuntimeError(
                "EMA history was read before any optimizer step; there are no observed "
                "weights to average. Check has_history first."
            )
        return self._correction

    def history(self, name: str) -> Tensor:
        """Debiased history weights for one parameter."""
        return self._shadow[name] / self._require_history()

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "correction": self._correction,
            "shadow": {k: v.detach().clone() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        version = state.get("version")
        if version != self.STATE_VERSION:
            raise ValueError(
                f"unsupported EMA state version {version!r}; expected {self.STATE_VERSION}. "
                "Pre-v2 checkpoints hold an uncorrected accumulator seeded at the random "
                "initialization and cannot be rescaled into a debiased one."
            )
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA state is missing its shadow weights")
        missing = set(self._shadow) - set(shadow)
        extra = set(shadow) - set(self._shadow)
        if missing or extra:
            raise KeyError(f"EMA state mismatch; missing={sorted(missing)} extra={sorted(extra)}")
        correction = state.get("correction")
        if not isinstance(correction, (int, float)) or not 0.0 <= float(correction) <= 1.0:
            raise ValueError("EMA correction must be a weight in [0, 1]")
        for k, v in shadow.items():
            if not isinstance(v, Tensor):
                raise TypeError(f"EMA shadow entry {k!r} must be a Tensor")
            _copy_into_param_(self._shadow[k], v)
        self._correction = float(correction)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    @torch.no_grad()
    def update(self, named_params: Iterable[Tuple[str, Tensor]], *, alpha: Optional[float] = None) -> None:
        a = self.alpha if alpha is None else float(alpha)
        one_minus = 1.0 - a
        for name, p in named_params:
            shadow = self._shadow.get(name)
            if shadow is None:
                raise KeyError(
                    f"parameter {name!r} is absent from the EMA accumulator; seeding it now "
                    "would give it the wrong weight in the debiased average"
                )
            # s ← α s + (1−α) θ  (both sides are rank-local shards)
            shadow.mul_(a).add_(_local_tensor(p).detach(), alpha=one_minus)
        self._correction = a * self._correction + one_minus

    @torch.no_grad()
    def update_module(self, module: nn.Module, *, alpha: Optional[float] = None) -> None:
        self.update(((n, p) for n, p in module.named_parameters() if p.requires_grad), alpha=alpha)

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        """Overwrite ``module`` parameters with the debiased history (for history forward)."""
        correction = self._require_history()
        for name, p in module.named_parameters():
            if name in self._shadow:
                _copy_into_param_(p, self._shadow[name] / correction)

    @contextlib.contextmanager
    def swap_to(self, module: nn.Module):
        """Temporarily load history weights into ``module`` for one forward.

        FSDP-friendly alternative to keeping a second full model: we snapshot the live
        parameters, write the debiased history in, run the history forward under the
        ``with`` block, then restore. Must be used for a *no-grad* pass and fully exited
        before the next backward, so autograd never sees the swapped weights.

        Before the first optimizer step there is no observed history, so this is a no-op
        and the caller scores against the current weights (REL = 0) rather than against
        the random initialization.
        """
        if not self.has_history:
            yield module
            return
        correction = self._correction
        saved: Dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in self._shadow:
                        # Snapshot the local shard; restoring via _copy_into_param_ keeps
                        # DTensor destinations happy under FSDP2.
                        saved[name] = _local_tensor(p).detach().clone()
                        _copy_into_param_(p, self._shadow[name] / correction)
            yield module
        finally:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in saved:
                        _copy_into_param_(p, saved[name])

    def named_history_params(self) -> Iterator[Tuple[str, Tensor]]:
        correction = self._require_history()
        for name, shadow in self._shadow.items():
            yield name, shadow / correction

    def apply_to_modules(
        self,
        target: nn.Module,
        source_named: Optional[Iterable[Tuple[str, Tensor]]] = None,
    ) -> None:
        """Compatibility helper: prefer ``copy_to`` for full replace."""
        if source_named is not None:
            mapping: MutableMapping[str, Tensor] = dict(source_named)
            for name, p in target.named_parameters():
                if name in mapping:
                    _copy_into_param_(p, mapping[name])
            return
        self.copy_to(target)
