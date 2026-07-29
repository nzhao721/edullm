"""Frozen reference-model weight shadow for RHO-1 excess-loss scoring.

Stores a **full** (unsharded) copy of reference parameters and temporarily swaps
them into the training module for a no-grad scoring forward. Under FSDP2 / HSDP
the live parameters are DTensor shards that may all-gather during the forward, so
enter/exit must snapshot and restore via global tensors — not rank-local shards
alone (restoring a shard into an all-gathered buffer raises size mismatches).
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterable, Iterator, Mapping, Tuple

import torch
from torch import Tensor, nn

from .ema import _local_tensor


def _distribute_tensor(src: Tensor, param: Tensor):
    try:
        from torch.distributed.tensor import distribute_tensor
    except ImportError:  # pragma: no cover - older torch
        from torch.distributed._tensor import distribute_tensor  # type: ignore

    return distribute_tensor(src, param.device_mesh, param.placements)


def _snapshot_param(param: Tensor) -> Tensor:
    """Capture a restore-safe snapshot (prefer global / full tensor under FSDP2)."""
    full_fn = getattr(param, "full_tensor", None)
    if callable(full_fn):
        return full_fn().detach().clone()
    return _local_tensor(param).detach().clone()


def _write_param_(param: Tensor, values: Tensor) -> None:
    """Write ``values`` (global or matching-local) into ``param``'s current storage."""
    local = _local_tensor(param)
    src = _local_tensor(values).detach()
    if tuple(local.shape) == tuple(src.shape):
        local.copy_(src.to(device=local.device, dtype=local.dtype))
        return

    device_mesh = getattr(param, "device_mesh", None)
    placements = getattr(param, "placements", None)
    global_shape = tuple(param.shape)
    if device_mesh is not None and placements is not None and tuple(src.shape) == global_shape:
        dt = _distribute_tensor(
            src.to(device=local.device, dtype=local.dtype),
            param,
        )
        local.copy_(_local_tensor(dt))
        return

    raise RuntimeError(
        f"cannot write weight: local shape {tuple(local.shape)}, value shape "
        f"{tuple(src.shape)}, param global shape {global_shape}"
    )


class FrozenReference:
    """Immutable parameter shadow used as L_ref in ``excess = L_curr − L_ref``."""

    # v2: shadow tensors are always full/global shapes (not FSDP local shards).
    STATE_VERSION = 2

    def __init__(self, named_params: Iterable[Tuple[str, Tensor]]):
        self._shadow: Dict[str, Tensor] = {}
        for name, p in named_params:
            # Prefer global / full tensors. Using to_local() here would quietly
            # drop HSDP shards and persist rank-0-only weights into checkpoints.
            full_fn = getattr(p, "full_tensor", None)
            if callable(full_fn):
                self._shadow[name] = full_fn().detach().clone()
            else:
                self._shadow[name] = _local_tensor(p).detach().clone()
        if not self._shadow:
            raise ValueError("FrozenReference requires at least one parameter")

    @classmethod
    def from_module(cls, module: nn.Module) -> "FrozenReference":
        """Snapshot ``module`` parameters (typically a loaded reference checkpoint)."""
        pairs: list[Tuple[str, Tensor]] = []
        for name, p in module.named_parameters():
            pairs.append((name, _snapshot_param(p)))
        return cls(pairs)

    @classmethod
    def from_state_dict(
        cls,
        module: nn.Module,
        state_dict: Mapping[str, Tensor],
    ) -> "FrozenReference":
        """Store full reference weights; ``module`` is only used for key/shape checks.

        ``state_dict`` must be a full unsharded checkpoint whose keys match
        ``module.named_parameters()`` and whose shapes match each parameter's
        **global** shape (``param.shape`` under DTensor).
        """
        missing = [n for n, _ in module.named_parameters() if n not in state_dict]
        if missing:
            raise KeyError(
                f"reference state_dict missing parameters: {sorted(missing)[:8]}"
                + ("…" if len(missing) > 8 else "")
            )
        pairs: list[Tuple[str, Tensor]] = []
        for name, p in module.named_parameters():
            value = state_dict[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"reference weight {name!r} must be a Tensor")
            src = _local_tensor(value).detach().clone()
            global_shape = tuple(p.shape)
            if tuple(src.shape) != global_shape:
                raise ValueError(
                    f"reference weight {name!r} shape {tuple(src.shape)} does not "
                    f"match parameter global shape {global_shape}"
                )
            pairs.append((name, src))
        return cls(pairs)

    @property
    def shadow(self) -> Mapping[str, Tensor]:
        return self._shadow

    def load_weights(self, state_dict: Mapping[str, Tensor]) -> None:
        """Overwrite shadow tensors from a flat parameter state dict."""
        missing = set(self._shadow) - set(state_dict)
        # Allow extra keys (buffers, non-trainable) but refuse missing params.
        if missing:
            raise KeyError(
                f"reference state_dict missing parameters: {sorted(missing)[:8]}"
                + ("…" if len(missing) > 8 else "")
            )
        for name in self._shadow:
            value = state_dict[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"reference weight {name!r} must be a Tensor")
            local = _local_tensor(value)
            if local.shape != self._shadow[name].shape:
                raise ValueError(
                    f"reference weight {name!r} shape {tuple(local.shape)} does not "
                    f"match shadow shape {tuple(self._shadow[name].shape)}"
                )
            self._shadow[name].copy_(local.detach())

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "shadow": {k: v.detach().clone() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        version = state.get("version")
        if version not in (1, self.STATE_VERSION):
            raise ValueError(
                f"unsupported FrozenReference state version {version!r}; "
                f"expected {self.STATE_VERSION}"
            )
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("FrozenReference state is missing its shadow weights")
        # v1 shards cannot be reloaded into v2 full shadows; refuse clearly.
        if version == 1:
            raise ValueError(
                "FrozenReference state version 1 (FSDP-local shards) cannot be "
                "loaded into version 2 (full weights); rebuild from reference.load_path"
            )
        self.load_weights(shadow)  # type: ignore[arg-type]

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if name in self._shadow:
                _write_param_(p, self._shadow[name])

    @contextlib.contextmanager
    def swap_to(self, module: nn.Module):
        """Temporarily load reference weights into ``module`` for one no-grad forward."""
        saved: Dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in self._shadow:
                        saved[name] = _snapshot_param(p)
                        _write_param_(p, self._shadow[name])
            yield module
        finally:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in saved:
                        _write_param_(p, saved[name])

    def named_params(self) -> Iterator[Tuple[str, Tensor]]:
        yield from self._shadow.items()
