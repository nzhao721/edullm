"""Token-selection train helpers + optional OLMo-core TrainModule subclass.

Supports ``method=full``, ``random`` (uniform random keep-k), ``rel_ema`` (REL + EMA),
``rho_excess`` (frozen-ref excess loss), ``middle_ppl`` (middle-k by frozen-ref CE),
``attention_topk`` (last-layer attention-received), and ``learnability`` (dual frozen
RefHQ early−late). When ``olmo_core`` is not installed, ``TokenSelectLoop`` still
runs for local smokes.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention_score import capture_last_layer_attention_input, scores_from_capture
from .ema import DEFAULT_ALPHA_TAU, EMAHistory, alpha_at_step
from .frozen_ref import FrozenReference
from .scorers import MethodName, build_mask


@dataclass
class TokenSelectConfig:
    method: MethodName = "rel_ema"
    k: float = 0.6
    t0_steps: int = 0
    total_steps: int = 1000
    alpha_start: float = 0.999
    alpha_end: float = 0.995
    # ``linear`` (legacy piecewise) or ``exp`` (α=1−exp(−t/τ); rel-ema-exp, τ=300).
    alpha_schedule: str = "linear"
    alpha_tau: float = 300.0
    label_ignore_index: int = -100
    seed: int = 42
    # Production RHO: local path to a reference checkpoint (fingerprinted). Smoke may
    # omit this and pass an in-memory FrozenReference into TokenSelectState instead.
    # RefHQ-seeded REL (``ema_seed_mode="refhq"``) also loads this path into EMA history.
    reference_load_path: Optional[str] = None
    # ``"zero"``: bias-corrected accumulator from empty (rel-ema-exp). ``"refhq"``: seed
    # history buffers from ``reference_load_path`` (rel-ema-refhq ONLY).
    ema_seed_mode: Literal["zero", "refhq"] = "zero"
    # Learnability dual frozen refs (early = RefHQ step250; late = avg of late steps).
    early_reference_load_path: Optional[str] = None
    late_reference_load_path: Optional[str] = None

    @property
    def uses_random(self) -> bool:
        return self.method == "random"

    @property
    def uses_rel(self) -> bool:
        return self.method == "rel_ema"

    @property
    def uses_rho(self) -> bool:
        return self.method == "rho_excess"

    @property
    def uses_middle_ppl(self) -> bool:
        return self.method == "middle_ppl"

    @property
    def uses_attention(self) -> bool:
        return self.method == "attention_topk"

    @property
    def uses_learnability(self) -> bool:
        return self.method == "learnability"

    @property
    def uses_selection(self) -> bool:
        return (
            self.uses_random
            or self.uses_rel
            or self.uses_rho
            or self.uses_middle_ppl
            or self.uses_attention
            or self.uses_learnability
        )

    @property
    def needs_scoring_forward(self) -> bool:
        """True when selection needs ≥1 no-grad scoring forward."""
        return (
            self.uses_rel
            or self.uses_rho
            or self.uses_middle_ppl
            or self.uses_learnability
        )


def load_reference_state_dict(path: Union[str, Path]) -> Dict[str, Tensor]:
    """Load a flat parameter state dict from a local reference checkpoint.

    Accepts a ``.pt``/``.pth`` file (raw state dict, or a dict with ``model`` /
    ``state_dict`` / ``model_state_dict``) or a directory containing one of those
    filenames. Remote ``s3://`` URIs must be synced to a local path before launch.
    """
    raw = str(path)
    if raw.startswith("s3://"):
        raise ValueError(
            f"reference.load_path={raw!r} is remote; sync the checkpoint to a local "
            "path and point reference.load_path at that file before launch."
        )
    p = Path(raw)
    if p.is_dir():
        candidates = [
            p / "model.pt",
            p / "model.pth",
            p / "pytorch_model.bin",
            p / "model.safetensors",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"No model.pt / model.pth under reference directory {p}"
            )
        p = found
    if not p.exists():
        raise FileNotFoundError(f"reference checkpoint not found: {p}")
    if p.suffix == ".safetensors":
        raise ValueError(
            f"reference checkpoint {p} is safetensors; convert to a .pt state dict "
            "or add safetensors support before launch."
        )
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(p, map_location="cpu")
    if isinstance(obj, Mapping) and all(isinstance(v, Tensor) for v in obj.values()):
        return {str(k): v for k, v in obj.items()}
    if not isinstance(obj, Mapping):
        raise TypeError(f"reference checkpoint {p} is not a state-dict mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        inner = obj.get(key)
        if isinstance(inner, Mapping) and inner and all(
            isinstance(v, Tensor) for v in inner.values()
        ):
            return {str(k): v for k, v in inner.items()}
    # Some checkpoints nest tensors under a single prefix; accept if every value is a Tensor.
    if obj and all(isinstance(v, Tensor) for v in obj.values()):
        return {str(k): v for k, v in obj.items()}
    raise TypeError(
        f"Could not find a parameter state dict in {p}; expected a flat Tensor "
        "mapping or a dict with model / state_dict / model_state_dict."
    )


def per_token_ce(
    logits: Tensor,
    input_ids: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Causal next-token CE per position aligned to target token index.

    Returns float tensor ``[B, T]`` where position ``t`` is the loss for predicting
    ``input_ids[t]`` from context ``input_ids[:t]``. Position 0 is 0 (no target).
    """
    del ignore_index
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = loss_flat.view(shift_labels.shape)
    return F.pad(loss, (1, 0), value=0.0)


def per_token_z_loss(
    logits: Tensor,
    *,
    z_loss_multiplier: float,
) -> Tensor:
    """Per-target-position Z-loss aligned like :func:`per_token_ce`.

    Matches OLMo-core ``cross_entropy_loss``::

        z = λ · (logsumexp(logits))^2

    on the shifted (causal) logits, then pads so index ``t`` corresponds to the
    prediction of token ``t``. Position 0 is 0.
    """
    shift_logits = logits[:, :-1, :].float()
    z = float(z_loss_multiplier) * shift_logits.logsumexp(-1).pow(2)
    return F.pad(z, (1, 0), value=0.0)


def masked_ce_from_token_ce(token_ce: Tensor, label_mask: Tensor) -> Tuple[Tensor, int]:
    """Sum an already-computed per-token CE over the selected positions.

    Takes the ``[B, T]`` output of :func:`per_token_ce` rather than logits, so a step
    needs only *one* cross-entropy over the ``[B, T, vocab]`` logits instead of two.
    That matters at scale: a second cross-entropy allocates another log-softmax tensor
    the full size of the logits and keeps it alive for the backward pass.

    ``label_mask`` selects target positions, and :meth:`_valid_targets` always clears
    column 0 (nothing predicts the first token), so masking here is equivalent to
    building shifted labels with an ignore index. The sum accumulates in at least
    float32, since a bf16 sum over millions of positions loses too much precision;
    ``promote_types`` is used rather than a plain ``.float()`` so a float64 input is
    not silently narrowed.
    """
    mask = label_mask.to(dtype=torch.bool)
    accumulate_in = torch.promote_types(token_ce.dtype, torch.float32)
    loss = (token_ce.to(accumulate_in) * mask).sum()
    return loss, int(mask.sum().item())


def masked_z_from_token_z(token_z: Tensor, label_mask: Tensor) -> Tensor:
    """Sum per-token Z-loss over the same selected positions as CE."""
    mask = label_mask.to(dtype=torch.bool)
    accumulate_in = torch.promote_types(token_z.dtype, torch.float32)
    return (token_z.to(accumulate_in) * mask).sum()


def average_reference_state_dicts(
    state_dicts: Sequence[Mapping[str, Tensor]],
) -> Dict[str, Tensor]:
    """Element-wise mean of full parameter state dicts (late RefHQ average)."""
    if not state_dicts:
        raise ValueError("average_reference_state_dicts requires at least one state dict")
    first = state_dicts[0]
    keys = set(first)
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd) != keys:
            missing = sorted(keys - set(sd))[:8]
            extra = sorted(set(sd) - keys)[:8]
            raise KeyError(
                f"state dict {i} keys differ: missing={missing} extra={extra}"
            )
    out: Dict[str, Tensor] = {}
    n = float(len(state_dicts))
    for key, tensor0 in first.items():
        if not isinstance(tensor0, Tensor):
            raise TypeError(f"reference weight {key!r} must be a Tensor")
        if tensor0.is_floating_point():
            acc = tensor0.detach().to(dtype=torch.float32).clone()
            for sd in state_dicts[1:]:
                t = sd[key]
                if tuple(t.shape) != tuple(tensor0.shape):
                    raise ValueError(
                        f"shape mismatch for {key!r}: {tuple(t.shape)} vs "
                        f"{tuple(tensor0.shape)}"
                    )
                acc.add_(t.detach().to(dtype=torch.float32))
            out[key] = (acc / n).to(dtype=tensor0.dtype)
        else:
            out[key] = tensor0.detach().clone()
    return out


class TokenSelectState:
    """Mutable run state: step counter + REL EMA and/or frozen reference(s).

    The EMA accumulates from step 0, including through warmup, but it is debiased so the
    random initialization is excluded from the average (see :class:`EMAHistory`). Warmup
    controls when selection is *read*, not what the history contains.

    ``build_history_module`` controls how the REL history forward is served:
    - ``True`` (default, single-process / smoke): keep a deep-copied ``history_model``.
    - ``False`` (FSDP / OLMo-core): keep only the EMA shadow and run the history forward
      via :meth:`EMAHistory.swap_to` on the training model (no second full-size copy).

    RHO uses one :class:`FrozenReference`. Learnability uses two (early + late).
    Pass in-memory refs for smokes, or set the corresponding ``*_load_path`` fields.
    """

    def __init__(
        self,
        cfg: TokenSelectConfig,
        model: nn.Module,
        *,
        build_history_module: bool = True,
        frozen_ref: Optional[FrozenReference] = None,
        frozen_ref_early: Optional[FrozenReference] = None,
        frozen_ref_late: Optional[FrozenReference] = None,
    ):
        self.cfg = cfg
        self.step = 0
        self.tokens_seen = 0
        self.ema: Optional[EMAHistory] = None
        self.history_model: Optional[nn.Module] = None
        self.frozen_ref: Optional[FrozenReference] = None
        self.frozen_ref_early: Optional[FrozenReference] = None
        self.frozen_ref_late: Optional[FrozenReference] = None
        if cfg.uses_rel:
            seed_mode = str(cfg.ema_seed_mode or "zero").lower()
            if seed_mode == "refhq":
                if not cfg.reference_load_path:
                    raise ValueError(
                        "ema_seed_mode='refhq' requires TokenSelectConfig.reference_load_path "
                        "(exported RefHQ step1315 model.pt). This is the ONLY arm that seeds "
                        "EMA history from RefHQ; zero-init bias-corrected EMA must use "
                        "ema_seed_mode='zero'."
                    )
                weights = load_reference_state_dict(cfg.reference_load_path)
                self.ema = EMAHistory.from_module_seeded(
                    model, weights, alpha=cfg.alpha_start
                )
            elif seed_mode == "zero":
                # Bias-corrected empty accumulator (rel-ema-exp). Do not seed from RefHQ.
                self.ema = EMAHistory.from_module(model, alpha=cfg.alpha_start)
            else:
                raise ValueError(
                    f"ema_seed_mode={seed_mode!r} unsupported; expected 'zero' or 'refhq'"
                )
            if build_history_module:
                self.history_model = copy.deepcopy(model)
                self.history_model.eval()
                for p in self.history_model.parameters():
                    p.requires_grad_(False)
                if self.ema is not None and self.ema.has_history:
                    self.ema.copy_to(self.history_model)
        if cfg.uses_rho or cfg.uses_middle_ppl:
            if frozen_ref is not None:
                self.frozen_ref = frozen_ref
            elif cfg.reference_load_path:
                weights = load_reference_state_dict(cfg.reference_load_path)
                self.frozen_ref = FrozenReference.from_state_dict(model, weights)
            else:
                raise ValueError(
                    f"{cfg.method} requires frozen_ref=... or "
                    "TokenSelectConfig.reference_load_path"
                )
        if cfg.uses_learnability:
            if frozen_ref_early is not None:
                self.frozen_ref_early = frozen_ref_early
            elif cfg.early_reference_load_path:
                weights = load_reference_state_dict(cfg.early_reference_load_path)
                self.frozen_ref_early = FrozenReference.from_state_dict(model, weights)
            else:
                raise ValueError(
                    "learnability requires frozen_ref_early=... or "
                    "TokenSelectConfig.early_reference_load_path"
                )
            if frozen_ref_late is not None:
                self.frozen_ref_late = frozen_ref_late
            elif cfg.late_reference_load_path:
                weights = load_reference_state_dict(cfg.late_reference_load_path)
                self.frozen_ref_late = FrozenReference.from_state_dict(model, weights)
            else:
                raise ValueError(
                    "learnability requires frozen_ref_late=... or "
                    "TokenSelectConfig.late_reference_load_path"
                )

    def current_alpha(self) -> float:
        if not self.cfg.uses_rel:
            return 1.0
        return alpha_at_step(
            self.step,
            t0=self.cfg.t0_steps,
            total_steps=self.cfg.total_steps,
            alpha_start=self.cfg.alpha_start,
            alpha_end=self.cfg.alpha_end,
            schedule=self.cfg.alpha_schedule,
            tau=self.cfg.alpha_tau,
        )

    def in_warmup(self) -> bool:
        if not self.cfg.uses_selection:
            return False
        return self.step < self.cfg.t0_steps

    @torch.no_grad()
    def sync_history_model(self, model: nn.Module) -> None:
        if self.ema is None or self.history_model is None:
            return
        if self.ema.has_history:
            self.ema.copy_to(self.history_model)
            return
        # No optimizer step has landed yet, so the debiased average has no observed
        # weights. Score against the live model (REL = 0) rather than the random init.
        live = dict(model.named_parameters())
        for name, p in self.history_model.named_parameters():
            if (source := live.get(name)) is not None:
                p.copy_(source.detach())

    def after_optim_step(self, model: nn.Module) -> None:
        if self.cfg.uses_rel and self.ema is not None:
            a = self.current_alpha()
            self.ema.set_alpha(a)
            self.ema.update_module(model, alpha=a)
        self.step += 1

    def state_dict(self) -> Dict[str, Any]:
        """Return the state which must survive an OLMo trainer checkpoint."""
        state: Dict[str, Any] = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.frozen_ref is not None:
            state["frozen_ref"] = self.frozen_ref.state_dict()
        if self.frozen_ref_early is not None:
            state["frozen_ref_early"] = self.frozen_ref_early.state_dict()
        if self.frozen_ref_late is not None:
            state["frozen_ref_late"] = self.frozen_ref_late.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore selection bookkeeping, rejecting partial or incompatible states."""
        try:
            step = int(state["step"])
            tokens_seen = int(state.get("tokens_seen", 0))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("invalid token-selection checkpoint state") from e
        if step < 0 or tokens_seen < 0:
            raise ValueError("token-selection checkpoint counters must be non-negative")

        if self.ema is None:
            if "ema" in state:
                raise ValueError(
                    f"received EMA checkpoint state for method={self.cfg.method!r}"
                )
        else:
            ema_state = state.get("ema")
            if not isinstance(ema_state, Mapping):
                raise ValueError("REL checkpoint state is missing EMA weights")
            self.ema.load_state_dict(ema_state)

        if self.frozen_ref is None:
            if "frozen_ref" in state:
                raise ValueError(
                    f"received frozen-ref checkpoint state for method={self.cfg.method!r}"
                )
        else:
            ref_state = state.get("frozen_ref")
            if isinstance(ref_state, Mapping):
                self.frozen_ref.load_state_dict(ref_state)
            elif not self.cfg.reference_load_path:
                raise ValueError("RHO checkpoint state is missing frozen reference weights")

        if self.frozen_ref_early is None:
            if "frozen_ref_early" in state:
                raise ValueError(
                    f"received early-ref checkpoint state for method={self.cfg.method!r}"
                )
        else:
            early_state = state.get("frozen_ref_early")
            if isinstance(early_state, Mapping):
                self.frozen_ref_early.load_state_dict(early_state)
            elif not self.cfg.early_reference_load_path:
                raise ValueError(
                    "learnability checkpoint state is missing early frozen reference weights"
                )

        if self.frozen_ref_late is None:
            if "frozen_ref_late" in state:
                raise ValueError(
                    f"received late-ref checkpoint state for method={self.cfg.method!r}"
                )
        else:
            late_state = state.get("frozen_ref_late")
            if isinstance(late_state, Mapping):
                self.frozen_ref_late.load_state_dict(late_state)
            elif not self.cfg.late_reference_load_path:
                raise ValueError(
                    "learnability checkpoint state is missing late frozen reference weights"
                )

        self.step = step
        self.tokens_seen = tokens_seen
        if self.ema is not None:
            self.ema.set_alpha(self.current_alpha())


class TokenSelectLoop:
    """Minimal train step for smoke / unit use (all supported methods).

    Expects ``model(input_ids) -> logits [B,T,V]``.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TokenSelectConfig,
        *,
        frozen_ref: Optional[FrozenReference] = None,
        frozen_ref_early: Optional[FrozenReference] = None,
        frozen_ref_late: Optional[FrozenReference] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.state = TokenSelectState(
            cfg,
            model,
            frozen_ref=frozen_ref,
            frozen_ref_early=frozen_ref_early,
            frozen_ref_late=frozen_ref_late,
        )

    def train_step(self, input_ids: Tensor) -> Dict[str, Any]:
        cfg = self.cfg
        st = self.state
        warmup = st.in_warmup()
        alpha = st.current_alpha()
        rel_active = bool(cfg.uses_rel and not warmup)
        rho_active = bool(cfg.uses_rho and not warmup)
        middle_active = bool(cfg.uses_middle_ppl and not warmup)
        attn_active = bool(cfg.uses_attention and not warmup)
        learn_active = bool(cfg.uses_learnability and not warmup)
        random_active = bool(cfg.uses_random and not warmup)
        select_active = (
            rel_active
            or rho_active
            or middle_active
            or attn_active
            or learn_active
            or random_active
        )
        scoring_forward = rel_active or rho_active or middle_active or learn_active
        scoring_passes = (
            1 if (rel_active or rho_active or middle_active) else 0
        ) + (2 if learn_active else 0)

        valid = torch.ones_like(input_ids, dtype=torch.bool)
        valid[:, 0] = False

        current_loss: Optional[Tensor] = None
        history_loss: Optional[Tensor] = None
        reference_loss: Optional[Tensor] = None
        early_loss: Optional[Tensor] = None
        late_loss: Optional[Tensor] = None
        attention_score: Optional[Tensor] = None
        scoring_tokens = 0

        # Frozen-ref scoring must run *before* the grad-enabled forward: swap_to mutates
        # parameters in place, which would invalidate autograd if the train forward ran first.
        if rho_active or middle_active:
            assert st.frozen_ref is not None
            with torch.no_grad():
                with st.frozen_ref.swap_to(self.model):
                    logits_r = self.model(input_ids)
                    reference_loss = per_token_ce(logits_r, input_ids)
            scoring_tokens += int(input_ids.numel())
        elif learn_active:
            assert st.frozen_ref_early is not None and st.frozen_ref_late is not None
            with torch.no_grad():
                with st.frozen_ref_early.swap_to(self.model):
                    logits_e = self.model(input_ids)
                    early_loss = per_token_ce(logits_e, input_ids)
                with st.frozen_ref_late.swap_to(self.model):
                    logits_l = self.model(input_ids)
                    late_loss = per_token_ce(logits_l, input_ids)
            scoring_tokens += 2 * int(input_ids.numel())

        # Folded: ONE training forward (with grad) and ONE cross-entropy over its logits.
        # REL / rho_excess reuse that CE as the current-model score (detached).
        # attention_topk hooks last-layer attention input during this forward, then
        # recomputes Q/K (FlashAttention-safe) for the received-mass score.
        self.model.train()
        if attn_active:
            with capture_last_layer_attention_input(self.model) as attn_cap:
                logits = self.model(input_ids)
            attention_score = scores_from_capture(attn_cap)
        else:
            logits = self.model(input_ids)
        token_ce = per_token_ce(logits, input_ids)

        if rel_active:
            current_loss = token_ce.detach()
            st.sync_history_model(self.model)
            assert st.history_model is not None
            with torch.no_grad():
                logits_h = st.history_model(input_ids)
                history_loss = per_token_ce(logits_h, input_ids)
            scoring_tokens += int(input_ids.numel())
        elif rho_active:
            current_loss = token_ce.detach()

        mask_generator: Optional[torch.Generator] = None
        if random_active:
            mask_generator = torch.Generator(device=input_ids.device)
            mask_generator.manual_seed(cfg.seed + st.step * 1_000_003)

        label_mask = build_mask(
            method=cfg.method,
            k=cfg.k,
            current_loss=current_loss,
            history_loss=history_loss,
            reference_loss=reference_loss,
            early_loss=early_loss,
            late_loss=late_loss,
            attention_score=attention_score,
            shape_ref=input_ids,
            valid=valid,
            warmup=warmup,
            generator=mask_generator,
        )
        loss_sum, n_tok = masked_ce_from_token_ce(token_ce, label_mask)
        if n_tok == 0:
            label_mask = valid
            loss_sum, n_tok = masked_ce_from_token_ce(token_ce, label_mask)
        loss = loss_sum / max(n_tok, 1)

        selected_frac = float(label_mask.float().mean().item())
        mean_kept = mean_dropped = None
        score: Optional[Tensor] = None
        if rel_active and current_loss is not None and history_loss is not None:
            score = history_loss - current_loss
        elif rho_active and current_loss is not None and reference_loss is not None:
            score = current_loss - reference_loss
        elif middle_active and reference_loss is not None:
            score = reference_loss
        elif attn_active and attention_score is not None:
            score = attention_score
        elif learn_active and early_loss is not None and late_loss is not None:
            score = early_loss - late_loss
        if score is not None:
            kept = label_mask & valid
            dropped = (~label_mask) & valid
            if kept.any():
                mean_kept = float(score[kept].mean().item())
            if dropped.any():
                mean_dropped = float(score[dropped].mean().item())

        n_input = int(input_ids.numel())
        compute = {
            "selected_tokens": int(n_tok),
            "forward_tokens_train": n_input,
            "forward_tokens_history": n_input * scoring_passes if scoring_forward else 0,
            "forward_tokens_current": 0,  # folded into the training forward
            "fwd_passes_train": 1,
            "fwd_passes_history": scoring_passes,
            "fwd_passes_current": 0,
        }

        return {
            "loss": loss,
            "loss_sum": loss_sum,
            "n_tokens": n_tok,
            "label_mask": label_mask,
            "selected_frac": selected_frac,
            "mean_score_kept": mean_kept,
            "mean_score_dropped": mean_dropped,
            "alpha": alpha,
            "warmup": warmup,
            "scoring_tokens": scoring_tokens,
            "compute": compute,
            "step": st.step,
            "method": cfg.method,
        }

    def optim_step_done(self) -> None:
        self.state.after_optim_step(self.model)


try:
    from olmo_core.train.train_module.transformer.train_module import (  # type: ignore
        TransformerTrainModule,
    )
    from olmo_core.train import ReduceType  # type: ignore
    from olmo_core.data.utils import split_batch  # type: ignore

    try:
        from olmo_core.train.callbacks import Callback  # type: ignore
    except Exception:  # pragma: no cover
        Callback = object  # type: ignore

    _HAS_OLMO = True
except Exception:  # pragma: no cover
    TransformerTrainModule = object  # type: ignore
    ReduceType = None  # type: ignore
    split_batch = None  # type: ignore
    Callback = object  # type: ignore
    _HAS_OLMO = False


class RELCallback(Callback if _HAS_OLMO else object):  # type: ignore[misc]
    """Persist and advance REL's EMA exactly once for each optimizer step.

    OLMo-core checkpoints in ``CheckpointerCallback.post_train_batch()``, before its
    later ``post_step()`` callback phase. The callback therefore advances from
    ``post_train_batch()`` (priority above the checkpointer) so a just-saved checkpoint
    contains the matching model, EMA, and step. ``post_step()`` remains the public
    fallback hook and is idempotent for the current trainer step.

    Priority 3 (higher than the priority-2 raw-compute callback and the priority-1
    checkpointer) makes the intra-batch order explicit -- EMA advance, then raw compute,
    then checkpoint -- instead of relying on insertion order to break a priority tie.
    Advancing first means the metrics row and the checkpoint both describe the same step.
    """

    priority = 3

    def __init__(self) -> None:
        super().__init__()  # type: ignore[misc]
        self._last_advanced_trainer_step: Optional[int] = None
        self._pending_state: Optional[Mapping[str, Any]] = None

    def _train_module(self) -> Optional[Any]:
        trainer = getattr(self, "trainer", None)
        return getattr(trainer, "train_module", None)

    def _apply_pending_state(self) -> None:
        if self._pending_state is None:
            return
        train_module = self._train_module()
        if train_module is None or not hasattr(train_module, "load_token_selection_state"):
            return
        train_module.load_token_selection_state(self._pending_state)
        self._pending_state = None

    def _advance(self) -> None:  # pragma: no cover - requires olmo_core
        self._apply_pending_state()
        train_module = self._train_module()
        if train_module is None or not hasattr(train_module, "on_optim_step_end"):
            return
        trainer = getattr(self, "trainer", None)
        trainer_step = getattr(trainer, "global_step", None)
        if trainer_step is not None:
            trainer_step = int(trainer_step)
            if self._last_advanced_trainer_step == trainer_step:
                return
        train_module.on_optim_step_end()
        self._last_advanced_trainer_step = trainer_step

    def post_attach(self) -> None:  # pragma: no cover - requires olmo_core
        self._apply_pending_state()

    def post_train_batch(self) -> None:  # pragma: no cover - requires olmo_core
        self._advance()

    def post_step(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        del args, kwargs
        self._advance()

    def state_dict(self) -> Dict[str, Any]:
        self._apply_pending_state()
        train_module = self._train_module()
        state = None
        if train_module is not None and hasattr(train_module, "token_selection_state_dict"):
            state = train_module.token_selection_state_dict()
        return {
            "version": 1,
            "last_advanced_trainer_step": self._last_advanced_trainer_step,
            "token_selection": state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("version", 1) != 1:
            raise ValueError(f"unsupported REL callback state version: {state.get('version')!r}")
        last_step = state.get("last_advanced_trainer_step")
        self._last_advanced_trainer_step = None if last_step is None else int(last_step)
        token_selection = state.get("token_selection")
        if token_selection is None:
            return
        if not isinstance(token_selection, Mapping):
            raise ValueError("invalid REL callback token-selection state")
        self._pending_state = token_selection
        self._apply_pending_state()


# Rank-local sums a trainer callback reduces into one selection row per optimizer step.
# Everything here is additive so it survives an all-reduce; the derived means
# (mean REL kept/dropped, selected fraction, batch CE) are computed after reduction.
_EMPTY_SELECTION_DELTA: Dict[str, float] = {
    "rel_score_sum_kept": 0.0,
    "rel_score_sum_dropped": 0.0,
    "n_kept": 0.0,
    "n_dropped": 0.0,
    "n_valid": 0.0,
    "ce_loss_sum": 0.0,
    "n_batches": 0.0,
}


class TokenSelectTrainModule(TransformerTrainModule if _HAS_OLMO else object):  # type: ignore[misc]
    """OLMo-core train module for full-token, random keep-k, REL+EMA, RHO, middle-PPL,
    attention, and learnability.

    REL/RHO use a no-grad scoring forward (EMA history or frozen reference) plus one
    grad-enabled current forward whose logits are reused for both the current score and
    the selected-token CE. Learnability runs *two* frozen-ref scoring forwards
    (early + late) then the train forward. ``middle_ppl`` runs one frozen-ref scoring
    forward then the train forward. ``attention_topk`` hooks last-layer attention input during the
    train forward and recomputes Q/K for FlashAttention-safe received-mass scores.
    Intentionally rejects TP/CP and z-loss configurations: their public APIs do not
    expose unsharded per-token logits compatible with this scoring path.
    """

    def __init__(self, *args: Any, ts_config: Optional[TokenSelectConfig] = None, **kwargs: Any):
        if not _HAS_OLMO:
            raise ImportError(
                "olmo_core is required for TokenSelectTrainModule; "
                "use TokenSelectLoop for local smokes without OLMo-core"
            )
        self.ts_config = ts_config or TokenSelectConfig()
        super().__init__(*args, **kwargs)
        self._ts_state: Optional[TokenSelectState] = None
        self._compute_delta: Dict[str, int] = {
            "selected_tokens": 0,
            "forward_tokens_train": 0,
            "forward_tokens_history": 0,
            "forward_tokens_current": 0,
            "fwd_passes_train": 0,
            "fwd_passes_history": 0,
            "fwd_passes_current": 0,
        }
        self._selection_delta: Dict[str, float] = dict(_EMPTY_SELECTION_DELTA)

    def _ensure_state(self) -> TokenSelectState:
        if self._ts_state is None:
            # FSDP-friendly: no deep-copied history module; scoring forwards use
            # EMA/FrozenReference.swap_to on this model.
            self._ts_state = TokenSelectState(
                self.ts_config, self.model, build_history_module=False
            )
        return self._ts_state

    def on_optim_step_end(self) -> None:  # pragma: no cover - requires olmo_core
        """Called by :class:`RELCallback` once per optimizer step."""
        self._ensure_state().after_optim_step(self.model)

    def token_selection_state_dict(self) -> Optional[Dict[str, Any]]:
        if self._ts_state is None:
            return None
        return self._ts_state.state_dict()

    def load_token_selection_state(self, state: Mapping[str, Any]) -> None:
        self._ensure_state().load_state_dict(state)

    def consume_token_selection_compute_delta(self) -> Dict[str, int]:
        """Return rank-local raw work since the previous trainer callback."""
        delta = dict(self._compute_delta)
        for key in self._compute_delta:
            self._compute_delta[key] = 0
        return delta

    def consume_token_selection_selection_delta(self) -> Dict[str, float]:
        """Return rank-local REL selection sums since the previous trainer callback."""
        delta = dict(self._selection_delta)
        self._selection_delta = dict(_EMPTY_SELECTION_DELTA)
        return delta

    def token_selection_progress(self) -> Dict[str, Any]:
        """Schedule state for the current step; identical on every rank, so never reduced."""
        state = self._ensure_state()
        return {
            "step": int(state.step),
            "k": float(self.ts_config.k),
            "alpha": float(state.current_alpha()),
            "warmup": bool(state.in_warmup()),
            "method": str(self.ts_config.method),
        }

    def _assert_supported_execution(self) -> None:
        if self.tp_enabled or self.cp_enabled:
            raise RuntimeError(
                "TokenSelectTrainModule requires full, unsharded logits for REL scoring; "
                "the pinned OLMo-core API does not expose them with TP or CP enabled."
            )
        # z_loss_multiplier is supported: CE+z are folded over the same selection mask
        # (see train_batch). None disables z-loss (OLMo-core default).

    @staticmethod
    def _valid_targets(batch: Mapping[str, Any], input_ids: Tensor) -> Tensor:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
        if (label_mask := batch.get("label_mask")) is not None:
            valid &= label_mask.to(device=input_ids.device, dtype=torch.bool)
        if (attention_mask := batch.get("attention_mask")) is not None:
            valid &= attention_mask.to(device=input_ids.device) != 0
        if (instance_mask := batch.get("instance_mask")) is not None:
            valid &= instance_mask.to(device=input_ids.device, dtype=torch.bool).unsqueeze(-1)
        valid[:, 0] = False
        return valid

    @staticmethod
    def _selected_count(valid: Tensor, *, select_active: bool, k: float) -> int:
        if not select_active:
            return int(valid.sum().item())
        rows = valid.reshape(-1, valid.shape[-1])
        n_valid = rows.sum(dim=1)
        n_keep = torch.clamp((n_valid.to(torch.float32) * float(k)).round().long(), min=1)
        n_keep = torch.minimum(n_keep, n_valid)
        return int(n_keep.sum().item())

    @staticmethod
    def _model_kwargs(batch: Mapping[str, Any]) -> Dict[str, Any]:
        # label/attention/instance masks are consumed locally when selecting labels.
        # The target OLMo-core Transformer accepts document-layout fields such as
        # ``doc_lens`` and ``max_doc_lens`` through ``**kwargs``.
        excluded = {"input_ids", "labels", "label_mask", "attention_mask", "instance_mask"}
        return {key: value for key, value in batch.items() if key not in excluded}

    def _forward_logits(self, input_ids: Tensor, **model_kwargs: Any) -> Tensor:
        logits = self.model_forward(input_ids, **model_kwargs)
        if not isinstance(logits, Tensor):
            raise RuntimeError(
                "TokenSelectTrainModule expected logits from a labels=None OLMo forward; "
                f"received {type(logits).__name__}."
            )
        return logits

    @contextlib.contextmanager
    def _score_eval_mode(self) -> Iterator[None]:
        """Run a scoring forward with dropout disabled, then restore train mode exactly."""
        prior_training = self.model.training
        prior_mode = self._model_mode
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(prior_training)
            self._model_mode = prior_mode

    def train_batch(self, batch: Dict[str, Any], *args: Any, **kwargs: Any):  # pragma: no cover
        dry_run = bool(kwargs.pop("dry_run", False))
        if args or kwargs:
            raise TypeError("TokenSelectTrainModule.train_batch accepts only batch and dry_run")
        self._assert_supported_execution()
        if "labels" in batch:
            raise RuntimeError(
                "TokenSelectTrainModule generates labels from its online mask; "
                "precomputed batch['labels'] is not supported."
            )
        if split_batch is None:
            raise RuntimeError("olmo_core.data.utils.split_batch is unavailable")

        st = self._ensure_state()
        cfg = self.ts_config
        self._set_model_mode("train")
        input_ids: Tensor = batch["input_ids"]
        warmup = st.in_warmup()
        rel_active = bool(cfg.uses_rel and not warmup)
        rho_active = bool(cfg.uses_rho and not warmup)
        middle_active = bool(cfg.uses_middle_ppl and not warmup)
        attn_active = bool(cfg.uses_attention and not warmup)
        learn_active = bool(cfg.uses_learnability and not warmup)
        random_active = bool(cfg.uses_random and not warmup)
        select_active = (
            rel_active
            or rho_active
            or middle_active
            or attn_active
            or learn_active
            or random_active
        )
        scoring_passes = (
            1 if (rel_active or rho_active or middle_active) else 0
        ) + (2 if learn_active else 0)
        scoring_forward = scoring_passes > 0
        valid = self._valid_targets(batch, input_ids)
        selected_total = self._selected_count(valid, select_active=select_active, k=cfg.k)
        if selected_total == 0:
            raise RuntimeError("TokenSelectTrainModule received a batch with no valid target tokens")

        batch_num_tokens = input_ids.numel()
        loss_divisor = selected_total
        if (instance_mask := batch.get("instance_mask")) is not None:
            # Match OLMo-core's distributed loss-scaling convention for filtered rows.
            loss_divisor += int((~instance_mask.to(dtype=torch.bool)).sum().item()) * input_ids.shape[1]
        if loss_divisor <= 0:
            raise RuntimeError("TokenSelectTrainModule computed a zero loss divisor")
        self.record_metric(
            "masked labels (%)",
            (batch_num_tokens - selected_total) / batch_num_tokens,
            ReduceType.mean,
            namespace="train",
        )
        if instance_mask is not None:
            self.record_metric(
                "masked instances (%)",
                (~instance_mask.to(dtype=torch.bool)).float().mean(),
                ReduceType.mean,
                namespace="train",
            )

        if self.rank_microbatch_size < (seq_len := input_ids.shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to "
                f"sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)
        ce_batch_loss = torch.zeros((), device=self.device)
        z_batch_loss: Optional[Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = torch.zeros((), device=self.device)
        selected_seen = 0
        # Accumulated on-device so the score curves cost one host sync per batch
        # rather than one per micro-batch.
        score_sums = torch.zeros(4, device=self.device)

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                micro_input_ids = micro_batch["input_ids"].to(self.device)
                micro_valid = self._valid_targets(micro_batch, micro_input_ids)
                model_kwargs = self._model_kwargs(micro_batch)

                history_loss = None
                reference_loss = None
                early_loss = None
                late_loss = None
                attention_score = None
                if rel_active:
                    assert st.ema is not None
                    with self._score_eval_mode(), torch.no_grad(), st.ema.swap_to(self.model):
                        history_logits = self._forward_logits(micro_input_ids, **model_kwargs)
                        history_loss = per_token_ce(history_logits, micro_input_ids)
                    del history_logits
                    self.model.reset_auxiliary_metrics()
                elif rho_active or middle_active:
                    assert st.frozen_ref is not None
                    with self._score_eval_mode(), torch.no_grad(), st.frozen_ref.swap_to(self.model):
                        ref_logits = self._forward_logits(micro_input_ids, **model_kwargs)
                        reference_loss = per_token_ce(ref_logits, micro_input_ids)
                    del ref_logits
                    self.model.reset_auxiliary_metrics()
                elif learn_active:
                    assert st.frozen_ref_early is not None and st.frozen_ref_late is not None
                    with self._score_eval_mode(), torch.no_grad():
                        with st.frozen_ref_early.swap_to(self.model):
                            early_logits = self._forward_logits(micro_input_ids, **model_kwargs)
                            early_loss = per_token_ce(early_logits, micro_input_ids)
                        del early_logits
                        with st.frozen_ref_late.swap_to(self.model):
                            late_logits = self._forward_logits(micro_input_ids, **model_kwargs)
                            late_loss = per_token_ce(late_logits, micro_input_ids)
                        del late_logits
                    self.model.reset_auxiliary_metrics()

                if attn_active:
                    with capture_last_layer_attention_input(self.model) as attn_cap:
                        logits = self._forward_logits(micro_input_ids, **model_kwargs)
                    attention_score = scores_from_capture(attn_cap)
                else:
                    logits = self._forward_logits(micro_input_ids, **model_kwargs)
                token_ce = per_token_ce(logits, micro_input_ids)
                current_loss = (
                    token_ce.detach()
                    if (rel_active or rho_active)
                    else None
                )
                mask_generator: Optional[torch.Generator] = None
                if random_active:
                    mask_generator = torch.Generator(device=self.device)
                    mask_generator.manual_seed(
                        cfg.seed + int(st.step) * 1_000_003 + micro_batch_idx
                    )
                label_mask = build_mask(
                    method=cfg.method,
                    k=cfg.k,
                    current_loss=current_loss,
                    history_loss=history_loss,
                    reference_loss=reference_loss,
                    early_loss=early_loss,
                    late_loss=late_loss,
                    attention_score=attention_score,
                    shape_ref=micro_input_ids,
                    valid=micro_valid,
                    warmup=warmup,
                    generator=mask_generator,
                )
                loss_sum, n_tokens = masked_ce_from_token_ce(token_ce, label_mask)
                planned = self._selected_count(micro_valid, select_active=select_active, k=cfg.k)
                if n_tokens != planned:
                    raise RuntimeError(
                        "token-selection mask count diverged from the planned keep count; "
                        "refusing to apply a mis-scaled gradient"
                    )
                selected_seen += n_tokens
                score = None
                if rel_active and history_loss is not None and current_loss is not None:
                    score = history_loss - current_loss
                elif rho_active and reference_loss is not None and current_loss is not None:
                    score = current_loss - reference_loss
                elif middle_active and reference_loss is not None:
                    score = reference_loss
                elif attn_active and attention_score is not None:
                    score = attention_score
                elif learn_active and early_loss is not None and late_loss is not None:
                    score = early_loss - late_loss
                if score is not None:
                    kept = label_mask & micro_valid
                    dropped = (~label_mask) & micro_valid
                    zero = torch.zeros((), device=score.device, dtype=score.dtype)
                    score_sums += torch.stack(
                        [
                            torch.where(kept, score, zero).sum(),
                            torch.where(dropped, score, zero).sum(),
                            kept.sum().to(score.dtype),
                            dropped.sum().to(score.dtype),
                        ]
                    )
                # CE (+ optional Z) over the same selection mask; both scale by loss_divisor.
                ce_loss = loss_sum / loss_divisor
                loss = ce_loss
                if self.z_loss_multiplier is not None:
                    assert z_batch_loss is not None
                    z_sum = masked_z_from_token_z(
                        per_token_z_loss(logits, z_loss_multiplier=float(self.z_loss_multiplier)),
                        label_mask,
                    )
                    z_loss = z_sum / loss_divisor
                    loss = ce_loss + z_loss
                    z_batch_loss += z_loss.detach()
                    del z_sum, z_loss
                ce_batch_loss += ce_loss.detach()
                del logits, token_ce
                loss.backward()

        if selected_seen != selected_total:
            raise RuntimeError(
                "micro-batch token-selection counts diverged from the batch loss divisor"
            )
        self._compute_delta["selected_tokens"] += selected_seen
        self._compute_delta["forward_tokens_train"] += int(batch_num_tokens)
        self._compute_delta["fwd_passes_train"] += num_micro_batches
        if scoring_forward:
            self._compute_delta["forward_tokens_history"] += int(batch_num_tokens) * scoring_passes
            self._compute_delta["fwd_passes_history"] += num_micro_batches * scoring_passes
        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        score_kept, score_dropped, n_kept, n_dropped = score_sums.tolist()
        self._selection_delta["rel_score_sum_kept"] += score_kept
        self._selection_delta["rel_score_sum_dropped"] += score_dropped
        self._selection_delta["n_kept"] += n_kept
        self._selection_delta["n_dropped"] += n_dropped
        self._selection_delta["n_valid"] += float(valid.sum().item())
        self._selection_delta["ce_loss_sum"] += float(ce_batch_loss.item())
        self._selection_delta["n_batches"] += 1.0

        self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric("Z loss", z_batch_loss, ReduceType.mean, namespace="train")
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(metric_name, metric_val, reduction, namespace="train")


def has_olmo_core() -> bool:
    return _HAS_OLMO


def make_ts_config(
    cfg: Dict[str, Any],
    *,
    method: Literal[
        "full",
        "random",
        "rel_ema",
        "rho_excess",
        "middle_ppl",
        "attention_topk",
        "learnability",
    ],
    total_steps: Optional[int] = None,
    t0_steps: Optional[int] = None,
) -> TokenSelectConfig:
    """Build ``TokenSelectConfig`` from experiment YAML + derived steps."""
    if total_steps is None or t0_steps is None:
        from token_selection.scripts import derive_steps

        total_steps, t0_steps = derive_steps(cfg)
    uses_selection = method in (
        "random",
        "rel_ema",
        "rho_excess",
        "middle_ppl",
        "attention_topk",
        "learnability",
    )
    ref = cfg.get("reference") or {}
    ref_path = ref.get("load_path")
    early = ref.get("early") or {}
    late = ref.get("late") or {}
    early_path = early.get("load_path")
    late_path = late.get("load_path")
    ema_block = cfg.get("ema") or {}
    seed_mode = str(
        ema_block.get("seed_mode") or cfg.get("ema_seed_mode") or "zero"
    ).lower()
    if seed_mode not in ("zero", "refhq"):
        raise ValueError(
            f"ema.seed_mode / ema_seed_mode={seed_mode!r} unsupported; "
            "expected 'zero' (bias-corrected) or 'refhq' (rel-ema-refhq only)"
        )
    if method != "rel_ema" and seed_mode == "refhq":
        raise ValueError("ema_seed_mode='refhq' is only valid for method=rel_ema")
    schedule = str(
        ema_block.get("schedule") or cfg.get("alpha_schedule") or "linear"
    ).strip().lower()
    if schedule not in ("linear", "exp"):
        raise ValueError(
            f"ema.schedule / alpha_schedule={schedule!r} unsupported; "
            "expected 'linear' or 'exp'"
        )
    tau_raw = ema_block.get("tau")
    if tau_raw is None:
        tau_raw = cfg.get("alpha_tau", DEFAULT_ALPHA_TAU)
    tau = float(tau_raw)
    return TokenSelectConfig(
        method=method,
        k=float(cfg.get("k", 0.6)),
        t0_steps=int(t0_steps) if uses_selection else 0,
        total_steps=int(total_steps),
        alpha_start=float(cfg.get("alpha_start", 0.999)),
        alpha_end=float(cfg.get("alpha_end", 0.995)),
        alpha_schedule=schedule,
        alpha_tau=tau,
        seed=int(cfg.get("seed", 42)),
        reference_load_path=str(ref_path) if ref_path else None,
        early_reference_load_path=str(early_path) if early_path else None,
        late_reference_load_path=str(late_path) if late_path else None,
        ema_seed_mode=seed_mode,  # type: ignore[arg-type]
    )
