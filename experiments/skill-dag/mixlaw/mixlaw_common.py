#!/usr/bin/env python3
"""Shared constants and helpers for the 24-mixture DataDecide-60M mixing-law probe.

Single source of truth for:
  * the exact DataDecide 60M geometry / optimizer / batch schedule,
  * the 7 RegMix domains and the S3 layout of the tokenized 10B corpus,
  * how a mixture weight vector becomes an exact per-domain *sequence* count,
  * the OLMo-ladder task-loss (bits-per-byte) evaluation set.

Mixture proportions are realized purely through **token counts on disk**: OLMo's
``MemMapDataset`` concatenates every path into one index of non-overlapping
``seq_len`` chunks and the sampler shuffles that index, so one epoch over a set of
per-domain slices reproduces the mixture exactly with no repeats and no
importance weighting. Everything downstream therefore depends on allocating
sequences per domain precisely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_TORCH_LOAD_PATCHED = False


def patch_torch_load_for_olmo_checkpoints() -> None:
    """PyTorch >=2.6 defaults torch.load(weights_only=True); OLMo checkpoints need False."""
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    import torch

    _orig_torch_load = torch.load

    def _torch_load_trusted(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_trusted  # type: ignore[assignment]
    try:
        import pathlib as _pathlib

        _safe = [_pathlib.Path, _pathlib.PosixPath, _pathlib.WindowsPath]
        try:
            from pathlib import _local as _pathlib_local  # type: ignore[attr-defined]

            _safe.extend(
                [
                    getattr(_pathlib_local, "Path", None),
                    getattr(_pathlib_local, "PosixPath", None),
                    getattr(_pathlib_local, "WindowsPath", None),
                ]
            )
        except Exception:
            pass
        torch.serialization.add_safe_globals([x for x in _safe if x is not None])
    except Exception:
        pass
    _TORCH_LOAD_PATCHED = True

# --- DataDecide 60M, exactly as published -----------------------------------
# allenai/DataDecide-dolma1_7-60M/config.json + DataDecide Appendix Table 2:
#   batch 96 seq | hidden 384 | LR 5.8e-3 | 57.1M non-embedding | 12 heads
#   16 layers | 29,042 steps | 5.7B tokens (token/param ratio 100)
D_MODEL = 384
N_HEADS = 12
N_LAYERS = 16
MLP_RATIO = 8
SEQ_LEN = 2048
GLOBAL_BATCH_SEQS = 96
LEARNING_RATE = 5.8e-3

# OLMo counts "model size" as every parameter except ``wte`` (so the untied LM
# head is included). With DataDecide's 50,304-row embedding that is exactly
# 57,078,144, which reproduces the published 5.7B token budget at ratio 100.
DATADECIDE_MODEL_SIZE = 57_078_144
# Transformer blocks + final norm only, i.e. independent of vocabulary size.
BODY_PARAMS = 37_761_408

TOKENS_PER_STEP = GLOBAL_BATCH_SEQS * SEQ_LEN  # 196,608

# --- Tokenizer / vocabulary --------------------------------------------------
# The RegMix corpus is already tokenized with dolma2, so the embedding matrix is
# resized to dolma2 while every shape that defines the DataDecide 60M *body*
# (d_model, layers, heads, mlp_ratio, seq_len) is left untouched.
TOKENIZER_ID = "allenai/dolma2-tokenizer"
VOCAB_SIZE = 100_278
EMBEDDING_SIZE = 100_352
EOS_TOKEN_ID = 100_257
PAD_TOKEN_ID = 100_277
MEMMAP_DTYPE = "uint32"
BYTES_PER_TOKEN = 4

# --- OLMoHQ 30B pool (edullm-datasets/olmo100b) ---------------------------------
# Raw document shards live under olmo-mix-1124-30b/data/<domain>/*.json.gz.
# There is no pre-tokenized copy on this bucket; prepare_data.sh builds a
# *working pool* of uint32 memmaps sized to the peak per-domain demand of the
# 24 mixtures (not the full ~95B-token pool), then build_mixture_data.py draws
# random sequence-aligned blocks from those memmaps.
DOMAINS: tuple[str, ...] = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)

OLMOHQ_S3 = "s3://edullm-datasets/olmo100b/olmo-mix-1124-30b"
OLMOHQ_DATA_PREFIX = "data"
# Local layout after prepare: tokenized/<domain>/<domain>.npy
TOKENIZED_PREFIX = "tokenized"

# Tokens available per domain in the olmohq upsample (user inventory / plan summary).
# These bound r_max for a 30B target; they are *not* the per-run training budget.
DOMAIN_AVAILABLE_TOKENS: dict[str, int] = {
    "dclm": 28_600_000_000,
    "arxiv": 20_800_000_000,
    "starcoder": 20_300_000_000,
    "pes2o": 26_300_000_000,
    "open-web-math": 12_200_000_000,
    "algebraic-stack": 11_800_000_000,
    "wiki": 3_660_000_000,
}

# RegMix base weights (used only as the Alg. 2 "base" mixture, not as pool sizes).
DOMAIN_BASE_WEIGHTS: dict[str, float] = {
    "dclm": 0.375,
    "arxiv": 0.25,
    "starcoder": 0.1406,
    "pes2o": 0.0938,
    "open-web-math": 0.0635,
    "algebraic-stack": 0.0615,
    "wiki": 0.0156,
}

# Natural token fractions in allenai/olmo-mix-1124 (HF card / tech report).
_OLMO_MIX_1124_DOMAIN_TOKENS: dict[str, float] = {
    "dclm": 3.70e12,
    "arxiv": 20.8e9,
    "starcoder": 83.0e9,
    "pes2o": 58.6e9,
    "open-web-math": 12.2e9,
    "algebraic-stack": 11.8e9,
    "wiki": 3.66e9,
}
_OLMO_MIX_1124_TOTAL = sum(_OLMO_MIX_1124_DOMAIN_TOKENS.values())
OLMO_MIX_1124_WEIGHTS: dict[str, float] = {
    d: _OLMO_MIX_1124_DOMAIN_TOKENS[d] / _OLMO_MIX_1124_TOTAL for d in DOMAINS
}

# --- Compute budget (≈12 B200 GPU-hours for all 24 mixtures) -----------------
# 12 GPU-hours / 24 mixes ≈ 30 min/mix including final task-loss eval. Default
# tokens/param=5 (~285M tokens, 1451 steps) targets that envelope at ~200k tok/s
# (measured on GPU7 smoke). Data is *not* the limit: the olmohq pool supports
# ~30B tokens/mix (wiki binds).
TOTAL_GPU_HOURS = 12.0
DEFAULT_TOKENS_PER_PARAM = 5.0

# Backward-compatible alias used by older call sites / preflight.
DOMAIN_TARGET_TOKENS = DOMAIN_AVAILABLE_TOKENS

MIXTURES_JSON = Path(__file__).with_name("mixtures.json")

# --- OLMo-ladder task loss ---------------------------------------------------
# The fitting target is *task loss*: bits-per-byte of the gold continuation on the
# OLMES 5-shot RC suite. Bhagia et al. (arXiv:2412.04403) fit power laws in this
# metric because it is smooth and byte-normalized (so tokenizer-independent),
# whereas accuracy is not fittable at small scale.
#
# These are exactly the ``_bpb`` labels OLMo-ladder selects out of
# ``olmo.eval.downstream.label_to_task_map_new`` (its filter drops ``_train_``,
# ``_mc_`` and ``_var``), so each one resolves to a bundled ``OEEvalTask`` with
# ``metric_type="bpb"`` and needs no network access at eval time.
LADDER_TASK_LOSS_LABELS: tuple[str, ...] = (
    "arc_challenge_val_rc_5shot_bpb",
    "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
)

# In-run curve subset. A full pass over all 20 labels costs more forward tokens
# than a short probe run costs training tokens, so the loss *curve* (used for the
# step-law extrapolation) is measured on the cheapest reliable labels and the
# full suite is run once at the end for the mixing-law fit itself. DataDecide
# found ARC and MMLU give usable small-scale signal most cheaply. Only these six
# families are used for mixing-law targets (Chinchilla-extrapolated from curves).
CURVE_TASK_LOSS_LABELS: tuple[str, ...] = (
    "arc_challenge_val_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
)


def task_family(label: str) -> str:
    """Collapse a ladder label to its task family (drops split and metric suffix)."""
    stem = label.removesuffix("_bpb")
    for split in ("_val_", "_test_"):
        if split in stem:
            return stem.split(split)[0]
    return stem


# Task families with in-run loss curves (one val label each). Mixing-law fits use
# only these six; Chinchilla targets come from step-law extrapolation on the curves.
CURVE_FAMILIES: tuple[str, ...] = tuple(
    sorted({task_family(label) for label in CURVE_TASK_LOSS_LABELS})
)


def macro_curve(task_loss_families: dict[str, float]) -> float:
    """Mean task loss over the six curve families."""
    return sum(float(task_loss_families[f]) for f in CURVE_FAMILIES) / len(CURVE_FAMILIES)


def normalize_eval_key(key: str) -> Optional[str]:
    """Recover the ladder label from an OLMo metric key, or None if it is not a bpb metric.

    ``Evaluator.compute_metrics`` builds ``eval/downstream_bpb/{label}_{metric_type}``,
    and because every task-loss label already ends in ``_bpb`` the emitted key ends in
    ``_bpb_bpb``. Normalizing here keeps the in-run curve keyed by the same labels the
    final evaluation writes, so ``fit_mixing_law.py`` does not need two spellings.
    """
    if not key.startswith("eval/"):
        return None
    tail = key.rsplit("/", 1)[-1]
    if not tail.endswith("_bpb_bpb"):
        return None
    label = tail.removesuffix("_bpb")
    return label if label in LADDER_TASK_LOSS_LABELS else None


@dataclass(frozen=True)
class Mixture:
    id: int
    tag: str
    weights: dict[str, float]
    name: str | None = None

    @property
    def run_name(self) -> str:
        return self.name or f"mix{self.id:02d}"


def load_mixtures(path: Path | None = None) -> list[Mixture]:
    """Load mixtures.json, validating that each row is a proper simplex point."""
    payload = json.loads((path or MIXTURES_JSON).read_text(encoding="utf-8"))
    order = list(payload["domain_order"])
    if tuple(order) != DOMAINS:
        raise SystemExit(f"mixtures.json domain_order {order} != {list(DOMAINS)}")

    out: list[Mixture] = []
    for row in payload["mixtures"]:
        weights = [float(w) for w in row["weights"]]
        if len(weights) != len(order):
            raise SystemExit(f"mixture {row['id']}: expected {len(order)} weights")
        if any(w < 0 for w in weights):
            raise SystemExit(f"mixture {row['id']}: negative weight")
        total = sum(weights)
        if abs(total - 1.0) > 2e-3:
            raise SystemExit(f"mixture {row['id']}: weights sum to {total:.6f}, not 1")
        # Renormalize away the rounding in the published 4-decimal table.
        out.append(
            Mixture(
                id=int(row["id"]),
                tag=str(row["tag"]),
                weights={d: w / total for d, w in zip(order, weights)},
                name=str(row["run_name"]) if row.get("run_name") else None,
            )
        )

    ids = [m.id for m in out]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate mixture ids")
    return out


def token_budget(tokens_per_param: float) -> tuple[int, int, int]:
    """Return (total_sequences, total_steps, total_tokens) for a token/param ratio.

    Sequences are rounded *down* to a whole number of optimizer steps so every
    run is exactly one epoch over its slices with no partial final batch.
    """
    if tokens_per_param <= 0:
        raise SystemExit("tokens_per_param must be > 0")
    raw_seqs = int(tokens_per_param * DATADECIDE_MODEL_SIZE) // SEQ_LEN
    total_steps = raw_seqs // GLOBAL_BATCH_SEQS
    if total_steps < 1:
        raise SystemExit(f"tokens_per_param={tokens_per_param} yields 0 steps")
    total_seqs = total_steps * GLOBAL_BATCH_SEQS
    return total_seqs, total_steps, total_seqs * SEQ_LEN


def token_budget_fixed(total_tokens: int) -> tuple[int, int, int]:
    """Return (total_sequences, approx_370m_steps, total_tokens) for a fixed token budget.

    Sequences are rounded down to ``SEQ_LEN``. The step count assumes the 370M
    global batch (4_194_304 tokens) used by the RegMix control trainer.
    """
    if total_tokens <= 0:
        raise SystemExit("total_tokens must be > 0")
    total_seqs = int(total_tokens) // SEQ_LEN
    if total_seqs < 1:
        raise SystemExit(f"total_tokens={total_tokens} yields 0 sequences")
    realized = total_seqs * SEQ_LEN
    gbs = 4_194_304
    total_steps = realized // gbs
    return total_seqs, total_steps, realized


def allocate_sequences(weights: dict[str, float], total_seqs: int) -> dict[str, int]:
    """Largest-remainder split of ``total_seqs`` across domains.

    Guarantees the counts sum to ``total_seqs`` exactly, so the realized mixture
    is the closest achievable point to ``weights`` at sequence granularity. A
    domain with weight 0 always gets 0 sequences (deliberate ablations stay
    exact ablations).
    """
    if total_seqs <= 0:
        raise SystemExit("total_seqs must be > 0")

    raw = {d: weights.get(d, 0.0) * total_seqs for d in DOMAINS}
    counts = {d: int(raw[d]) for d in DOMAINS}
    remainder = total_seqs - sum(counts.values())

    # Hand out leftover sequences to the largest fractional parts, skipping
    # domains the mixture excludes entirely.
    eligible = [d for d in DOMAINS if weights.get(d, 0.0) > 0.0]
    order = sorted(eligible, key=lambda d: (raw[d] - int(raw[d]), raw[d]), reverse=True)
    i = 0
    while remainder > 0 and order:
        counts[order[i % len(order)]] += 1
        remainder -= 1
        i += 1
    return counts


def realized_weights(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        raise SystemExit("empty sequence allocation")
    return {d: counts.get(d, 0) / total for d in DOMAINS}


def domain_npy_name(domain: str) -> str:
    return f"{domain}.npy"


def memmap_tokens(path: Path) -> int:
    """Token count of a raw uint32 memmap (these files have no NumPy header)."""
    return path.stat().st_size // BYTES_PER_TOKEN


def ladder_warmup_steps() -> int:
    """OLMo-ladder heuristic: warm up over roughly one model-size worth of tokens."""
    return round(DATADECIDE_MODEL_SIZE / TOKENS_PER_STEP)  # 290


def peak_domain_tokens(tokens_per_param: float) -> dict[str, int]:
    """Largest per-domain token demand across the 24 mixtures at this budget."""
    total_seqs, _, _ = token_budget(tokens_per_param)
    peak = {d: 0 for d in DOMAINS}
    for mix in load_mixtures():
        counts = allocate_sequences(mix.weights, total_seqs)
        for d in DOMAINS:
            peak[d] = max(peak[d], counts[d] * SEQ_LEN)
    return peak


def max_data_feasible_tokens() -> tuple[int, str, float]:
    """Largest one-mix token budget the olmohq pool can supply without repeats.

    Returns (max_tokens, binding_domain, max_weight_of_that_domain).
    """
    mixes = load_mixtures()
    best = None
    for d in DOMAINS:
        w = max(m.weights[d] for m in mixes)
        if w <= 0:
            continue
        cap = int(DOMAIN_AVAILABLE_TOKENS[d] / w)
        if best is None or cap < best[0]:
            best = (cap, d, w)
    assert best is not None
    return best


def gpu_hours_for_budget(
    tokens_per_param: float,
    *,
    tok_per_sec: float = 150_000.0,
    eval_minutes_per_mix: float = 12.0,
    n_mixes: int = 24,
) -> dict[str, float]:
    """Estimate GPU-hours for the full pilot at a given tokens/param.

    ``tok_per_sec`` is a B200 estimate for this 60M body; measure it on a smoke
    run and override. Eval time is the final 20-label task-loss pass.
    """
    _, _, tokens = token_budget(tokens_per_param)
    train_sec = tokens / max(tok_per_sec, 1.0)
    eval_sec = eval_minutes_per_mix * 60.0
    per_mix_hours = (train_sec + eval_sec) / 3600.0
    return {
        "tokens_per_mix": float(tokens),
        "train_minutes": train_sec / 60.0,
        "eval_minutes": eval_minutes_per_mix,
        "hours_per_mix": per_mix_hours,
        "total_gpu_hours": per_mix_hours * n_mixes,
        "wall_hours_at_n_gpus": {
            str(n): per_mix_hours * n_mixes / n for n in (1, 2, 4, 8, 16, 24)
        },
    }