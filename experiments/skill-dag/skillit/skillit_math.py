#!/usr/bin/env python3
"""Skill-It weight update and adjacency construction helpers.

Offline A (probe arm): loaded from ``artifacts/A_offline.npy``.
Online A (derivative arm): from the parametric mixing law

    L_j(r) = c_j + k_j * exp( sum_i t_ij * r_i )

with Skill-It-compatible adjacency

    A_ij = max( 0, -t_ij * (L_j(r) - c_j) )
         = max( 0, -(dL_j / dr_i) )

so that A_ij > 0 means domain i helps task family j.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

ETA_DEFAULT = 0.2

# Defaults match mixlaw_common; kept local so this module is importable without
# mixlaw on sys.path (tests). Callers may pass explicit domain/family orders.
DEFAULT_DOMAINS: tuple[str, ...] = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "algebraic-stack",
    "wiki",
)
DEFAULT_FAMILIES: tuple[str, ...] = (
    "arc_challenge",
    "arc_easy",
    "mmlu_humanities",
    "mmlu_other",
    "mmlu_social_sciences",
    "mmlu_stem",
)

def softmax_weights(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax → simplex vector."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    s = float(e.sum())
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError(f"softmax denominator invalid: {s}")
    return e / s


def skillit_update(
    A: np.ndarray,
    losses: Union[np.ndarray, Sequence[float]],
    *,
    eta: float = ETA_DEFAULT,
    w: float = 1.0,
) -> np.ndarray:
    """Skill-It domain-weight update (eta, w=1).

    ``p_i ∝ exp( eta * w * sum_j A_ij * L_j )``
    """
    A_arr = np.asarray(A, dtype=np.float64)
    L = np.asarray(losses, dtype=np.float64).reshape(-1)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A_arr.shape}")
    if A_arr.shape[1] != L.shape[0]:
        raise ValueError(f"A columns {A_arr.shape[1]} != len(losses) {L.shape[0]}")
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    scores = (A_arr @ L) * float(eta) * float(w)
    return softmax_weights(scores)


def load_offline_A(
    path: Union[str, Path],
    *,
    n_domains: int = 7,
    n_families: int = 6,
) -> np.ndarray:
    """Load offline adjacency matrix from ``.npy`` (shape n_domains × n_families)."""
    A = np.load(Path(path))
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (n_domains, n_families):
        raise ValueError(f"expected A shape {(n_domains, n_families)}, got {A.shape}")
    return A


def predict_family_loss(
    target: Mapping[str, object],
    r: Mapping[str, float] | np.ndarray,
    domains: Sequence[str] = DEFAULT_DOMAINS,
) -> float:
    """Evaluate L_j(r) = c + k * exp(sum_i t_i r_i) for one mixing-law target."""
    c = float(target["c"])  # type: ignore[arg-type]
    k = float(target["k"])  # type: ignore[arg-type]
    t_map = target["t"]  # type: ignore[assignment]
    assert isinstance(t_map, Mapping)
    if isinstance(r, np.ndarray):
        r_vec = {d: float(r[i]) for i, d in enumerate(domains)}
    else:
        r_vec = {d: float(r[d]) for d in domains}
    expo = sum(float(t_map[d]) * r_vec[d] for d in domains)
    return c + k * float(np.exp(expo))


def online_A_from_fit(
    fit: Mapping[str, object],
    r: Mapping[str, float] | np.ndarray | Sequence[float],
    *,
    domains: Sequence[str] = DEFAULT_DOMAINS,
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> np.ndarray:
    """Build A(r) from ``mixlaw_fit_chinchilla.json`` at mixture weights ``r``.

    ``A_ij = max(0, -t_ij * (L_j(r) - c_j))``.
    """
    targets = fit["targets"]
    assert isinstance(targets, Mapping)
    if isinstance(r, (list, tuple, np.ndarray)) and not isinstance(r, Mapping):
        r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
        if r_arr.shape[0] != len(domains):
            raise ValueError(f"r length {r_arr.shape[0]} != n_domains {len(domains)}")
        r_map = {d: float(r_arr[i]) for i, d in enumerate(domains)}
    else:
        r_map = {d: float(r[d]) for d in domains}  # type: ignore[index]

    A = np.zeros((len(domains), len(families)), dtype=np.float64)
    for j, fam in enumerate(families):
        if fam not in targets:
            raise KeyError(f"family {fam!r} missing from fit targets")
        tgt = targets[fam]
        assert isinstance(tgt, Mapping)
        L_j = predict_family_loss(tgt, r_map, domains=domains)
        c_j = float(tgt["c"])  # type: ignore[arg-type]
        delta = L_j - c_j
        t_map = tgt["t"]
        assert isinstance(t_map, Mapping)
        for i, dom in enumerate(domains):
            t_ij = float(t_map[dom])
            A[i, j] = max(0.0, -t_ij * delta)
    return A


def load_fit_json(path: Union[str, Path]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def regmix_weight_vector(
    domains: Sequence[str] = DEFAULT_DOMAINS,
) -> np.ndarray:
    """RegMix base weights in ``domains`` order (sums to ~1)."""
    base = {
        "dclm": 0.375,
        "arxiv": 0.25,
        "starcoder": 0.1406,
        "pes2o": 0.0938,
        "open-web-math": 0.0635,
        "algebraic-stack": 0.0615,
        "wiki": 0.0156,
    }
    w = np.array([base[d] for d in domains], dtype=np.float64)
    return w / w.sum()


def A_to_named_dict(
    A: np.ndarray,
    *,
    domains: Sequence[str] = DEFAULT_DOMAINS,
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> dict:
    """Serialize A with named rows/cols for JSON snapshots."""
    A_arr = np.asarray(A, dtype=np.float64)
    return {
        "domain_order": list(domains),
        "family_order": list(families),
        "A": A_arr.tolist(),
        "rows_by_domain": {
            d: {f: float(A_arr[i, j]) for j, f in enumerate(families)}
            for i, d in enumerate(domains)
        },
    }


def losses_dict_to_vector(
    losses: Mapping[str, float],
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> np.ndarray:
    missing = [f for f in families if f not in losses]
    if missing:
        raise KeyError(f"missing family losses: {missing}")
    return np.array([float(losses[f]) for f in families], dtype=np.float64)


def optional_mixlaw_paths() -> Optional[Path]:
    """Sibling mixlaw dir when skillit lives next to mixlaw."""
    here = Path(__file__).resolve().parent
    cand = here.parent / "mixlaw"
    return cand if cand.is_dir() else None
