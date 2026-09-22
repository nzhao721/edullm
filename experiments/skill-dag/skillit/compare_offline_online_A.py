#!/usr/bin/env python3
"""Print the offline (probe) and online (derivative) adjacency matrices side by side.

The online matrix is a *snapshot*: ``A_ij = max(0, -t_ij (L_j(r) - c_j))`` depends
on the mixture ``r`` it is evaluated at, and the derivative arm recomputes it at
the current weights on every update. Two reference points matter here and they do
not give the same matrix:

  * ``r_DML``  -- the Data Mixing Laws paper mixture (``mix01``). This is the
    reference the *offline* probe matrix is built against
    (``A_ij = max(0, L_j(r_DML) - L_j(i))``), so it is the like-for-like point
    for comparing the two constructions. The published Figure IV plots this one.
  * ``LGB-min1pct`` -- the LightGBM-optimized mixture the derivative arm actually
    starts from, i.e. the matrix used at its first update (step 500).

Evaluated at ``r_DML`` the online matrix is roughly twice the magnitude it has at
``LGB-min1pct`` (e.g. dclm -> arc_challenge 0.154 vs 0.065), and its Pearson
correlation with the offline matrix is 0.171 rather than 0.065. Edge *presence*
is unchanged: 13 of 42 nonzero either way, and 10 of 42 cells disagree with the
offline matrix either way. Report both so a reader can tell which number goes
with which panel.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mixlaw"), str(ROOT / "skillit")]
from mixlaw_common import CURVE_FAMILIES, DOMAINS  # noqa: E402
from skillit_math import (  # noqa: E402
    default_mixlaw_fit_path,
    load_fit_json,
    online_A_from_fit,
    regmix_weight_vector,
)

LGB_MIN1PCT = {
    "dclm": 0.5529,
    "arxiv": 0.2118,
    "starcoder": 0.0872,
    "pes2o": 0.0816,
    "open-web-math": 0.0418,
    "algebraic-stack": 0.0136,
    "wiki": 0.0111,
}


def _describe(A: np.ndarray, reference: np.ndarray, label: str) -> None:
    nonzero = int((A > 0).sum())
    disagree = int(((A > 0) != (reference > 0)).sum())
    r = float(np.corrcoef(A.ravel(), reference.ravel())[0, 1])
    print(
        f"{label}: {nonzero}/{A.size} nonzero ({nonzero / A.size:.1%} dense), "
        f"Pearson r vs offline = {r:.3f}, {disagree}/{A.size} cells disagree on edge presence"
    )


def main() -> None:
    fit = load_fit_json(default_mixlaw_fit_path())
    A_off = np.load(Path(__file__).parent / "artifacts/probes_full/A_offline.npy")

    points = {
        "ONLINE @ r_DML (mix01; the point Figure IV plots)": regmix_weight_vector(DOMAINS),
        "ONLINE @ LGB-min1pct (the arm's own starting mixture)": np.array(
            [LGB_MIN1PCT[d] for d in DOMAINS], dtype=np.float64
        ),
    }

    matrices = [("OFFLINE (probe)", A_off)]
    matrices += [(name, online_A_from_fit(fit, r)) for name, r in points.items()]

    width = max(len(d) for d in DOMAINS)
    for name, A in matrices:
        print(f"\n{name}")
        print(" " * width, *[f"{f[:9]:>9s}" for f in CURVE_FAMILIES])
        for i, d in enumerate(DOMAINS):
            print(f"{d:<{width}}", *[f"{A[i, j]:9.4f}" for j in range(len(CURVE_FAMILIES))])

    print()
    for name, A in matrices[1:]:
        _describe(A, A_off, name)


if __name__ == "__main__":
    main()
