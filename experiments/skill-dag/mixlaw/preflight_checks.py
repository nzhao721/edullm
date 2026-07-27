"""Local sanity checks for the mixlaw scripts. No GPU, no AWS, no data required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from build_mixture_data import _blocks_for_domain  # noqa: E402
from fit_mixing_law import _predict, fit_mixing_law, fit_step_law, loo_cv, optimize_simplex  # noqa: E402
from mixlaw_common import (
    CURVE_TASK_LOSS_LABELS,
    DEFAULT_TOKENS_PER_PARAM,
    DOMAIN_AVAILABLE_TOKENS,
    DOMAINS,
    LADDER_TASK_LOSS_LABELS,
    SEQ_LEN,
    allocate_sequences,
    load_mixtures,
    realized_weights,
    token_budget,
)

print("=" * 70)
print("1. mixtures.json + sequence allocation")
print("=" * 70)

TPP = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOKENS_PER_PARAM

mixtures = load_mixtures()
print(f"loaded {len(mixtures)} mixtures")
for tpp in (3.0, 6.0, 10.0, 20.0, 100.0):
    total_seqs, total_steps, total_tokens = token_budget(tpp)
    print(f"  tpp={tpp:>5}: {total_tokens/1e6:8.1f}M tokens  {total_steps:>6,} steps  {total_seqs:>7,} seqs")

total_seqs, total_steps, total_tokens = token_budget(TPP)
worst = 0.0
for mix in mixtures:
    counts = allocate_sequences(mix.weights, total_seqs)
    assert sum(counts.values()) == total_seqs, f"mix {mix.id} allocation does not sum"
    rw = realized_weights(counts)
    for d in DOMAINS:
        if mix.weights[d] == 0.0:
            assert counts[d] == 0, f"mix {mix.id}: zero-weight {d} got {counts[d]} seqs"
    worst = max(worst, max(abs(rw[d] - mix.weights[d]) for d in DOMAINS))
print(f"  all allocations exact; worst realized-vs-target weight error {worst:.3e}")

print()
print("=" * 70)
print(f"2. per-domain token demand at tokens/param={TPP:g} vs olmohq pool")
print("=" * 70)
peak = {d: 0 for d in DOMAINS}
total_demand = {d: 0 for d in DOMAINS}
for mix in mixtures:
    counts = allocate_sequences(mix.weights, total_seqs)
    for d in DOMAINS:
        peak[d] = max(peak[d], counts[d] * SEQ_LEN)
        total_demand[d] += counts[d] * SEQ_LEN
print(f"{'domain':<18}{'available':>14}{'peak 1 mix':>14}{'sum 24 mixes':>15}  status")
infeasible = []
for d in DOMAINS:
    avail = DOMAIN_AVAILABLE_TOKENS[d]
    ok = "OK" if peak[d] <= avail else "EXCEEDS CORPUS"
    if peak[d] > avail:
        infeasible.append(d)
    reuse = "" if total_demand[d] <= avail else " (cross-mix overlap)"
    print(f"{d:<18}{avail/1e9:>11.1f}B{peak[d]/1e6:>12.1f}M{total_demand[d]/1e6:>13.1f}M  {ok}{reuse}")
assert not infeasible, f"tokens/param={TPP:g} is infeasible for {infeasible}; see budget_calculator.py"

print()
print("=" * 70)
print("3. random block planner")
print("=" * 70)
rng = np.random.default_rng(0)
for need, avail, block in ((10_000, 1_800_000, 256), (5, 1_000_000, 256), (100, 300, 256)):
    blocks = _blocks_for_domain(rng, avail, need, block)
    got = sum(n for _, n in blocks)
    starts = [s for s, _ in blocks]
    assert got == need, f"planned {got} != {need}"
    assert starts == sorted(starts), "blocks not in sequential read order"
    covered = set()
    for s, n in blocks:
        rows = set(range(s, s + n))
        assert not (covered & rows), "blocks overlap: a sequence would repeat"
        covered |= rows
    print(f"  need={need:>7} avail={avail:>9} -> {len(blocks):>4} blocks, exact, non-overlapping")

print()
print("=" * 70)
print("4. task-loss label sets")
print("=" * 70)
print(f"  full ladder bpb suite: {len(LADDER_TASK_LOSS_LABELS)} labels")
print(f"  in-run curve subset:   {len(CURVE_TASK_LOSS_LABELS)} labels")
assert set(CURVE_TASK_LOSS_LABELS) <= set(LADDER_TASK_LOSS_LABELS)

print()
print("=" * 70)
print("5. mixing-law recovery on synthetic data (can 24 points identify 9 params?)")
print("=" * 70)
R = np.array([[m.weights[d] for d in DOMAINS] for m in mixtures])
rng = np.random.default_rng(7)
true_theta = np.concatenate([[np.log(0.55)], [np.log(0.35)], rng.normal(0, 0.8, len(DOMAINS))])
y_clean = _predict(true_theta, R)
print(f"  synthetic task loss range: {y_clean.min():.4f} .. {y_clean.max():.4f}")

for noise in (0.0, 0.002, 0.010):
    y = y_clean + rng.normal(0, noise, len(y_clean)) if noise else y_clean.copy()
    theta, rmse = fit_mixing_law(R, y, n_starts=96, seed=1)
    cv = loo_cv(R, y, n_starts=48, delta=1e-3, seed=1)
    print(
        f"  noise sd={noise:.3f}: fit RMSE={rmse:.5f}  LOO RMSE={cv['rmse']:.5f}  "
        f"({cv['rmse']/max(cv['target_std'],1e-9):.1%} of target std)"
    )

print()
print("=" * 70)
print("6. simplex optimization under availability caps")
print("=" * 70)
theta, _ = fit_mixing_law(R, y_clean, n_starts=96, seed=1)
caps = [1.0] * len(DOMAINS)
caps[DOMAINS.index("wiki")] = 0.122
r_star, val = optimize_simplex(
    lambda r: float(_predict(theta, r[None, :])[0]),
    len(DOMAINS),
    caps,
    [0.0] * len(DOMAINS),
    n_starts=128,
    seed=1,
)
assert abs(r_star.sum() - 1.0) < 1e-6, "optimum is off the simplex"
assert r_star[DOMAINS.index("wiki")] <= 0.122 + 1e-6, "wiki cap violated"
print(f"  optimum on simplex (sum={r_star.sum():.6f}), wiki={r_star[DOMAINS.index('wiki')]:.4f} <= 0.122")
print(f"  predicted loss {val:.5f} vs best synthetic observation {y_clean.min():.5f}")

print()
print("=" * 70)
print("7. step law recovery")
print("=" * 70)
steps = np.linspace(72, 1440, 20)
truth = 0.62 + 4.0 * steps ** (-0.45)
law = fit_step_law(steps, truth + rng.normal(0, 0.0005, len(steps)), seed=3)
print(f"  true  L_inf=0.6200 A=4.0000 alpha=0.4500")
print(f"  fitted L_inf={law['L_inf']:.4f} A={law['A']:.4f} alpha={law['alpha']:.4f} rmse={law['rmse']:.6f}")

print()
print("ALL CHECKS PASSED")
