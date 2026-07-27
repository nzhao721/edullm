"""Budget calculator: 12 GPU-hour envelope on B200s vs olmohq data limits.

Default tokens/param=5 is compute-limited (set after GPU7 smoke at ~206k tok/s).
The olmohq pool supports ~30B tokens per mixture (wiki binds); DataDecide's
published 5.7B (tpp=100) is data-feasible but would burn ~100+ GPU-hours for 24 mixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mixlaw_common import (
    BODY_PARAMS,
    D_MODEL,
    DATADECIDE_MODEL_SIZE,
    DEFAULT_TOKENS_PER_PARAM,
    DOMAIN_AVAILABLE_TOKENS,
    DOMAINS,
    EMBEDDING_SIZE,
    N_LAYERS,
    SEQ_LEN,
    TOTAL_GPU_HOURS,
    gpu_hours_for_budget,
    load_mixtures,
    max_data_feasible_tokens,
    token_budget,
)

B200_BF16_FLOPS = 2.25e15

mixes = load_mixtures()
cap, bind_d, bind_w = max_data_feasible_tokens()

print("Data ceiling (olmohq pool, no within-mix repeats)")
print(f"  binding: {bind_d} at weight {bind_w:.4f} -> max {cap/1e9:.2f}B tokens/mix")
print(f"  = tokens/param {cap/DATADECIDE_MODEL_SIZE:.1f}")
print()
print(f"{'domain':<18}{'available':>12}{'max weight':>12}{'max budget':>13}")
for d in DOMAINS:
    top = max(mixes, key=lambda m: m.weights[d])
    w = top.weights[d]
    if w <= 0:
        continue
    print(
        f"{d:<18}{DOMAIN_AVAILABLE_TOKENS[d]/1e9:>10.1f}B"
        f"{w:>12.4f}{DOMAIN_AVAILABLE_TOKENS[d]/w/1e9:>12.2f}B"
    )

print()
print("=" * 72)
print(f"Compute envelope: {TOTAL_GPU_HOURS:g} B200 GPU-hours for all 24 mixtures")
print("=" * 72)

head_params = D_MODEL * EMBEDDING_SIZE
gflop_per_token = (
    6 * (BODY_PARAMS + head_params) + 3 * 4 * N_LAYERS * SEQ_LEN * D_MODEL
) / 1e9
print(f"FLOPs/token (fwd+bwd): {gflop_per_token:.3f} GFLOP")
print()

# Realistic small-model throughput band on B200.
for label, tps in (("pessimistic 80k tok/s", 80_000), ("default 150k tok/s", 150_000), ("optimistic 250k tok/s", 250_000)):
    print(f"--- {label} + 12 min final eval/mix ---")
    hdr = (
        f"{'tok/param':>9}{'tokens':>10}{'steps':>8}"
        f"{'GPU-h':>8}{'wall@8':>8}{'wall@24':>9}{'fits 12h':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for tpp in (2, 3, 5, 6, 8, 10, 20, 100):
        est = gpu_hours_for_budget(float(tpp), tok_per_sec=float(tps), eval_minutes_per_mix=12.0)
        _, steps, toks = token_budget(float(tpp))
        ok = "YES" if est["total_gpu_hours"] <= TOTAL_GPU_HOURS * 1.15 else "no"
        mark = " <-- default" if tpp == DEFAULT_TOKENS_PER_PARAM else ""
        print(
            f"{tpp:>9}{toks/1e6:>9.0f}M{steps:>8,}"
            f"{est['total_gpu_hours']:>8.1f}"
            f"{est['wall_hours_at_n_gpus']['8']:>8.1f}"
            f"{est['wall_hours_at_n_gpus']['24']:>9.1f}"
            f"{ok:>10}{mark}"
        )
    print()

print("Parallelization:")
print("  bash run_mixture.sh <mix_id> <gpu_index>   # one mixture per GPU")
print("  # Use Slurm array or your scheduler to fan out mix IDs 1-24")
print()
print("Wall clock ~= 12 / N hours when all 24 mixtures run at the default budget.")
