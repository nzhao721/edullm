# Curriculum learning (Mixing Laws Dataset 10B × OLMo-2 370M)

**Question.** Under a fixed one-epoch token budget, does ordering Mixing Laws Dataset documents by difficulty (with paced exposure) improve macro task-loss relative to a **random shuffle** CE baseline?

**Answer.** Partially. All five curriculum arms beat **random shuffle** on EMA-centered task-loss. Linear + MTLD, Linear + Flesch, and Interleaved + Flesch do so with two-sided residual-bootstrap \(p \le 0.011\); Learnability and Warmup also finish below control on EMA but are not significant (\(p = 0.21\) and \(0.42\)). Gains are smaller than MixLaw mixture search, and the comparison crosses LR schedules (curriculum uses constant LR + EMA; random shuffle uses cosine).

---

## Setup

| Knob | Value |
|------|-------|
| Architecture | OLMo-2 370M (full attention): \(d=1024\), 16 layers / 16 heads, gated SiLU FFN 4096, RoPE \(\theta=5\times10^5\), vocab 100,352 |
| Train corpus | Mixing Laws Dataset 10B (~9.989B tokens) |
| Global batch / seq | 4,194,304 / 2048 |
| LR | Peak \(4\times10^{-4}\), warmup 24, then **constant** (\(\alpha_f=1.0\)) |
| Steps | 2384 ≈ one epoch |
| FLOPs / full arm | \(2.63\times10^{19}\) (measured from W&B) |
| Final estimate | Post-hoc EMA over checkpoints at 2000, 2125, 2250, 2384 (\(\alpha=0.8\)) |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels |

All curriculum arms use the **same tokens** as the flat corpus. What changes is (1) a **fixed easy→hard ranking** of documents under a difficulty metric, and (2) a **pacing schedule** that decides which ranks are eligible (or active) at each training step. The comparison control is **random shuffle**: flat-shuffle CE on the same corpus (cosine LR, no EMA).

Training is partitioned on a **250-step segment grid** aligned with checkpoints: boundaries at 0, 250, 500, …, 2250, 2384 (final segment 134 steps).

### Difficulty metrics

Documents are sorted once into an easy→hard order. “Easier” means:

| Metric | Sort sense | Intuition | Paper |
|--------|------------|-----------|-------|
| Flesch | Higher Flesch reading-ease first | Surface readability / shorter words and sentences first | Difficulty signal ablated in [Zhang et al., Beyond Random Sampling](https://arxiv.org/abs/2506.11300) |
| MTLD | Lower MTLD first | Lower lexical diversity first; more repetitive / simpler vocabulary earlier | Lexical-diversity signal in [Zhang et al.](https://arxiv.org/abs/2506.11300) |
| Learnability | Lower early→late NLL improvement under a frozen **RefHQ** reference first | Documents RefHQ found “already easy” (small late−early loss drop) come earlier | [Irreducible Curriculum](https://arxiv.org/abs/2310.15389) |

**RefHQ training data (learnability labels only).** Learnability scores come from a frozen OLMo-2 370M CE model (**RefHQ**) trained on an HQ-filtered ~5.5B-token corpus in the same 7-domain Mixing Laws proportions (scaled to that budget; ~5.509B published train tokens). Domain pulls are high-quality filtered sources, not the full Mixing Laws Dataset 10B stream:

| Domain | Approx. target tokens | Source character |
|--------|----------------------:|------------------|
| dclm | ~2.07B | DataDecide DCLM-Baseline QC (FineWeb-2 7% recipe stream) |
| arxiv | ~1.38B | arXiv papers |
| starcoder | ~0.78B | StarCoder data with Dolma code-HQ filtering |
| pes2o | ~0.52B | peS2o scholarly text |
| open-web-math | ~0.35B | OpenWebMath (HQ filter; pool binds the mix) |
| algebraic-stack | ~0.34B | AlgebraicStack |
| wiki | ~0.09B | Wikipedia |

Tokenizer: dolma2. Reference training FLOPs \(\approx 1.45\times10^{19}\). Early/late NLLs for learnability use RefHQ checkpoints at step 250 (early) and the mean of steps 1000 / 1125 / 1315 (late).

### Pacing schedules

Pacing schedules follow [Zhang et al., Beyond Random Sampling](https://arxiv.org/abs/2506.11300). Constant LR + late EMA follows [Luo et al., Learning Rate Decay Wastes Your Best Data](https://arxiv.org/abs/2511.18903).

**Linear (\(n=10\)).** Split the easy→hard ranking into **10 equal-mass difficulty buckets**. Training walks the buckets **in order, non-overlapping**: segment 1 trains only on the easiest tenth, segment 2 only on the next tenth, …, segment 10 only on the hardest tenth. Within a segment the active bucket is shuffled. The model therefore never revisits earlier (easier) material once it has moved on — a strict curriculum staircase over the one-epoch budget ([Zhang et al.](https://arxiv.org/abs/2506.11300)).

**Warmup (1000 steps).** This is **naive sequential** easy→hard — not Linear (\(n=10\)) buckets. For steps 0–999, consume the ranked corpus as a **single non-overlapping sequential pass** in easy→hard document order (at the global batch rate). From step 1000 onward, sample **uniformly from the full corpus** (ordinary shuffle). Manipulation: a long naive easy-first prefix, then abandon ordering for the rest of training ([Zhang et al.](https://arxiv.org/abs/2506.11300)).

**Interleaved (\(i=10\)).** Split training into **10 outer segments** (same 250-step grid). **Inside each segment**, replay a full Linear (\(n=10\)) mini-curriculum: walk easy→hard sub-buckets again before the next outer segment. Manipulation: the model repeatedly restarts the easy→hard progression instead of seeing each difficulty band only once ([Zhang et al.](https://arxiv.org/abs/2506.11300)).

### Arms actually run

| Arm | What is manipulated | A100-h | FLOPs |
|-----|---------------------|-------:|------:|
| Linear + Flesch | Linear (\(n=10\)) over Flesch order | 50.81 | \(2.63\times10^{19}\) |
| Linear + MTLD | Linear (\(n=10\)) over MTLD order | 43.97 | \(2.63\times10^{19}\) |
| Warmup + Flesch | Naive sequential easy→hard for 1000 steps (Flesch), then shuffle | 43.84 | \(2.63\times10^{19}\) |
| Interleaved + Flesch | Interleaved (\(i=10\)) over Flesch order | 47.74 | \(2.63\times10^{19}\) |
| Linear + Learnability | Linear (\(n=10\)) over learnability order | 43.84 | \(2.63\times10^{19}\) |
| **Total** | | **230.20** | **\(1.32\times10^{20}\)** |

Comparisons use the **random shuffle** full run (cosine LR, raw final — no EMA). A100-hours are the no-waste totals measured for these runs.

---

## Evaluation and uncertainty

Curriculum runs hold **constant LR** after LR warmup so that late, harder data is not deweighted by cosine decay ([Luo et al.](https://arxiv.org/abs/2511.18903)). With a flat LR, the last raw checkpoint is a noisy snapshot of an undamped optimizer trajectory. Instead of reporting that single checkpoint, the scientific final is a **post-hoc EMA** of late checkpoints (steps 2000, 2125, 2250, 2384; \(\alpha=0.8\)): averaging damps late-training noise while keeping the constant-LR curriculum protocol intact. The random-shuffle control already stabilizes via cosine LR decay, so its final is the raw last-checkpoint eval.

CI construction (EMA-centered residual bootstrap):

1. Fit \(y = a + b/\mathrm{step}^{\alpha}\) on raw checkpoint evals with step ≥ 1000 through 2384 (\(\alpha \in [0.05, 3]\)).
2. Residual bootstrap (10k) for spread at step 2384.
3. Center the CI on the **EMA** value. For random shuffle (no EMA), center on raw step 2384.
4. Pairwise $\Delta = \mathrm{arm} - \mathrm{control}$ from independent bootstraps; two-sided $p$ reported for every arm whose EMA (or raw final) beats control.

**Caveat on the power-law fit.** Curriculum ordering can produce loss curves that do not follow the usual smooth power-law decline seen under shuffled data: difficulty pacing changes what the model sees when, so late-training dynamics can be non-monotonic or otherwise atypical. The \(a + b/\mathrm{step}^{\alpha}\) model is still used for CI spread (and is clearly weak for some arms, e.g. Linear + Learnability), but fitted shape and residual-bootstrap widths may be less reliable here than for mixture or Skill-It runs. Prefer the EMA centers for ranking; treat the intervals as approximate.

---

## Results

### EMA-centered macro task-loss (bpb)

| Arm | Center | 95% CI | vs random shuffle |
|-----|-------:|--------|-----------------|
| **Linear + MTLD** | **1.6095** (EMA) | [1.5988, 1.6202] | $p_{\mathrm{two}} < 0.0001$ |
| **Linear + Flesch** | **1.6273** (EMA) | [1.6216, 1.6329] | $p_{\mathrm{two}} < 0.0001$ |
| **Interleaved + Flesch** | **1.6384** (EMA) | [1.6303, 1.6465] | $p_{\mathrm{two}} = 0.011$ |
| Linear + Learnability | 1.6451 (EMA) | [1.6317, 1.6586] | $p_{\mathrm{two}} = 0.21$ (weak fit; \(\alpha\) hits 3.0) |
| Warmup + Flesch | 1.6474 (EMA) | [1.6380, 1.6567] | $p_{\mathrm{two}} = 0.42$ |
| Random shuffle (control) | 1.6518 (raw 2384) | [1.6483, 1.6552] | — |

Lower is better. Every curriculum arm’s EMA center is below random shuffle; only the top three are significant at conventional levels.

### Pairwise comparisons (same EMA-centered residual bootstrap)

Δ = arm A − arm B (negative ⇒ A better). Two-sided \(p\) from independent bootstraps.

**Pacing (fixed Flesch order)**

| Comparison | Δ EMA | 95% Δ CI | \(p_{\mathrm{two}}\) |
|------------|------:|----------|---------------------|
| Linear + Flesch vs Interleaved + Flesch | −0.0111 | [−0.0222, −0.0022] | 0.012 |
| Linear + Flesch vs Warmup + Flesch | −0.0201 | [−0.0312, −0.0096] | 0.0002 |

Linear pacing beats both other Flesch schedules.

**Difficulty metric (fixed Linear \(n=10\) pacing)**

| Comparison | Δ EMA | 95% Δ CI | \(p_{\mathrm{two}}\) |
|------------|------:|----------|---------------------|
| Linear + MTLD vs Linear + Flesch | −0.0178 | [−0.0303, −0.0060] | 0.002 |
| Linear + MTLD vs Linear + Learnability | −0.0356 | [−0.0516, −0.0166] | \(< 0.0001\) |

MTLD beats both other metrics under linear pacing.

### Takeaways

1. **MTLD + linear pacing is the best curriculum arm** (~0.042 bpb below random shuffle on EMA center) — a single easy→hard pass with lexical-diversity ordering.
2. **Flesch + linear and interleaved pacing also beat random shuffle significantly** (\(p < 10^{-4}\) and \(p = 0.011\)). Learnability and Warmup (naive sequential easy→hard for 1000 steps, then shuffle) beat on EMA center but not significantly (\(p = 0.21\), \(0.42\)).
3. **Linear pacing beats the other Flesch schedules** (vs interleaved \(p = 0.012\); vs warmup \(p = 0.0002\)).
4. **MTLD beats the other linear-paced metrics** (vs Flesch \(p = 0.002\); vs learnability \(p < 10^{-4}\)).
5. **Cross-lever caveat.** Curriculum uses constant LR + EMA; random shuffle uses cosine without EMA. Absolute gaps are informative but not a pure pacing A/B.
6. **Cost.** Five arms 230.20 A100-hours and \(\approx 1.32\times10^{20}\) FLOPs.

---

## Conclusions

Difficulty-ordered curricula can help at this scale, especially **a single linear easy→hard pass with MTLD**. That combination significantly outperforms both alternate Flesch pacings and both alternate linear-paced metrics. Interleaving still beats random shuffle; warmup (naive sequential easy→hard prefix) and learnability improve the EMA point estimate without clear significance vs random shuffle.
