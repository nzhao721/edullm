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

## Execution environment

**Ephemeral runtime.** Training assumes job-scoped scratch that starts empty and is
wiped when the job ends. Train and curriculum bytes are staged from validated
`s3://edullm-data/` into a job-local cache; checkpoints, progress, metrics and
task-loss outputs stay on that scratch and upload to the Weights & Biases project
`curriculum`, so no run artifact is written back to S3. Local smoke runs pass
`--wandb-mode disabled --allow-local-only`. Every launch picks its recovery
explicitly: `FRESH=1` / `--fresh`, or `LOAD_PATH` / `--load-path` pointing at a
local step directory or a `wandb-artifact://entity/project/name:version` reference.

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

---

## Extension: MTLD pacing on OLMoE-1B-7B (cosine LR)

This section is an **extension** of the 370M dense curriculum study above. It keeps the MTLD easy→hard document order and the one-epoch ~10B token budget, but scales the test to **OLMoE-1B-7B** and switches the optimizer schedule to **cosine LR decay** (the same family used by the dense random-shuffle control). The goal is narrower than the original five-arm matrix: hold the difficulty metric fixed at MTLD and ablate **pacing** only, including two schedules that were not part of the dense 370M campaign (pure quadratic and warmup-quadratic).

Unlike the dense curriculum arms above, these runs do **not** use constant LR + late EMA. Finals are raw last-checkpoint macro task-loss, and uncertainty comes from residual-bootstrap power-law fits restricted to **step ≥ 1000** (matching the dense study’s late-window convention). Runs live in W&B `eduLLM/curriculum-moe` (not the dense `eduLLM/curriculum` project). Hyperparameters were **not** retuned; this is curriculum-only.

### Setup deltas vs the dense study

| Knob | Dense curriculum (above) | This extension |
|------|--------------------------|----------------|
| Architecture | OLMo-2 370M dense | OLMoE-1B-7B (\(d=2048\), 16 layers / 16 heads, 64 experts, top-\(k=8\), expert FFN 1024) |
| Parent corpus | Mixing Laws Dataset 10B | `pretrain/opt-with-synthetic-10b` (~10B tokens) |
| Difficulty order | Flesch / MTLD / Learnability | **MTLD only** (`curriculum/opt-with-synthetic-10b` MTLD token order) |
| LR schedule | Constant after 24-step warmup + late EMA | **Cosine** after 24-step warmup (peak \(4\times10^{-4}\)); raw final checkpoint |
| Global batch / seq | 4,194,304 / 2048 | Same |
| Steps / seed | 2384 / (dense campaign) | 2384 / 42 |
| Search / HPO | None | None |

### Pacing arms

| Arm | Pacing | Manipulation |
|-----|--------|--------------|
| Stock control | Shuffled / full-pool | Flat shuffle over the parent corpus (no MTLD order) |
| Linear + MTLD | `linear_n10` | Same 10 equal-mass easy→hard staircase as the dense Linear (\(n=10\)) arm |
| Quadratic + MTLD | `quadratic_n10` | Same MTLD buckets, but segment lengths grow with weights \(1\ldots10\) (largest-remainder to 2384) |
| Warmup + MTLD | `warmup_1000` | Naive sequential easy→hard for steps 0–999, then full-pool shuffle |
| Warmup-quadratic + MTLD | `warmup_quadratic_n10_1000` | Quadratic MTLD warmup over the first 1000 steps, then full-pool shuffle for the remainder |

All four curriculum arms share the same immutable MTLD order, model, token budget, seed, peak LR, and 4 Mi-token global batch. Warmup-quadratic used a 32 Ki rank microbatch; the others used 16 Ki. That changes accumulation geometry only, not the nominal optimizer batch.

### Evaluation and uncertainty

1. Fit \(y = a + b/\mathrm{step}^{\alpha}\) on finite `eval/macro_bpb` checkpoints with **step ≥ 1000 and ≤ 2384** (\(n=12\) points per curve; \(\alpha \in [0.001, 3]\), \(a,b \ge 0\)).
2. Residual bootstrap (5k for endpoint CIs; 10k for null \(p\)-values): center residuals, resample with replacement, refit, and take percentile intervals at step 2384.
3. Pairwise \(\Delta = \mathrm{arm} - \mathrm{control}\) from independent bootstraps of both curves. Negative \(\Delta\) favors the curriculum arm.
4. One-sided and two-sided \(p\)-values use 10k residual resamples under a constrained null in which the two compared curves share the same fitted endpoint. One-sided \(p\) is the probability of a fitted difference at least as favorable to the first-named arm.

**Warmup-quadratic step-2384 note.** W&B history for that run logs the final macro at `_step=2385`, but the durable eval artifact is `eval-step0002384` (`step2384_task_loss.json`) with the same macro **1.5139**. There is no separate EMA eval. Analyses below treat that point as the raw step-2384 final.

**Caveats.** These intervals quantify trajectory-fit uncertainty for a **single seed**, not seed-to-seed variance. Restricting to step ≥ 1000 matches the dense curriculum CI window and drops early-training noise, but leaves only 12 points per curve. The cosine schedule also means this extension is **not** a pure scale-up of the constant-LR dense curriculum protocol.

### Results

#### Raw and fitted final macro task-loss (bpb)

| Arm | Observed final | Fitted final [95% CI] | Observed \(\Delta\) | Fitted \(\Delta\) [95% CI] | \(p_{\mathrm{one}}\) | \(p_{\mathrm{two}}\) |
|-----|---------------:|-----------------------|--------------------:|----------------------------|---------------------:|---------------------:|
| Stock control | 1.5288 | 1.5206 [1.5063, 1.5335] | — | — | — | — |
| Linear + MTLD | 1.5605 | 1.5511 [1.5402, 1.5606] | +0.0317 | +0.0305 [+0.0135, +0.0484] | 0.998 | 0.0050 |
| Quadratic + MTLD | 1.5895 | 1.5816 [1.5661, 1.6055] | +0.0606 | +0.0609 [+0.0416, +0.0886] | 1.000 | 0.0002 |
| Warmup + MTLD | 1.5364 | 1.5376 [1.5244, 1.5488] | +0.0075 | +0.0170 [−0.0013, +0.0352] | 0.954 | 0.091 |
| **Warmup-quadratic + MTLD** | **1.5139** | **1.5105** [1.5018, 1.5190] | **−0.0149** | **−0.0101** [−0.0263, +0.0061] | **0.113** | **0.226** |

Lower is better. \(\Delta\) is curriculum minus control. One-sided \(p\) favors the curriculum arm (smaller fitted endpoint); values near 1 mean the arm is worse than control.

#### Pacing ablation: warmup-quadratic vs warmup (fixed MTLD order)

Both arms share the same MTLD ranking and the same post-warmup full-pool shuffle phase; the only manipulation is whether the first 1000 steps use **naive sequential** easy→hard (`warmup_1000`) or a **quadratic decile staircase** (`warmup_quadratic_n10_1000`).

| Comparison | Observed \(\Delta\) | Fitted \(\Delta\) | 95% \(\Delta\) CI | \(p_{\mathrm{one}}\) | \(p_{\mathrm{two}}\) |
|------------|--------------------:|------------------:|-------------------|---------------------:|---------------------:|
| Warmup-quadratic − Warmup | **−0.0224** | **−0.0271** | **[−0.0420, −0.0112]** | **0.0002** | **0.0004** |

\(\Delta\) is warmup-quadratic minus warmup; negative favors the quadratic warmup. Same ≥1000-step residual-bootstrap / equal-endpoint-null construction (5k CI, 10k \(p\)).

**Reading.** Under the late-window fit, quadratic warmup beats naive warmup on both the raw endpoint (−0.022 bpb) and the fitted trajectory (−0.027 bpb), with a 95% CI entirely below zero and \(p_{\mathrm{two}}=0.0004\). That is the cleanest evidence in this extension that **quadratic pacing inside the warmup phase helps** relative to a naive sequential warmup, holding MTLD order and stock HPs fixed.

### Takeaways

1. **Warmup-quadratic + MTLD is the only schedule that beats control** on both raw (−0.0149 bpb) and fitted (−0.0101) endpoints, but the control comparison is still not significant at \(\alpha=0.05\) (\(p_{\mathrm{one}}=0.113\), \(p_{\mathrm{two}}=0.226\); CI crosses zero).
2. **Versus naive warmup, quadratic warmup is a clear win** under the ≥1000-step fit: fitted \(\Delta = -0.0271\), 95% CI [−0.0420, −0.0112], \(p_{\mathrm{two}}=0.0004\).
3. **Warmup + MTLD does not beat control** once early steps are excluded from the fit (fitted \(\Delta = +0.0170\), \(p_{\mathrm{two}}=0.091\)).
4. **Pure quadratic pacing is clearly harmful**: fitted \(\Delta = +0.0609\) with CI entirely above zero (\(p_{\mathrm{two}}=0.0002\)).
5. **Linear + MTLD also fails to transfer**: significantly worse than control under the late-window fit (\(p_{\mathrm{two}}=0.005\)).
6. **Pacing remains schedule-sensitive at MoE scale.** Ranking on late fitted endpoints: warmup-quadratic ≻ control ≳ warmup ≻ linear ≫ quadratic. The decisive positive claim is the matched warmup-quadratic vs warmup ablation, not the control comparison.

### Extension conclusions

Scaling MTLD curricula to OLMoE-1B-7B under cosine LR decay does **not** reproduce the dense Linear + MTLD win: linear and pure quadratic pacing are significantly worse than shuffle. The best stock-HP MoE schedule is still **warmup-quadratic MTLD**, which improves on control without conventional significance, but **does** significantly beat naive warmup under the same late-window residual bootstrap (\(p_{\mathrm{two}}=0.0004\)). Together with the dense study, the practical reading is: MTLD ordering can help, the useful pacing is architecture- and LR-schedule-dependent, and at MoE scale a quadratic warmup prefix is preferable to a naive sequential warmup when HPs are held fixed.
