# Token selection (Mixing Laws Dataset 10B × OLMo-2 370M)

**Question.** Under a matched one-epoch Mixing Laws Dataset budget, can selecting a subset of tokens per sequence beat full-token CE on macro task-loss?

**Answer.** No. Every selection arm finished **worse** than the full-CE baseline. Excess-loss ρ-1 was the least harmful: it does beat a random-60% keep-rate control (−0.0075 bpb, two-sided bootstrap **p = 0.048**, measured against a two-seed fit of that control), but it still loses to full CE by 0.0106 bpb. Middle-perplexity and relative-EMA selection collapsed performance.

---

## Setup


| Knob                         | Value                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Architecture                 | OLMo-2 370M (full attention): d=1024, 16L/16H, FFN 4096, RoPE \theta=5\times10^5, vocab 100,352 |
| Train corpus                 | `pretrain/regmix-10b` **v1** — realized **10,004,807,041** tokens, flat shuffle; identical dataset for all seven arms |
| Global batch / seq / LR      | 4,194,304 / 2048 / 4\times10^{-4} cosine (warmup 24, \alpha_f=0.1)                              |
| Steps                        | 2360 = 9,898,557,440 tokens for every arm — one epoch, no wrap |
| FLOPs / arm                  | **analytic**, 26.03-49.21\times10^{18} depending on arm - see [Cost and FLOPs](#cost-and-flops). The logged W&B throughput counter (2.63\times10^{19} for every arm) **excludes the scoring forward passes** and so understates every selection arm. |
| Keep rate (where applicable) | top / middle **60%** of valid target tokens per sequence; **realized 0.599609** (1228 kept of 2047 valid target positions, logged over all 2048 positions) — see [Realized keep rate](#realized-keep-rate) |
| Primary metric               | Macro mean CE bits-per-byte over the **20 (task, split) labels** of the OLMo ladder — 10 OLMES benchmarks × val/test. MMLU supplies 8 of the 20 (**40% of the macro weight**), and 7 of the 20 are **test** splits, so "validation macro bpb" is a misnomer. |


Arm-by-arm contract table: [ARMS.md](ARMS.md). Contamination audit (paper Section 2.3):
[contamination/](contamination/). Code that built the training corpus and both reference
corpora from Hugging Face sources (Appendix A): [datasets/](datasets/).

Shared contract: same architecture, batch, LR, and step budget. The manipulation is **which tokens receive gradient** on each step (full CE vs a scored subset). Sequences still come from the same shuffled corpus; selection is per-token inside the batch.

### Frozen reference training data

Several arms score tokens against a **frozen** same-architecture CE model. Two reference checkpoints were used; they differ in the corpus they were trained on (not in student architecture).

**Reference A — HQ web mix (~5.5B).** Plain CE on an HQ-filtered ~5.509B-token corpus in Mixing Laws 7-domain proportions (scaled to that budget):


| Domain          | Approx. target tokens | Source character                                  |
| --------------- | --------------------- | ------------------------------------------------- |
| dclm            | ~2.07B                | DataDecide DCLM-Baseline QC (FineWeb-2 7% recipe) |
| arxiv           | ~1.38B                | arXiv                                             |
| starcoder       | ~0.78B                | StarCoder + Dolma code-HQ filter                  |
| pes2o           | ~0.52B                | peS2o                                             |
| open-web-math   | ~0.35B                | OpenWebMath (HQ; pool binds)                      |
| algebraic-stack | ~0.34B                | AlgebraicStack                                    |
| wiki            | ~0.09B                | Wikipedia                                         |


dolma2 tokenizer. Training FLOPs \approx 1.45\times10^{19}. Used as the frozen scorer for Middle-PPL late-average scores (mean of late checkpoints).

**Reference B — instruct mix.** Plain CE on a one-pass **English-filtered instruct** corpus (no upsampling to a fixed 5.5B cap; realized size is whatever the filtered unique pass yields). Sources are instruction / chat / math / code SFT-style collections with safety / tool / IFEval-oriented subsets dropped, then Dolma English document-score ≥ 0.5:


| Source family  | Kept (summary)                                                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tulu-v2        | all                                                                                                                                                                              |
| OpenHermes-2.5 | all (optional non-English metadata drop)                                                                                                                                         |
| Tulu-3         | FLAN, WildChat, math/code/science personas, Numina-TIR, Evol CodeAlpaca, SciRIFF, TableGPT, No Robots, OASST, etc.; drop WildGuard / WildJailbreak / CoCoNot / Aya / IF personas |
| Hermes-3       | all                                                                                                                                                                              |
| SmolTalk       | all configs except API-gen / constraint suites                                                                                                                                   |
| Dolci          | keep non-safety / non-tool / non-IF / non-Aya rows                                                                                                                               |


Domains labeled general / math / code / science / chat from metadata. dolma2 tokenizer. Training FLOPs \approx 1.04\times10^{19}. Used as the frozen scorer for ρ-1.

### What each method manipulates

**ρ-1 (excess loss).** For each target token, score L_{\mathrm{curr}} - L_{\mathrm{ref}} under the live student vs a **frozen** reference model of the same architecture. Keep the top 60% (largest excess loss — tokens where the student is worst relative to the reference). Selection is active from step 0. Method from [RHO-1](https://arxiv.org/abs/2404.07965).

**Attention top-k.** No external reference. Score each token by **causal attention received** on the last transformer layer (mean over heads of attention mass from later positions). Keep the top 60%. Active from step 0. Manipulation: prefer tokens that the model currently “attends to” as context for later predictions. Method from [ssToken](https://arxiv.org/abs/2510.18250) (attention-based score).

**Middle-PPL.** Score tokens by **late-checkpoint average** token CE under **Reference A**. Keep the **middle 60%** of the per-sequence score distribution (drop both easiest and hardest tails). Masks can be precomputed. Manipulation: train on “medium difficulty” tokens under that frozen scorer, excluding extremes. Method from [Marion et al., Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/abs/2309.04564).

**BLADE.** Bi-level setup with a **proxy** (trained student) and a **dynamic reference** that is periodically reset from the proxy. Steps 0–499: full CE on the proxy only (no selection). At sync steps 500, 875, 1250, 1625, 2000: copy proxy → reference, run K=75 reference updates, then keep proxy tokens with largest L_{\mathrm{proxy}}-L_{\mathrm{ref}} at keep-rate \gamma=0.6 (\tau=375). After the last sync, hold that reference to the end. Manipulation: select tokens where the fast proxy outruns a lagged copy of itself. Method from [BLADE](https://arxiv.org/abs/2606.18650).

**REL-EMA (exponential).** Online relative loss vs an **EMA of the student** (bias-corrected from zero; no external seed). EMA rate \alpha(t)=1-e^{-t/300}. Score \mathrm{REL}=L_{\mathrm{curr}}-L_{\mathrm{hist}}; keep top 60%. Active from step 0. Manipulation: prefer tokens where the live model is worse than its own exponential history. Method from [ssToken](https://arxiv.org/abs/2510.18250) (retrospective excess loss / REL). Corrected 2026-09-05: the score was previously computed as $L_{\mathrm{hist}}-L_{\mathrm{curr}}$ (inverted relative to rho\_excess/blade's convention); re-run with the fixed polarity.

### Arms actually run

> **Compute is reported as analytic FLOPs only.** Wall-clock hours are not comparable
> across these runs: four arms ran on **8xA100-80GB** (rho-1, BLADE, Attention,
> Middle-PPL) while the **full-loss control, Random control and REL-EMA ran on 4xL40S**.
> The logged W&B throughput counter is also incomplete: it **excludes every scoring
> forward pass**, so it reports the same 2.63e19 for a full-CE run as for an arm that
> additionally evaluates a frozen reference on every token. See
> [Cost and FLOPs](#cost-and-flops) for the model and the in-run/with-reference split.

| Arm | Analytic FLOPs (x10^18) | Relative to control |
| --- | --- | --- |
| **Control (full-CE)** | **26.03** | **1.00x** |
| Random control | 26.03 | 1.00x |
| Attention top-k | 26.11 | 1.00x |
| REL-EMA (exponential) | 34.71 | 1.33x |
| rho-1 | 45.08 | 1.73x |
| BLADE | 47.98 | 1.84x |
| Middle-PPL | 49.21 | 1.89x |

The rho-1, BLADE and Middle-PPL totals include pretraining their reference models.
Middle-PPL's figure also covers precomputing its masks with one forward pass over the
whole corpus, which is why it is the most expensive arm despite adding no scoring cost
during training itself.

**BLADE's W&B record does not contain its own curve.** Run `005xjces` has a **7-second
runtime** and logs **no throughput and no keep-rate metrics**, so its loss curve was
backfilled or resumed into that record rather than produced by it. The BLADE numbers
are usable but their provenance is one hop removed from the run ID they are attributed
to.


Control is `eduLLM/token-selection/349f144dc23ee52d18396be695d6b6b0`
(`full-loss-control-regmix10b-v3`), a native FarmShare run on 4xL40S, `eval/macro_bpb`
final 1.6751. It replaced `full-loss-control-regmix10b-v2` (`hh19uatg`), which was cloned
from `eduLLM/hpo-ladder` and mismatched the selection arms on dataloader seed (6199 vs 42),
step count (2384 vs 2360) and eval grid (~119 vs 125 steps). v3 is matched on init
(confirmed via the step-0 eval fingerprint), data seed, step count and eval grid, so that
confound is resolved. See `ARMS.md` for the field-by-field comparison.

Initialization is not uniform across the arms: the step-0 eval splits the eight runs into
**4.4610** bpb (random control seed 69), **4.4662** bpb (full-loss control, random control
seed 42, REL-EMA) and **4.4838** bpb (rho-1, Attention, BLADE, Middle-PPL), a 0.0228 bpb
spread before any training. The full-loss control shares an initialization with the seed-42
random control but not with seed 69, so the pooled random-control baseline mixes two
initializations; rho-1 is in a third group again.

A random-60% selection control (keep-rate 60% chosen uniformly at random per row, no scoring
signal) was run on 4×L40S at **two seeds** — `random-control-regmix10b-v1` (seed 42) and
`random-control-regmix10b-seed69-v1` (seed 69) — using the same `pretrain/regmix-10b` v1
corpus. It is reported as a single two-run fit; see Results below.

---

## Evaluation and uncertainty

### Fitting and bootstrap procedure

Every fitted-final number and every CI in this README comes from the following
procedure. It is the matched protocol used for every number in this file.

1. **Model.** `y = a + b * step^(-alpha)` fitted to the macro task-loss curve.
2. **Fit window.** Only steps **>= 1000** are used; earlier points are dominated by the
   LR warmup transient and bias `alpha`.
3. **alpha search.** `alpha` is chosen by grid search over
   `np.linspace(0.05, 6.0, 1192)`, with `a` and `b` solved in closed form by least
   squares at each `alpha`. The bounds are wide enough that **no arm's fitted
   `alpha` sits on a boundary** (largest: REL-EMA at 3.502; smallest: BLADE at 0.794),
   so the exponent is data-determined rather than clipped.
4. **Bootstrap.** 10,000 i.i.d. residual bootstrap draws: residuals from the point fit
   are rescaled by `sqrt(n/(n-p))` with `p=3` (1.173 at `n=11`, 1.076 at `n=22`),
   then resampled with replacement and added back to the fitted curve. OLS residuals
   are shrunk relative to the true errors by that factor on average, so resampling
   them raw understates the spread. Without the rescaling every interval here would
   be ~15% narrower and the rho-1-vs-random p-value would read 0.029 rather than
   0.048.
5. **alpha re-estimated on every draw.** Each bootstrap replicate re-runs the full
   `alpha` grid search rather than holding `alpha` at its point estimate. This was
   chosen deliberately **because it yields the wider intervals** -- it propagates
   curvature uncertainty instead of conditioning it away.
6. **Interval.** 95% CI = the 2.5 / 97.5 percentiles of the bootstrap distribution of
   the fitted final.
7. **Fitted final** is evaluated at **each arm's own final logged step** (2360 for all
   seven arms).
8. **RNG seed 0**, so the numbers are reproducible.

**The random control is fitted across two runs.** It was run at seeds 42 and 69. Rather
than picking one, it is reported as a single **two-run fit**: one power law fitted by
least squares to the union of both runs' fit-window points (11 each, 22 total), with
`alpha` profiled on the same grid and the same 10,000-draw residual bootstrap taken over
the **pooled** residuals. Because the two runs are offset from one another, that offset
enters the residual pool, so this arm's interval carries seed-to-seed spread as well as
within-run noise. Every comparison involving the random control uses that fit.

**What these intervals are not.** For the other six arms they are **curve-fit intervals
for a SINGLE run**, quantifying only how well a power law pins down the endpoint of one
observed trajectory, with **no seed-to-seed variance**. The one direct measurement we
have of that variance is the random control's two seeds, which differ by **0.0082 bpb**
(95% CI [0.0033, 0.0126], two-sided **p = 0.0002**) — larger than several between-arm
gaps. Treat any single-seed gap of that order as unresolved.

**Pairwise significance.** Paired bootstrap on the fitted finals: both arms' 10k draws
are generated with `default_rng(0)`, so the resample index matrices are identical and the
draws are **paired by index**; `delta_stats` then forms the empirical distribution of
`fitted_final(A) - fitted_final(B)`. The reported **one-sided** p-value is the bootstrap
mass on the side opposite the point estimate.

---

## Results

### Fitted final macro task-loss (bpb)

Matched-protocol Table 1. Lower is better.

| Arm                     | Fitted final | Observed | 95% CI           |
| ----------------------- | ------------ | -------- | ---------------- |
| Control (full-CE)       | **1.6718**   | 1.6751   | [1.6642, 1.6798] |
| rho-1                   | 1.6824       | 1.6843   | [1.6762, 1.6863] |
| Random control seed 42  | 1.6857       | 1.6870   | [1.6761, 1.6930] |
| Random control seed 69  | 1.6939       | 1.6951   | [1.6852, 1.7002] |
| **Random control (two-run fit)** | **1.6900** | 1.6911 | **[1.6837, 1.6950]** |
| Attention top-k         | 1.6991       | 1.7005   | [1.6925, 1.7045] |
| BLADE                   | 1.7090       | 1.7115   | [1.7020, 1.7197] |
| Middle-PPL              | 1.9047       | 1.9001   | [1.9000, 1.9101] |
| REL-EMA (exponential)   | 1.9199       | 1.9207   | [1.9176, 1.9220] |

Only **rho-1** overlaps the control's interval. The overlapping pairs at 95% are
control↔rho-1, rho-1↔random control, random control↔Attention, and Attention↔BLADE.

### Paired differences

Against the full-CE control:

| Arm | Delta | 95% CI | One-sided p |
| --- | --- | --- | --- |
| rho-1 | +0.0106 | [+0.0039, +0.0172] | 0.0005 |
| Random control (two-run) | +0.0181 | [+0.0092, +0.0264] | 0.0000 |
| Attention top-k | +0.0273 | [+0.0192, +0.0344] | 0.0000 |
| BLADE | +0.0372 | [+0.0318, +0.0426] | 0.0000 |
| Middle-PPL | +0.2329 | [+0.2252, +0.2418] | 0.0000 |
| REL-EMA | +0.2482 | [+0.2406, +0.2551] | 0.0000 |

Against the two-run random control:

| Arm | Delta | 95% CI | One-sided p |
| --- | --- | --- | --- |
| Control (full-CE) | -0.0181 | [-0.0264, -0.0092] | 0.0000 |
| **rho-1** | **-0.0075** | **[-0.0155, -0.0001]** | **0.024** |
| Attention top-k | +0.0091 | [+0.0017, +0.0164] | 0.0074 |
| BLADE | +0.0190 | [+0.0109, +0.0296] | 0.0000 |
| Middle-PPL | +0.2148 | [+0.2085, +0.2222] | 0.0000 |
| REL-EMA | +0.2300 | [+0.2248, +0.2359] | 0.0000 |

**rho-1 does beat random token masking** (two-sided p = 0.048) once the random control is
fitted across both of its seeds — under the single seed-42 control the same test returned
-0.0032, p = 0.434, i.e. no difference. That reversal is itself a caution: the comparison
is sensitive to which random-control run you pick, which is why the two-run fit is the
reported baseline. rho-1 still loses to full CE by 0.0106 bpb.

### Realized keep rate

The four online-scoring arms all realized a keep rate of exactly **0.599609** --
**1228 of 2047** target positions per 2048-token sequence -- and it was **constant at
every logged step**. This is the W&B metric `train/selected token fraction`, and it is
identical for `rho-1`, `attention`, `rel_ema` and `random_control`. The value is
`round(2047 * 0.6) / 2048`: the nominal 0.6 is applied to the 2047 valid next-token
targets in a 2048-token sequence and rounded to an integer count per row (1228), while
the logged metric divides by all 2048 positions. Against the 2047 valid targets the
realized rate is 1228/2047 = 0.599902. Because the
count is fixed per row rather than thresholded on the score, the keep rate carries no
information about the scorer -- the arms differ only in *which* 1228 tokens they keep.

**Middle-PPL used precomputed masks** (offline scoring against Reference A), so it never
logs `train/selected token fraction`; the absence of that metric for `middle-ppl-token`
is expected and is not evidence of a different keep rate.

### Cost and FLOPs

Wall-clock hours are not a usable cost measure here: four arms (rho-1, BLADE,
Attention, Middle-PPL) ran on **8xA100-80GB** while the **full-loss control, Random
control and REL-EMA ran on 4xL40S**, and hours on those two platforms are not
interchangeable. The **logged throughput FLOPs counter** is also incomplete, because it
**excludes the scoring forward passes** and so reports the same 2.63e19 for a full-CE
run as for a run that additionally evaluates a frozen reference on every token, an
understatement of up to ~1.9x. We therefore report analytic FLOPs throughout.

**Model.** Forward cost per token = `2N + 4*L*T*d`, with `N = 371,195,904` matmul
parameters (**including the untied output head**), `L = 16` layers, `T = 2048` sequence
length, `d = 1024` model width. The `4*L*T*d` term is the attention score/value
matmuls, which scale with sequence length and are not captured by a parameter count.
Forward+backward = **3x** forward. This model reproduces the logged W&B throughput
counter to within **0.05%**, which is what licenses using it for the arms whose counter
is incomplete.

| Arm | In-run FLOPs (x10^18) | With reference pretraining (x10^18) | Relative to control |
| --- | --- | --- | --- |
| **Control (full-CE)** | **26.03** | **26.03** | **1.00x** |
| Random control | 26.03 | 26.03 | 1.00x |
| Attention top-k | 26.11 | 26.11 | 1.00x |
| REL-EMA (exponential) | 34.71 | 34.71 | 1.33x |
| rho-1 | 34.71 | 45.08 | 1.73x |
| BLADE | 39.71 | 47.98 | 1.84x |
| Perplexity (Middle-PPL) | 34.71 | 49.21 | 1.89x |

Reading the table: Attention top-k is essentially free -- the score is read off
attention already computed in the forward pass, and dropping 40% of tokens from the
loss makes it marginally *cheaper* than full CE. Every arm that needs a second model's
forward pass -- REL-EMA's EMA copy, rho-1's and Middle-PPL's frozen reference, BLADE's
lagged reference -- pays ~1.33x in-run, and the arms that also had to *pretrain* that
reference pay up to **1.89x** end to end. **The two most expensive arms are two of
the worst-performing ones** (Perplexity 1.89x, BLADE 1.84x, rho-1 1.73x, against
Attention's 1.00x), so token selection bought negative return on a large compute
premium.

**BLADE's K-update overhead, itemized.** 5 syncs x 75 K-steps x 2 streams (proxy and
reference) = **750 full batches** of 4,194,304 tokens = **3,145,728,000 tokens** of
forward+backward, on top of the 2360 training steps. At 2.6298e9 FLOPs/token that is
8.27e18, which accounts for the **47.98 vs 39.71** total-column difference -- not the
39.71 vs 34.71 in-run difference, which is BLADE's second per-step scoring pass over the
1860 steps after step 500 (13.68e18 vs 8.68e18 for a single-scoring arm).

### Takeaways

1. **Full-token CE wins.** Dropping ~40% of tokens never improved macro task-loss at
   this budget.
2. **Ranking among failures.** rho-1 < Random control < Attention top-k < BLADE <<
   Middle-PPL ~ REL-EMA. Excess loss vs a strong frozen reference is least damaging;
   middle-percentile perplexity and EMA-relative selection (once corrected to the same
   current-minus-history polarity as rho-1/BLADE) land in the same worst tier, close to
   each other (~+0.25 bpb over full-CE) rather than REL-EMA being a catastrophic
   outlier.
3. **rho-1 beats random masking, but only just, and only against the two-run control.**
   The two-run random control lands at 1.6900 and rho-1 at 1.6824, a -0.0075 bpb edge
   (two-sided **p = 0.048**). Against the seed-42 run alone the same test gave -0.0033,
   **p = 0.363** -- no difference. The conclusion flips with the choice of control run,
   which is why the two-run fit is the reported baseline and why this edge should be
   read as suggestive rather than settled. Every *other* selection arm (Attention,
   BLADE, Middle-PPL, REL-EMA) is clearly worse than the random control.
4. **BLADE's extra machinery did not pay off** -- worse than simple rho-1 despite
   3,145,728,000 extra tokens of K-update forward+backward (1.84x control FLOPs).
5. **Cost.** Token selection was the most expensive lever and the weakest scientific
   return: up to **1.89x** the control's analytic FLOPs for a strictly worse result.
6. **Seed variance is the binding constraint.** The random control's two seeds differ by
   0.0082 bpb -- larger than rho-1's 0.0106 bpb deficit to full CE is comfortable with,
   and larger than the Attention-vs-random gap. Only the random control has a seed
   replicate; the other six arms are single runs, so their intervals bound within-run
   noise only.

---

## Conclusions

Under the P1 Mixing Laws Dataset × 370M one-epoch contract, **token selection is a negative result**: every tested scorer underperforms full CE. The best scorer (ρ-1) does edge past a random-60% keep-rate control (delta -0.0075, 95% CI [-0.0155, -0.0001], **two-sided p = 0.048**), so the scoring rule is not worthless -- but it still loses to full CE by 0.0106 bpb, so selection does not pay for itself here. Prefer mixture optimization (MixLaw) or, secondarily, difficulty curricula over token masking for this setup.

Two caveats a reader should carry out of this page. First, only the random control has a
seed replicate; the other six arms are single runs whose intervals contain **no
seed-to-seed variance**, and the one seed contrast we can measure is 0.0082 bpb, which is
of the same order as the closest between-arm gaps. Second, initialization was **not**
uniform across arms -- the step-0 eval splits them into three groups spanning 0.0228 bpb,
ρ-1 sits apart from both controls, and the two random-control seeds do not share an
initialization with each other. The direction of the headline result
is robust (the selection arms lose to full CE by 0.011 to 0.25 bpb), but the ordering
inside the top cluster is not settled.

Benchmark contamination for this corpus and both reference corpora is audited in
[`contamination/`](contamination/).
