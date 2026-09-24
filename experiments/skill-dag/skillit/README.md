# Skill-It (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can Skill-It domain reweighting — driven by an offline probe adjacency or by online mixing-law derivatives — improve macro task-loss over a fixed Data Mixing Laws paper mixture under a matched one-epoch budget?

**Answer.** Both Skill-It arms beat the Olmo-mix-1124 control decisively
($p < 10^{-4}$), but **neither beat the best static mixture** (the LightGBM
optimum they start from). The offline probe arm ends 0.0033 bpb above it — a
gap smaller than the 0.0044 bpb difference we measure between two dataloader
seeds of the same mixture, so the two are not meaningfully separable. The
online derivative arm ends 0.0086 bpb above it, about twice the seed noise
floor. Mid-run reweighting therefore bought nothing here.

---

## Setup

| Knob | Value |
|------|-------|
| Architecture | OLMo-2 370M (full attention), shared 370M contract |
| Train stream | Domain-stratified sampling over 7 OLMoHQ domains with **time-varying** mixture weights |
| Domains | dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki |
| Global batch / seq / LR | 4,194,304 / 2048 / \(4\times10^{-4}\) cosine (warmup 24, \(\alpha_f=0.1\)) |
| Full-run budget | ~2384 steps ≈ one epoch |
| FLOPs / full arm | \(2.63\times10^{19}\) (measured from W&B) |
| Skill-It update | \(\eta=0.2\), \(w=1\); five mid-run updates |
| Update schedule | steps 500, 875, 1250, 1625, 2000 |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels |

Unlike MixLaw (fixed weights for the whole run), Skill-It **reweights domains mid-training**. Between updates the sampler holds the current mixture fixed; at each update step it rebuilds domain probabilities from an adjacency \(A\) and the current per-family losses \(L\).

### Skill-It update (shared by all arms)

\[
p_i(t{+}1) \;\propto\; p_i(t)\,\exp\!\Big(\eta\, w \sum_j A_{ij} L_j\Big),\qquad \eta=0.2,\; w=1
\]

then renormalize \(p\) onto the simplex. This is the **multiplicative-weights**
rule of Chen et al. (their Eq. 4): the update *rescales the current mixture*, it
does not rebuild it from \(A L\) alone. A domain whose \(A\) row is all zeros
therefore keeps its existing share (scaled by the common normalizer) rather than
collapsing to parity with every other zero-row domain — which is exactly what the
logged trajectories below show. Intuition: domains that the adjacency says “help”
high-loss task families get more mass. Arms differ only in **how \(A\) is built**
and (for one arm) **where \(p\) starts**.

### Offline probe matrix

1. Train **7 one-hot** DataDecide-60M probes (100% of each domain in turn; 5 tokens/param → 1451 steps / ~285M tokens; \(N=57.1\mathrm{M}\) non-embedding).
2. Fit Chinchilla step-laws on in-run curves; extrapolate to tpp=20 (step 5806).
3. Build
   \[
   A_{ij} = \max\!\big(0,\; L_j(r_{\mathrm{DML}}) - L_j(i)\big)
   \]
   where \(L_j(i)\) is family \(j\)’s extrapolated loss after training on 100% domain \(i\), and \(r_{\mathrm{DML}}\) is the Data Mixing Laws paper mixture. Positive \(A_{ij}\) means domain \(i\) beat that paper mix on family \(j\).

**Probe FLOPs.** Kaplan et al. (2020) estimate including the attention term,
\(C \approx 6 N_{\text{non-emb}} D + 12\,n_{\text{layers}}\,s\,d_{\text{model}}\,D\),
at the trained model's \(N_{\text{non-emb}} = 76{,}296{,}576\):
\(\approx 1.74\times10^{17}\) per probe → **\(\approx 1.22\times10^{18}\)** for all 7.
(Plain \(6ND\) at the DataDecide budget-setting \(N=57.1\)M gives \(9.8\times10^{16}\)
per probe; that older figure understates the cost by ~1.8x and is superseded.)

**Offline \(A\) used by the Offline probe arm** (rows = domains, columns = task families; Chinchilla step 5806):

| domain \\ family | arc_challenge | arc_easy | mmlu_humanities | mmlu_other | mmlu_social_sciences | mmlu_stem |
|------------------|--------------:|---------:|----------------:|-----------:|---------------------:|----------:|
| dclm | 0.340 | 0.149 | 0.368 | 0.000 | 0.079 | 0.486 |
| arxiv | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.340 |
| starcoder | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| pes2o | 0.252 | 0.084 | 0.000 | 0.000 | 0.000 | 0.450 |
| open-web-math | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.232 |
| algebraic-stack | 0.017 | 0.000 | 0.000 | 0.000 | 0.000 | 0.242 |
| wiki | 0.227 | 0.000 | 0.470 | 0.013 | 0.000 | 0.352 |

Starcoder’s row is all zeros (never beats the Data Mixing Laws paper mix on these families at Chinchilla scale). DCLM and wiki dominate many columns.

### Online mixing-law derivative

Reuse the parametric MixLaw surrogate per family \(j\):

\[
L_j(r) = c_j + k_j \exp\!\Big(\sum_i t_{ij} r_i\Big)
\]

Skill-It-compatible adjacency at the **current** weights \(r\):

\[
A_{ij} = \max\!\big(0,\; -(dL_j/dr_i)\big) = \max\!\big(0,\; -t_{ij}(L_j(r)-c_j)\big)
\]

So \(A\) **changes every update** as \(r\) and predicted \(L(r)\) move. Fitted \(t_{ij}\) / \(c_j\) / \(k_j\) are those from the MixLaw parametric fit.

#### Which \(r\) a published derivative matrix is evaluated at

Because \(A\) depends on \(r\), a single printed derivative matrix is a snapshot and
the reference point has to be stated. Two are in play and they are not the same
matrix. Run
[`compare_offline_online_A.py`](compare_offline_online_A.py) to regenerate both.

| Evaluated at | Density | Pearson \(r\) vs offline probe \(A\) | Edge-presence disagreements |
|---|---:|---:|---:|
| \(r_{\mathrm{DML}}\) (`mix01`) | 13/42 (31%) | **0.171** | 10/42 |
| `LGB-min1pct` (the arm's own start) | 13/42 (31%) | **0.065** | 10/42 |

\(r_{\mathrm{DML}}\) is the like-for-like point, since the offline probe matrix is
also referenced to \(r_{\mathrm{DML}}\); `LGB-min1pct` is the matrix the derivative
arm actually used at its first update. At \(r_{\mathrm{DML}}\) the entries are
roughly twice as large (dclm → arc_challenge 0.154 vs 0.065). Density and the
10-of-42 edge-presence disagreement are the same at both points; only the
correlation and the magnitudes move.

Figure IV is generated by
[`plot_adjacency_comparison.py`](plot_adjacency_comparison.py) and its derivative
panel is evaluated at **`LGB-min1pct`**, which is the point the paper's body
describes and the point its \(r = 0.07\) is computed at; the committed output is
[`figures/adjacency_comparison.png`](figures/adjacency_comparison.png).

An earlier version of that figure was evaluated at \(r_{\mathrm{DML}}\) instead.
It put entries roughly twice as large in the derivative panel (dclm →
arc_challenge 0.154 rather than 0.065), so a reader who recomputed the
correlation from it got 0.17 rather than the 0.07 in the text. That version
survives in PDF exports of the paper taken before 2026-09-17, and three
independent reviewers working from one of those exports each flagged it; the
live document has carried the `LGB-min1pct` panel since then. Every qualitative
claim in the paper's Appendix E holds at either point —
StarCoder and Algebraic Stack all-zero, arXiv all-zero, OpenWebMath helping only
ARC Easy, pes2o trivial on both ARC skills, Wikipedia and DCLM broadly helpful,
31% density, 10 of 42 edge-presence disagreements — so only the printed
magnitudes and the correlation were ever affected.

### Arms actually run

Skill-It reweighting follows [Chen et al., Skill-It!](https://arxiv.org/abs/2307.14430). Online derivative \(A\) additionally uses the MixLaw parametric form from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952).

Two arms were trained, both starting from the **LightGBM-optimized mixture**
(`LGB-min1pct`, id 27 in
[`../mixlaw/validation_mixtures_10b.json`](../mixlaw/validation_mixtures_10b.json))
— confirmed by each run's own step-0 logged weights
(`skillit-370m-probe-rerun-20260918-011120`,
`skillit-370m-deriv-20260916-124719`), which match that published weight vector
to full float precision.

| Arm | Manipulation | A100-h | FLOPs |
|-----|--------------|-------:|------:|
| Offline probe | Start at the LightGBM-optimized mixture; at each update apply Skill-It with the **fixed** offline \(A\) above and current task losses | 47.26 | \(2.63\times10^{19}\) |
| Online derivative | Start at the LightGBM-optimized mixture; at each update **recompute** \(A(r)\) from MixLaw derivatives, then Skill-It-update | 53.8 | \(2.63\times10^{19}\) |
| **Total** | | **101.06** | **\(5.26\times10^{19}\)** |

Comparisons use the **Olmo-mix-1124 seed average** as the control (the corpus's
natural weighting), and additionally report each arm against the **LightGBM
static** mixture it starts from. Both are fixed-weight full runs, not extra
Skill-It trains. A100-hours are the no-waste totals measured for these runs. Online derivative’s A100-hours are throughput-repriced to steady-state (W&B wall was 70.51 A100-h; defective-pod I/O on a mid-run stretch is excluded).

### Domain weights after each update

Logged \(p\) (post-update domain mixture) read directly from each run's own
`skillit_updates.jsonl` progress log at step 0 and each Skill-It update.
Weights sum to 1. Step 0 for both arms is `LGB-min1pct`, the
LightGBM-optimized mixture (see Arms actually run above).

**Offline probe**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| 500 | 0.646 | 0.168 | 0.057 | 0.077 | 0.031 | 0.010 | 0.011 |
| 875 | 0.720 | 0.131 | 0.037 | 0.071 | 0.023 | 0.008 | 0.011 |
| 1250 | 0.779 | 0.101 | 0.023 | 0.064 | 0.017 | 0.006 | 0.010 |
| 1625 | 0.827 | 0.077 | 0.015 | 0.057 | 0.012 | 0.004 | 0.009 |
| 2000 | 0.864 | 0.057 | 0.009 | 0.050 | 0.008 | 0.003 | 0.008 |

**Online derivative**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| 500 | 0.556 | 0.200 | 0.082 | 0.086 | 0.047 | 0.013 | 0.016 |
| 875 | 0.557 | 0.189 | 0.078 | 0.091 | 0.052 | 0.012 | 0.021 |
| 1250 | 0.556 | 0.178 | 0.073 | 0.095 | 0.057 | 0.011 | 0.028 |
| 1625 | 0.553 | 0.168 | 0.069 | 0.099 | 0.062 | 0.011 | 0.037 |
| 2000 | 0.549 | 0.159 | 0.065 | 0.103 | 0.067 | 0.010 | 0.047 |

Offline probe's fixed adjacency drives weight toward `dclm` monotonically at
every update; online derivative's recomputed adjacency instead redistributes
weight from `dclm`/`arxiv` onto `wiki` while leaving `dclm` close to its
starting share. See
[`contamination/README.md`](contamination/README.md#reading-these-together)
for how this drives each arm's contaminated exposure.

---

## Training-code provenance

Where each Skill-It 370M run ran, as recorded in its W&B run metadata
(`eduLLM/skillit`):

| Arm | W&B run | Platform |
|-----|---------|----------|
| Offline probe | `87ad0201c4b5781a3df50d7bb394776c` | FarmShare, 4×L40S |
| Online derivative | `c0844ce36f24d6773c7f45cb31d810f4` | FarmShare, 4×L40S |

Both ran `.edullm/runpod/entrypoint.py` from a copy of OLMo-core's Skill-It
`.edullm/` code synced to FarmShare scratch. Those files live on OLMo-core's
Skill-It branches (`edullm/skillit-370m`, `reconnect/skillit-370m`), but the
versions that ran were uncommitted: the two run copies are identical to each
other, and their `train_skillit_370m.py`, `skillit_entrypoint.py` and
`skillit_controller.py` match no commit on any OLMo-core branch and no file in
this repository. The controls they are compared against are the
MixLaw runs; see
[`../mixlaw/README.md`](../mixlaw/README.md#training-code-provenance).

## Evaluation and uncertainty

Same as MixLaw: power law \(y = a + b/\mathrm{step}^{\alpha}\) on steps ≥ 1000;
fitted final as center; **alpha-free** residual bootstrap (10k draws, \(\alpha\)
re-selected on every draw) for the 95% CI. Reproduce with
[`../mixlaw/fit_and_bootstrap_370m.py`](../mixlaw/fit_and_bootstrap_370m.py) from
the committed curves in
[`../mixlaw/skill_dag_370m_wandb_curves.json`](../mixlaw/skill_dag_370m_wandb_curves.json).

---

## Results

Numbers below are the re-pulled, corrected figures for the two real runs
(`skillit-370m-probe-rerun-20260918-011120`,
`skillit-370m-deriv-20260916-124719`), both starting from the LightGBM optimum.
They supersede an earlier table on this page that was computed under the
mistaken assumption of a Data-Mixing-Laws-paper start and used that mixture as
the control.

### Fitted final macro task-loss (bpb)

| Arm | Fitted final | Observed | 95% CI | vs Olmo control |
|-----|-------------:|---------:|--------|-----------------|
| Olmo-mix-1124 average (control) | 1.6291 | 1.6327 | [1.6246, 1.6335] | — |
| LightGBM static (start mixture) | **1.6080** | 1.6077 | [1.6049, 1.6106] | \(p < 10^{-4}\) |
| Offline probe | 1.6112 | 1.6124 | [1.6078, 1.6141] | \(p < 10^{-4}\) |
| Online derivative | 1.6166 | 1.6216 | [1.6114, 1.6235] | \(p = 0.004\) |

Lower is better. Both Skill-It arms beat the Olmo-mix-1124 control (probe
\(p < 10^{-4}\), derivative \(p = 0.004\)). Neither beats the LightGBM static
mixture they start from:

| Comparison | Δ bpb | 95% CI | \(p\) |
|------------|------:|--------|------:|
| Offline probe − LightGBM static | +0.0033 | [-0.0011, +0.0075] | 0.14 |
| Online derivative − LightGBM static | +0.0086 | [+0.0026, +0.0161] | 0.0020 |

**Seed noise floor.** Two Olmo-mix-1124 runs differing only in dataloader seed
land 0.0044 bpb apart (95% CI [-0.0046, 0.0132], \(p = 0.34\)). The probe arm's
0.0033 bpb deficit is *below* that floor; the derivative arm's 0.0086 bpb deficit
is about twice it.

### Takeaways

1. **Skill-It did not help** under this one-epoch 370M contract — neither arm
   improved on the static mixture it started from.
2. **Offline probe ≈ LightGBM static.** The gap is not statistically
   distinguishable (\(p = 0.14\); the 95% CI [-0.0011, +0.0075] spans zero) and
   is smaller than the seed-to-seed spread, so we do not claim the static
   mixture is genuinely better.
3. **Online derivative clearly hurts** — 0.0086 bpb worse than its own starting
   mixture, ~2x the seed floor.
4. **Both still beat the natural corpus weighting** by 0.013–0.018 bpb (probe
   \(p < 10^{-4}\), derivative \(p = 0.004\)); the failure is specific to beating
   an *already-optimized* static mixture.
5. **Cost.** Two Skill-It trains 101.06 A100-hours and \(\approx 5.26\times10^{19}\)
   FLOPs (Online derivative alone 53.8 A100-h), plus \(\approx 1.22\times10^{18}\)
   FLOPs for the 60M probes.

---

## Conclusions

At this scale and budget, **a good static mixture is not improved by Skill-It
reweighting**. Holding the LightGBM optimum fixed for the full epoch is at least
as good as adapting domain weights mid-run from either a probe or a mixing-law
adjacency. Note the design limit: both arms start *at* an optimized mixture and
move away from it, so this tests whether Skill-It can improve on a good mix, not
whether it can rescue a bad one.

Benchmark contamination between the shared 127B-token reservoir and the
evaluation suite is audited in [contamination/](contamination/), including
each arm's time-weighted exposure over its realized, time-varying mixture.
