# Skill-It (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can Skill-It domain reweighting — driven by an offline probe adjacency or by online mixing-law derivatives — improve macro task-loss over a fixed Data Mixing Laws paper mixture under a matched one-epoch budget?

**Answer.** No. All three Skill-It arms finished at or worse than the Data Mixing Laws paper control. The best Skill-It arm (offline probe adjacency) is statistically indistinguishable from control; the derivative and MixLaw-seeded offline arms are clearly worse.

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
p_i \propto \exp\!\Big(\eta\, w \sum_j A_{ij} L_j\Big),\qquad \eta=0.2,\; w=1
\]

then renormalize \(p\) onto the simplex. Intuition: domains that the adjacency says “help” high-loss task families get more mass. Arms differ only in **how \(A\) is built** and (for one arm) **where \(p\) starts**.

### Offline probe matrix

1. Train **7 one-hot** DataDecide-60M probes (100% of each domain in turn; 5 tokens/param → 1451 steps / ~285M tokens; \(N=57.1\mathrm{M}\) non-embedding).
2. Fit Chinchilla step-laws on in-run curves; extrapolate to tpp=20 (step 5806).
3. Build
   \[
   A_{ij} = \max\!\big(0,\; L_j(r_{\mathrm{DML}}) - L_j(i)\big)
   \]
   where \(L_j(i)\) is family \(j\)’s extrapolated loss after training on 100% domain \(i\), and \(r_{\mathrm{DML}}\) is the Data Mixing Laws paper mixture. Positive \(A_{ij}\) means domain \(i\) beat that paper mix on family \(j\).

**Probe FLOPs** (Chinchilla \(C\approx 6ND\)): \(\approx 9.8\times10^{16}\) per probe → **\(\approx 6.8\times10^{17}\)** for all 7.

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

### Arms actually run

Skill-It reweighting follows [Chen et al., Skill-It!](https://arxiv.org/abs/2307.14430). Online derivative \(A\) additionally uses the MixLaw parametric form from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952).

| Arm | Manipulation | A100-h | FLOPs |
|-----|--------------|-------:|------:|
| Offline probe | Start at Data Mixing Laws paper mix; at each update apply Skill-It with the **fixed** offline \(A\) above and current task losses | 47.26 | \(2.63\times10^{19}\) |
| Online derivative | Start at Data Mixing Laws paper mix; at each update **recompute** \(A(r)\) from MixLaw derivatives, then Skill-It-update | 53.8 | \(2.63\times10^{19}\) |
| Offline (MixLaw start) | Same fixed offline \(A\) as Offline probe, but **initialize** domain weights at the MixLaw optimum instead of the Data Mixing Laws paper mix | 47.14 | \(2.63\times10^{19}\) |
| **Total** | | **148.2** | **\(7.89\times10^{19}\)** |

Comparisons use the Data Mixing Laws paper full run as control (fixed weights for the whole epoch; not an extra Skill-It train). A100-hours are the no-waste totals measured for these runs. Online derivative’s A100-hours are throughput-repriced to steady-state (W&B wall was 70.51 A100-h; defective-pod I/O on a mid-run stretch is excluded).

### Domain weights after each update

Logged \(p\) (post-update domain mixture) from W&B `skillit/weight/*` at step 0 and each Skill-It update. Weights sum to 1.

**Offline probe**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.062 | 0.016 |
| 500 | 0.194 | 0.131 | 0.106 | 0.157 | 0.122 | 0.124 | 0.165 |
| 875 | 0.190 | 0.132 | 0.109 | 0.156 | 0.124 | 0.126 | 0.163 |
| 1250 | 0.188 | 0.133 | 0.110 | 0.156 | 0.125 | 0.126 | 0.163 |
| 1625 | 0.187 | 0.133 | 0.111 | 0.156 | 0.125 | 0.126 | 0.162 |
| 2000 | 0.186 | 0.133 | 0.111 | 0.155 | 0.126 | 0.127 | 0.162 |

**Online derivative**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.062 | 0.016 |
| 500 | 0.141 | 0.124 | 0.124 | 0.144 | 0.145 | 0.124 | 0.199 |
| 875 | 0.151 | 0.129 | 0.129 | 0.145 | 0.137 | 0.129 | 0.180 |
| 1250 | 0.151 | 0.129 | 0.129 | 0.144 | 0.138 | 0.129 | 0.180 |
| 1625 | 0.151 | 0.129 | 0.129 | 0.145 | 0.138 | 0.129 | 0.180 |
| 2000 | 0.151 | 0.129 | 0.129 | 0.145 | 0.138 | 0.129 | 0.179 |

**Offline (MixLaw start)**

| step | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----:|-----:|------:|----------:|------:|--------------:|----------------:|-----:|
| 0 | 0.568 | 0.000 | 0.000 | 0.097 | 0.035 | 0.000 | 0.300 |
| 500 | 0.194 | 0.131 | 0.107 | 0.156 | 0.122 | 0.124 | 0.166 |
| 875 | 0.190 | 0.132 | 0.109 | 0.156 | 0.124 | 0.125 | 0.163 |
| 1250 | 0.188 | 0.132 | 0.110 | 0.156 | 0.125 | 0.126 | 0.162 |
| 1625 | 0.187 | 0.133 | 0.111 | 0.155 | 0.125 | 0.127 | 0.162 |
| 2000 | 0.186 | 0.133 | 0.111 | 0.155 | 0.126 | 0.127 | 0.162 |

---

## Evaluation and uncertainty

Same as MixLaw: power law \(y = a + b/\mathrm{step}^{\alpha}\) on steps ≥ 1000; fitted final as center; residual bootstrap (10k) 95% CI. \(p\)-values vs control are omitted because **no Skill-It arm beat control**.

---

## Results

### Fitted final macro task-loss (bpb)

| Arm | Fitted final | Observed | 95% CI |
|-----|-------------:|---------:|--------|
| Data Mixing Laws paper (control) | **1.6518** | 1.6518 | [1.6484, 1.6553] |
| Offline probe | 1.6544 | 1.6631 | [1.6476, 1.6614] |
| Online derivative | 1.6690 | 1.6745 | [1.6603, 1.6772] |
| Offline (MixLaw start) | 1.6747 | 1.6748 | [1.6693, 1.6792] |

Lower is better. Control is best. Offline probe overlaps control; Online derivative and Offline (MixLaw start) sit clearly above.

### Takeaways

1. **Skill-It did not help** under this one-epoch 370M contract — mid-run reweighting failed to beat a static Data Mixing Laws paper mix.
2. **Offline probe ≈ control** — five Skill-It updates with the probe adjacency neither help nor clearly hurt once curve uncertainty is accounted for.
3. **Online derivatives and starting from MixLaw weights hurt** — both finish ~0.02 bpb worse than the Data Mixing Laws paper control.
4. **Cost.** Three Skill-It trains 148.2 A100-hours and \(\approx 7.89\times10^{19}\) FLOPs (Online derivative alone 53.8 A100-h), plus \(\approx 6.8\times10^{17}\) FLOPs for the 60M probes.

---

## Conclusions

At this scale and budget, **static MixLaw mixtures dominate Skill-It reweighting**. Holding a good fixed mix for the full epoch beats adapting domain weights online from either probe or mixing-law adjacencies.
