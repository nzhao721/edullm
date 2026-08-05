# MixLaw (Mixing Laws Dataset / OLMoHQ × OLMo-2 370M)

**Question.** Can a mixing law fitted on cheap short runs predict a domain mixture that beats OLMo Mix 1124 under a matched one-epoch training budget?

**Answer.** Yes. Both fitted mixtures beat the OLMo Mix 1124 control on macro task-loss bits-per-byte (bpb), with non-overlapping 95% confidence intervals and two-sided residual-bootstrap $p < 10^{-4}$. The MixLaw optimum is slightly better than the LightGBM optimum.

---

## 370M validation setup

| Knob | Value |
|------|-------|
| Architecture | OLMo-2 370M (full attention) |
| Train stream | Domain-stratified sampling over 7 OLMoHQ domains at fixed recipe weights |
| Domains | dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki |
| Global batch / seq / LR | 4,194,304 tokens / 2048 / $4\times10^{-4}$ cosine ($T_{\max}=2360$, warmup 24, $\alpha_f=0.1$) |
| Full-run budget | 2360 steps ≈ one epoch (~10B tokens) |
| FLOPs / full arm | $2.63\times10^{19}$ (measured from W&B) |
| Primary metric | Macro mean CE bits-per-byte over 20 OLMES-style labels (task-loss) |

**Shared recipe across arms:** same architecture, tokenizer, batch, LR schedule, and one-epoch step budget. Arms differ only in **domain mixture weights**.

### Arms actually run (370M)

| Arm | Role | A100-h | FLOPs |
|-----|------|-------:|------:|
| OLMo Mix 1124 | Control — natural OLMo Mix 1124 domain proportions (~95% DCLM) | 47.46 | $2.63\times10^{19}$ |
| Data Mixing Laws paper | Fixed proportions from the Data Mixing Laws paper (Pilot 01) | 44.11 | $2.63\times10^{19}$ |
| LightGBM | LightGBM mixing-law optimum | 48.39 | $2.63\times10^{19}$ |
| MixLaw | Parametric mixing-law optimum | 48.78 | $2.63\times10^{19}$ |
| **Total** | | **188.74** | **$1.05\times10^{20}$** |

A100-hours are the no-waste totals measured for these runs.

Exact domain weights for these four arms are in Validated mixture weights below.

---

## Proxy pilot (DataDecide-60M)

24 designed mixtures over the same 7 domains, each trained with a **DataDecide-60M** proxy and scored on OLMo-ladder task-loss (bits-per-byte). Surrogates are fit on **Chinchilla-extrapolated** family losses (step 5806, tokens/param = 20).

### Proxy architecture

- Hidden size 384, 16 layers, 12 heads, MLP ratio 8, sequence length 2048
- Global batch 96 sequences; learning rate $5.8\times10^{-3}$
- Tokenizer: dolma2 (100,352 embedding rows); untied LM head
- Body params 37.8M; **non-embedding params 57.1M** (tokens/param denominator); total ~114.8M with dolma2 vocab
- **Budget:** tokens/param = 5 → **285M tokens / 1451 steps** per mixture
- **Pilot FLOPs** (Chinchilla $C\approx 6ND$): $\approx 9.8\times10^{16}$ per mix → **$\approx 2.3\times10^{18}$** for 24

### Evaluation and Chinchilla targets

In-run curves use six **ARC + MMLU** val families (subset batches). Step-laws fit **in-run curve points only** (steps 120–1440); a post-hoc full eval at step 1451 is kept for reporting but **not** used in the step law. Curves are extrapolated to Chinchilla step **5806** (tpp = 20). Mixing-law / LightGBM targets are those six extrapolated family losses.

Observed Chinchilla-target range across the 24 pilots:

| family | min | max | std |
|--------|----:|----:|----:|
| arc_challenge | 1.5043 | 1.7831 | 0.0790 |
| arc_easy | 1.8108 | 2.1452 | 0.0878 |
| mmlu_humanities | 1.6810 | 1.9250 | 0.0678 |
| mmlu_other | 2.2812 | 2.7868 | 0.1331 |
| mmlu_social_sciences | 1.2668 | 1.4775 | 0.0608 |
| mmlu_stem | 2.1722 | 2.4410 | 0.0805 |

---

## Mixture sampling

Mixtures follow **Algorithm 2** (double-diminishing grid) from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952):

1. Start from the **Data Mixing Laws paper** mixture (Pilot 01).
2. Compute per-domain **$r_{\max}$** from OLMoHQ pool availability at a 30B target corpus.
3. Sample a double-diminishing grid with step **$\delta = 0.05$** and **seed 42**.
4. Apply Algorithm 2 feasibility constraints from Ye et al. (availability-aware simplex sampling; wiki-ablation tags may zero wiki).
5. Inject three mid–high DCLM points at 50% / 55% / 60% DCLM that the coarse grid cannot reach with all domains positive.

Result: **24 designed probe points** (not uniform random simplex samples).

| Tag | Meaning |
|-----|---------|
| base | Data Mixing Laws paper reference mixture (Pilot 01) |
| C0-wiki0 | Wiki ablation (wiki = 0) |
| C0-dclm0 | DCLM ablation (dclm = 0) |
| C1-dclm50 / 55 / 60 | Injected high-DCLM points |
| C1 | Standard grid points |

### $r_{\max}$ at 30B target

| domain | $r_{\max}$ |
|--------|----------:|
| dclm | 0.9533 |
| arxiv | 0.6933 |
| starcoder | 0.6767 |
| pes2o | 0.8767 |
| open-web-math | 0.4067 |
| algebraic-stack | 0.3933 |
| wiki | 0.1220 |

### Pilot mixture domain weights

Weights sum to 1.

| mix | tag | dclm | arxiv | starcoder | pes2o | open-web-math | alg-stack | wiki |
|-----|-----|-----:|------:|----------:|------:|--------------:|----------:|-----:|
| Pilot 01 | base | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| Pilot 02 | C0-wiki0 | 0.059 | 0.041 | 0.650 | 0.106 | 0.100 | 0.044 | 0.000 |
| Pilot 03 | C0-wiki0 | 0.059 | 0.650 | 0.041 | 0.106 | 0.100 | 0.044 | 0.000 |
| Pilot 04 | C0-dclm0 | 0.000 | 0.041 | 0.041 | 0.425 | 0.400 | 0.044 | 0.050 |
| Pilot 05 | C1-dclm50 | 0.500 | 0.200 | 0.113 | 0.075 | 0.051 | 0.049 | 0.013 |
| Pilot 06 | C1-dclm55 | 0.550 | 0.180 | 0.101 | 0.068 | 0.046 | 0.044 | 0.011 |
| Pilot 07 | C1-dclm60 | 0.600 | 0.160 | 0.090 | 0.060 | 0.041 | 0.039 | 0.010 |
| Pilot 08 | C1 | 0.119 | 0.041 | 0.041 | 0.027 | 0.400 | 0.350 | 0.023 |
| Pilot 09 | C1 | 0.030 | 0.163 | 0.325 | 0.027 | 0.050 | 0.350 | 0.056 |
| Pilot 10 | C1 | 0.059 | 0.041 | 0.041 | 0.425 | 0.050 | 0.350 | 0.034 |
| Pilot 11 | C1 | 0.030 | 0.163 | 0.325 | 0.027 | 0.400 | 0.044 | 0.013 |
| Pilot 12 | C1 | 0.030 | 0.325 | 0.041 | 0.425 | 0.050 | 0.044 | 0.086 |
| Pilot 13 | C1 | 0.059 | 0.041 | 0.325 | 0.425 | 0.100 | 0.044 | 0.006 |
| Pilot 14 | C1 | 0.059 | 0.325 | 0.081 | 0.053 | 0.200 | 0.175 | 0.106 |
| Pilot 15 | C1 | 0.030 | 0.325 | 0.325 | 0.212 | 0.050 | 0.044 | 0.014 |
| Pilot 16 | C1 | 0.237 | 0.163 | 0.041 | 0.106 | 0.400 | 0.044 | 0.009 |
| Pilot 17 | C1 | 0.475 | 0.041 | 0.041 | 0.027 | 0.050 | 0.350 | 0.017 |
| Pilot 18 | C1 | 0.237 | 0.041 | 0.041 | 0.425 | 0.100 | 0.044 | 0.113 |
| Pilot 19 | C1 | 0.237 | 0.041 | 0.325 | 0.106 | 0.100 | 0.087 | 0.103 |
| Pilot 20 | C1 | 0.059 | 0.081 | 0.163 | 0.212 | 0.200 | 0.175 | 0.109 |
| Pilot 21 | C1 | 0.237 | 0.041 | 0.163 | 0.053 | 0.050 | 0.350 | 0.106 |
| Pilot 22 | C1 | 0.237 | 0.163 | 0.041 | 0.212 | 0.050 | 0.175 | 0.122 |
| Pilot 23 | C1 | 0.475 | 0.041 | 0.325 | 0.053 | 0.050 | 0.044 | 0.013 |
| Pilot 24 | C1 | 0.119 | 0.325 | 0.041 | 0.106 | 0.050 | 0.350 | 0.009 |

---

## Surrogate fits

Two surrogates map 7-domain weights → six Chinchilla-extrapolated losses.

| | Mixing law | LightGBM |
|---|---|---|
| Model | Regularized [Ye et al.](https://arxiv.org/abs/2403.16952) law | One gradient-boosted tree per family ([RegMix](https://arxiv.org/abs/2407.01492)-style) |
| Pilot runs | 24 | 24 |
| Chinchilla step | 5806 | 5806 |

### Parametric mixing law

$$
L_i(r) = c_i + k_i \exp\big(\mathrm{clip}(\sum_j t_{ij} r_j,\,-60,\,60)\big)
$$

More negative $t_{ij}$ means increasing domain $j$ lowers family $i$ loss. Among multi-start solutions within 1.35× best RMSE, pick the most parsimonious under:

| Parameter | Value |
|-----------|-------|
| t_soft | 4.0 |
| t_hard | 8.0 |
| k_ratio_soft | 20.0 |
| lambda_t | 0.02 |
| lambda_k | 0.05 |
| rmse_slack | 1.35 |
| n_starts | 128 |

#### Fitted $c_i$, $k_i$

| family | $c_i$ | $k_i$ | $k$/std | max$\|t\|$ | in-sample RMSE |
|--------|------:|------:|--------:|----------:|---------------:|
| arc_challenge | 1.5043 | 0.0790 | 1.00 | 3.72 | 0.0471 |
| arc_easy | 1.8108 | 0.0878 | 1.00 | 2.16 | 0.0887 |
| mmlu_humanities | 1.6022 | 0.0678 | 1.00 | 1.69 | 0.0163 |
| mmlu_other | 2.1336 | 0.1331 | 1.00 | 1.77 | 0.0951 |
| mmlu_social_sciences | 1.2571 | 0.0608 | 1.00 | 2.77 | 0.0198 |
| mmlu_stem | 2.1427 | 0.0805 | 1.00 | 1.88 | 0.0433 |

#### Skill / transfer matrix $t_{ij}$ (rows = families, columns = domains)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | -3.72 | 2.07 | 1.20 | -0.07 | 0.49 | 1.03 | -1.50 |
| arc_easy | 1.31 | 0.69 | 0.84 | 0.72 | -2.16 | 1.48 | -1.92 |
| mmlu_humanities | -0.67 | 1.63 | 1.69 | 0.72 | 1.63 | 1.61 | -1.27 |
| mmlu_other | 0.19 | 1.77 | 1.22 | 0.70 | 1.41 | 1.58 | -1.23 |
| mmlu_social_sciences | -2.77 | 1.69 | 1.34 | 0.17 | 1.50 | 1.82 | -1.64 |
| mmlu_stem | -0.07 | 1.88 | 1.43 | -1.66 | 1.18 | 1.59 | -0.49 |

### LightGBM

Features = 7 mixture weights; target = per-family Chinchilla loss; predicted macro = mean over families. Hyperparameters chosen by a **144-config LOO grid** minimizing macro LOO RMSE. Optima: **50k random simplex samples + SLSQP polish**.

| Parameter | Value |
|-----------|-------|
| objective | regression |
| metric | rmse |
| verbosity | -1 |
| feature_fraction | 1.0 |
| bagging_fraction | 1.0 |
| seed | 0 |
| num_leaves | 7 |
| max_depth | 3 |
| min_data_in_leaf | 2 |
| learning_rate | 0.05 |
| lambda_l2 | 0.1 |
| num_boost_round | 100 |

LOO grid: hand-picked default macro LOO RMSE 0.0439 → selected 0.0366
(num_leaves 7, max_depth 3, min_data_in_leaf 2, lr 0.05, λ₂ 0.1, rounds 100).

#### LightGBM feature importance (gain)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 0.75 | 0.06 | 0.02 | 0.15 | 0.32 | 0.05 | 0.18 |
| arc_easy | 0.27 | 0.32 | 0.26 | 0.08 | 0.53 | 0.19 | 0.19 |
| mmlu_humanities | 0.85 | 0.00 | 0.05 | 0.06 | 0.04 | 0.01 | 0.12 |
| mmlu_other | 1.76 | 0.25 | 0.59 | 0.96 | 0.30 | 0.06 | 0.40 |
| mmlu_social_sciences | 0.67 | 0.02 | 0.03 | 0.14 | 0.01 | 0.02 | 0.01 |
| mmlu_stem | 0.22 | 0.06 | 0.24 | 0.71 | 0.02 | 0.01 | 0.31 |

### Leave-one-out cross-validation

| Metric | Mixing law | LightGBM |
|--------|-----------:|---------:|
| Mean LOO RMSE | 0.0766 | 0.0751 |
| Mean LOO RMSE / std | 84.4% | 84.1% |
| Macro LOO RMSE | 0.0443 | 0.0366 |

| family | ML LOO | LGB LOO | ML in-sample | LGB in-sample |
|--------|-------:|--------:|-------------:|--------------:|
| arc_challenge | 0.0774 | 0.0757 | 0.0471 | 0.0079 |
| arc_easy | 0.1269 | 0.1163 | 0.0887 | 0.0159 |
| mmlu_humanities | 0.0271 | 0.0288 | 0.0163 | 0.0028 |
| mmlu_other | 0.1432 | 0.1306 | 0.0951 | 0.0111 |
| mmlu_social_sciences | 0.0269 | 0.0317 | 0.0198 | 0.0037 |
| mmlu_stem | 0.0581 | 0.0674 | 0.0433 | 0.0090 |

### Mixture optima and near-optimal candidates

Surrogate optima plus nearby mixtures (within +0.04 bpb of that model’s optimum and ≥ 8 pp ($L_\infty$) from the optimum). None exactly match a pilot point.

**Mixing law**

| candidate | pred macro | max_w | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----------|------------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| optimum | 1.7965 | 0.568 | 0.568 | — | — | 0.097 | 0.035 | — | 0.300 |
| near-opt 1 | 1.7995 | 0.440 | 0.440 | 0.011 | 0.011 | 0.207 | 0.013 | 0.018 | 0.300 |
| near-opt 2 | 1.8016 | 0.435 | 0.435 | 0.018 | 0.011 | 0.134 | 0.069 | 0.032 | 0.300 |
| near-opt 3 | 1.8022 | 0.354 | 0.354 | 0.020 | 0.012 | 0.258 | 0.027 | 0.029 | 0.300 |
| near-opt 4 | 1.8024 | 0.440 | 0.440 | 0.012 | 0.012 | 0.043 | 0.181 | 0.012 | 0.300 |
| near-opt 5 | 1.8026 | 0.346 | 0.346 | 0.012 | 0.012 | 0.180 | 0.137 | 0.014 | 0.300 |
| near-opt 6 | 1.8032 | 0.309 | 0.274 | 0.011 | 0.021 | 0.309 | 0.073 | 0.011 | 0.300 |
| near-opt 7 | 1.8036 | 0.397 | 0.223 | 0.010 | 0.010 | 0.397 | 0.051 | 0.010 | 0.300 |
| near-opt 8 | 1.8041 | 0.638 | 0.638 | 0.024 | 0.036 | 0.010 | 0.016 | 0.010 | 0.267 |

**LightGBM**

| candidate | pred macro | max_w | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----------|------------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| optimum | 1.8335 | 0.553 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| near-opt 1 | 1.8335 | 0.558 | 0.558 | 0.199 | 0.073 | 0.083 | 0.033 | 0.042 | 0.012 |
| near-opt 2 | 1.8380 | 0.491 | 0.491 | 0.167 | 0.071 | 0.087 | 0.046 | 0.073 | 0.064 |
| near-opt 3 | 1.8383 | 0.495 | 0.495 | 0.167 | 0.072 | 0.179 | 0.015 | 0.055 | 0.017 |
| near-opt 4 | 1.8391 | 0.502 | 0.502 | 0.172 | 0.095 | 0.128 | 0.048 | 0.022 | 0.033 |
| near-opt 5 | 1.8397 | 0.505 | 0.505 | 0.190 | 0.125 | 0.080 | 0.021 | 0.024 | 0.055 |
| near-opt 6 | 1.8398 | 0.520 | 0.520 | 0.188 | 0.083 | 0.078 | 0.049 | 0.044 | 0.037 |
| near-opt 7 | 1.8398 | 0.583 | 0.583 | 0.171 | 0.082 | 0.069 | 0.029 | 0.048 | 0.018 |
| near-opt 8 | 1.8410 | 0.377 | 0.182 | 0.182 | 0.076 | 0.377 | 0.020 | 0.037 | 0.127 |

### Random-simplex plausibility

1000 mixtures ~ Dirichlet(1,…,1) on the 7-simplex (seed 42). Pilot observed macro range: **2.0444 – 2.2888 bpb**.

| Metric | Mixing law | LightGBM |
|--------|-----------:|---------:|
| Macro min | 1.8526 | 1.8408 |
| Macro p50 | 1.9131 | 1.9297 |
| Macro p95 | 2.0166 | 2.0022 |
| Macro p99 | 2.0602 | 2.0217 |
| Macro max | 2.1395 | 2.0339 |
| Macro mean ± std | 1.9231 ± 0.0467 | 1.9393 ± 0.0441 |
| % inside pilot macro range | 2.3% | 0.0% |
| Mixtures with macro > 3 bpb | 0 | 0 |
| Mixtures with macro > 5 bpb | 0 | 0 |

Off-hull predictions stay bounded; claimed gains of the surrogate optima over the best measured pilot are **not distinguishable from LOO error** at 60M scale — hence the 370M validation.

### Pilot mixtures ranked by predicted macro (top 12)

| ML rank | LGB rank | mix | tag | ML pred | LGB pred | measured curve-6 |
|--------:|---------:|-----|-----|--------:|---------:|-----------------:|
| 1 | 4 | Pilot 18 | C1 | 1.8368 | 1.8700 | 2.1089 |
| 2 | 2 | Pilot 07 | C1-dclm60 | 1.8618 | 1.8451 | 2.0444 |
| 3 | 7 | Pilot 22 | C1 | 1.8666 | 1.8810 | 2.0899 |
| 4 | 1 | Pilot 06 | C1-dclm55 | 1.8668 | 1.8370 | 2.0963 |
| 5 | 5 | Pilot 05 | C1-dclm50 | 1.8726 | 1.8714 | 2.0830 |
| 6 | 14 | Pilot 23 | C1 | 1.8733 | 1.9107 | 2.0981 |
| 7 | 17 | Pilot 19 | C1 | 1.8752 | 1.9628 | 2.1456 |
| 8 | 13 | Pilot 17 | C1 | 1.8897 | 1.9093 | 2.0835 |
| 9 | 10 | Pilot 01 | base | 1.8913 | 1.8951 | 2.0727 |
| 10 | 8 | Pilot 04 | C0-dclm0 | 1.8964 | 1.8888 | 2.1174 |
| 11 | 6 | Pilot 21 | C1 | 1.8969 | 1.8764 | 2.1364 |
| 12 | 3 | Pilot 16 | C1 | 1.8992 | 1.8527 | 2.1125 |

---

## Validated mixture weights (370M)

Recipe domain weights for the four 370M arms that were trained:

| arm | source | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|-----|--------|---:|---:|---:|---:|---:|---:|---:|
| OLMo Mix 1124 | reference | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| Data Mixing Laws paper | pilot | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| MixLaw | mixing-law | 0.568 | 0.000 | 0.000 | 0.097 | 0.035 | 0.000 | 0.300 |
| LightGBM | lightgbm | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |

**What each validated arm manipulates**

- **OLMo Mix 1124 (control)** — hold mix fixed at the natural OLMo Mix 1124 corpus proportions (very DCLM-heavy).
- **Data Mixing Laws paper** — hold domain mix fixed at the Data Mixing Laws paper proportions (Pilot 01).
- **MixLaw** — train the parametric mixing-law optimum. Method from [Ye et al., Data Mixing Laws](https://arxiv.org/abs/2403.16952).
- **LightGBM** — train the LightGBM mixing-law optimum. Method from [Liu et al., RegMix](https://arxiv.org/abs/2407.01492) (regression / tree surrogate over mixture probes).

---

## 370M evaluation and uncertainty

Task-loss is evaluated periodically on the shared 20-label suite. Final performance uses a **power-law residual bootstrap**:

1. Fit $y = a + b / \mathrm{step}^{\alpha}$ on all eval points with step ≥ 1000 ($\alpha$ on a grid in $[0.05, 3]$).
2. Take the fitted value at the final step as the point estimate.
3. Residual bootstrap (10k draws) for a 95% CI on that fitted final.
4. Pairwise $\Delta = \mathrm{arm} - \mathrm{control}$ from independent bootstraps (control = OLMo Mix 1124); two-sided $p$ reported only when the arm beats control.

---

## 370M results

### Fitted final macro task-loss (bpb)

| Arm | Fitted final | Observed | 95% CI | vs control |
|-----|-------------:|---------:|--------|------------|
| **MixLaw** | **1.6062** | 1.6048 | [1.6027, 1.6097] | $p_{\mathrm{two}} < 0.0001$ |
| **LightGBM** | **1.6087** | 1.6077 | [1.6062, 1.6111] | $p_{\mathrm{two}} < 0.0001$ |
| OLMo Mix 1124 (control) | 1.6329 | 1.6370 | [1.6284, 1.6368] | — |
| Data Mixing Laws paper | 1.6518 | 1.6518 | [1.6483, 1.6552] | — |

Lower is better. Both fitted mixtures beat the OLMo Mix 1124 control. Data Mixing Laws paper does not.

### Takeaways

1. **Mixing-law search works at this scale.** Short-run fits produced mixtures that clearly dominate the OLMo Mix 1124 control under a matched one-epoch budget.
2. **MixLaw ≈ LightGBM**, with a small edge to the MixLaw optimum.
3. **Data Mixing Laws paper proportions underperform the control.** That fixed paper mix finishes worse than OLMo Mix 1124.
4. **Cost.** Four full arms 188.74 A100-hours and $\approx 1.05\times10^{20}$ FLOPs, plus $\approx 2.3\times10^{18}$ FLOPs for the 60M pilot grid.

---

## Conclusions

Under the P1 one-epoch 370M contract, **data mixture is a high-leverage lever**: mixing-law optimization yields ~0.025 bpb absolute macro task-loss improvement over the OLMo Mix 1124 control, with tight CIs and decisive bootstrap significance. Among levers in this campaign, MixLaw is the clearest positive result.
