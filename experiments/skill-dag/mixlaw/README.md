# Skill-DAG mixing-law pilot

24-mixture probe over 7 OLMoHQ domains, trained with **DataDecide-60M** proxy models,
evaluated on OLMo-ladder **task loss** (bits-per-byte). Fitted with a regularized
Ye et al. mixing law and a per-family **LightGBM** model on **Chinchilla-extrapolated**
targets (step 5806, tpp = 20).

---

## Proxy model architecture

Each pilot run trains the exact **DataDecide 60M** geometry from
[allenai/DataDecide-dolma1_7-60M](https://huggingface.co/allenai/DataDecide-dolma1_7-60M)
(Ye et al., arXiv:2403.16952), with dolma2 tokenization to match the olmohq corpus.

- **Hidden size (`d_model`)**: 384
- **Layers**: 16
- **Heads**: 12
- **MLP ratio**: 8
- **Sequence length**: 2048
- **Global batch**: 96 sequences
- **Learning rate**: 5.8e-3
- **Tokenizer**: dolma2 (`allenai/dolma2-tokenizer`)
- **Embedding rows**: 100,352 (vocab 100,278 + specials)
- **LM head**: Untied
- **Body params**: 37.8M
- **Non-embedding params (tokens/param denominator)**: **57.1M**
- **Total params (this run)**: **114.8M**

**Pilot budget:** tokens/param = 5 → **285M tokens / 1451 steps** per mixture (~30 min on one B200).

**Evaluation:** OLMo-ladder task loss — bits-per-byte on six in-run **ARC + MMLU**
curve families (val splits). Final eval and mixing-law fits use only these six.

---

## Mixture sampling (probe domain weights)

Mixtures are defined in `mixtures.json` using **Algorithm 2** (double-diminishing grid)
from Ye et al. (arXiv:2403.16952):

1. Set **base mixture** to RegMix proportions (mix01).
2. Compute **r_max** per domain from olmohq pool availability at a 30B target corpus.
3. Sample a **double-diminishing grid** with step **δ = 0.05** and **seed 42**.
4. Apply constraints: wiki floor 0.5% (except wiki-ablation tags), dclm cap 60%,
   other domains cap 70%.
5. **Inject 3 extra points** at 50% / 55% / 60% DCLM (tags `C1-dclm50/55/60`) —
   mid-high DCLM weights the coarse grid cannot reach with all domains positive.

Result: **24 designed probe points** (not random uniform samples). Tags:

| Tag | Meaning |
|-----|---------|
| `base` | RegMix reference mixture |
| `C0-wiki0` | Wiki ablation (wiki = 0) |
| `C0-dclm0` | DCLM ablation (dclm = 0) |
| `C1-dclm50/55/60` | Injected high-DCLM points |
| `C1` | Standard grid points |

Mixture proportions are realized as **exact per-domain sequence counts** on disk
(largest-remainder allocation); one epoch shuffles non-overlapping 2048-token chunks.

---

## Pilot mixture domain weights

Columns follow `domain_order` in `mixtures.json`. Weights sum to 1.

| mix | tag | dclm | arxiv | starcoder | pes2o | open-web-math | alg-stack | wiki |
|-----|-----|------|-------|-----------|-------|---------------|-----------|------|
| mix01 | base | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| mix02 | C0-wiki0 | 0.059 | 0.041 | 0.650 | 0.106 | 0.100 | 0.044 | 0.000 |
| mix03 | C0-wiki0 | 0.059 | 0.650 | 0.041 | 0.106 | 0.100 | 0.044 | 0.000 |
| mix04 | C0-dclm0 | 0.000 | 0.041 | 0.041 | 0.425 | 0.400 | 0.044 | 0.050 |
| mix05 | C1-dclm50 | 0.500 | 0.200 | 0.113 | 0.075 | 0.051 | 0.049 | 0.013 |
| mix06 | C1-dclm55 | 0.550 | 0.180 | 0.101 | 0.068 | 0.046 | 0.044 | 0.011 |
| mix07 | C1-dclm60 | 0.600 | 0.160 | 0.090 | 0.060 | 0.041 | 0.039 | 0.010 |
| mix08 | C1 | 0.119 | 0.041 | 0.041 | 0.027 | 0.400 | 0.350 | 0.023 |
| mix09 | C1 | 0.030 | 0.163 | 0.325 | 0.027 | 0.050 | 0.350 | 0.056 |
| mix10 | C1 | 0.059 | 0.041 | 0.041 | 0.425 | 0.050 | 0.350 | 0.034 |
| mix11 | C1 | 0.030 | 0.163 | 0.325 | 0.027 | 0.400 | 0.044 | 0.013 |
| mix12 | C1 | 0.030 | 0.325 | 0.041 | 0.425 | 0.050 | 0.044 | 0.086 |
| mix13 | C1 | 0.059 | 0.041 | 0.325 | 0.425 | 0.100 | 0.044 | 0.006 |
| mix14 | C1 | 0.059 | 0.325 | 0.081 | 0.053 | 0.200 | 0.175 | 0.106 |
| mix15 | C1 | 0.030 | 0.325 | 0.325 | 0.212 | 0.050 | 0.044 | 0.014 |
| mix16 | C1 | 0.237 | 0.163 | 0.041 | 0.106 | 0.400 | 0.044 | 0.009 |
| mix17 | C1 | 0.475 | 0.041 | 0.041 | 0.027 | 0.050 | 0.350 | 0.017 |
| mix18 | C1 | 0.237 | 0.041 | 0.041 | 0.425 | 0.100 | 0.044 | 0.113 |
| mix19 | C1 | 0.237 | 0.041 | 0.325 | 0.106 | 0.100 | 0.087 | 0.103 |
| mix20 | C1 | 0.059 | 0.081 | 0.163 | 0.212 | 0.200 | 0.175 | 0.109 |
| mix21 | C1 | 0.237 | 0.041 | 0.163 | 0.053 | 0.050 | 0.350 | 0.106 |
| mix22 | C1 | 0.237 | 0.163 | 0.041 | 0.212 | 0.050 | 0.175 | 0.122 |
| mix23 | C1 | 0.475 | 0.041 | 0.325 | 0.053 | 0.050 | 0.044 | 0.013 |
| mix24 | C1 | 0.119 | 0.325 | 0.041 | 0.106 | 0.050 | 0.350 | 0.009 |

---

## Script inventory

| Script | Role |
|--------|------|
| `prepare_data.sh` | One-time pipeline: fetch olmohq shards, tokenize working pool, build 24 mixture slices |
| `select_and_fetch_shards.py` | Random shard draw per domain from S3 inventory (overshoot-aware) |
| `tokenize_working_pool.py` | Tokenize fetched raw shards into per-domain memmaps |
| `build_mixture_data.py` | Plan + materialize per-mixture random subsamples from the working pool |
| `budget_calculator.py` | GPU-hour / token budget vs olmohq availability |
| `train_datadecide_60m.py` | Train one mixture (DataDecide 60M, in-run curve eval) |
| `run_mixture.sh` | Single-GPU worker: train + full eval for one `mixNN` |
| `eval_task_loss.py` | Task-loss eval on the six curve labels (or `--full-suite` for all 20) |
| `run_task_loss_eval.py` | Batch re-eval helper for finished checkpoints |
| `extrapolate_chinchilla.py` | Extrapolate in-run curves to Chinchilla step (tpp = 20) |
| `fit_mixing_law.py` | Baseline mixing-law fit + simplex optimization |
| `fit_chinchilla.py` | Regularized fit on Chinchilla targets + near-optimal sampling |
| `fit_lightgbm_chinchilla.py` | LightGBM fit on Chinchilla targets + near-optimal sampling |
| `loo_chinchilla.py` | Leave-one-out cross-validation |
| `sample_random_simplex.py` | Random-simplex plausibility check |
| `preflight_checks.py` | Sanity checks on data layout, step law, and simplex optimizer |
| `mixlaw_common.py` | Shared constants (domains, DataDecide geometry, task labels) |
| `mixtures.json` | 24 probe mixtures (Algorithm 2 grid + injected points) |
| `reoptimize_constraints.py` | Re-run optima / near-optimal sampling on saved fits |
| `write_validation_mixtures.py` | Emit 370M validation recipe (`validation_mixtures_10b.json`) |
| `build_working_pool_from_shards.py` | Peak olmohq pool for 370M validation slices |
| `finalize_mixlaw_upload.py` | Publish validation corpora to `s3://edullm-datasets/mixlaw/` |
| `validation_mixtures_10b.json` | Eight 10B-mix recipe for 370M scale-up |
| `generate_readme.py` | Regenerate this README from JSON artifacts |

---

## S3 artifact paths

| Artifact | Location | Notes |
|----------|----------|-------|
| **Training corpus** | `s3://edullm-datasets/olmo100b/olmo-mix-1124-30b` | Raw olmohq json.gz shards; only a subset is fetched locally |
| **Pilot results** | `s3://edullm-checkpoints/token-selection/mixlaw-pilot/mix01` … `mix24` | Logs + progress only (no weight checkpoints on S3) |
| **370M validation corpora** | `s3://edullm-datasets/mixlaw/` | 10B tokens × 8 mixes; `READY` + `validation_mixtures_10b.json` |
| Per-mix corpus | `…/mixlaw/mixes/<run_name>/` | Tokenized slices; mix01 copied from regmix-10b |
| Per-mix progress | `…/mixNN/progress/` | `run_meta.json`, `task_loss_final.json`, `task_loss.jsonl` |
| Per-mix logs | `…/mixNN/logs/` | `train.log`, `eval.log` |
| **Local mirror** | `pilot_runs/mixNN/progress/` | Checked-in progress JSON from the completed pilot |

Set `RESULTS_S3=s3://edullm-checkpoints/token-selection/mixlaw-pilot` in `run_mixture.sh`
to sync progress/logs after each mix finishes. Weight checkpoints stay on local NVMe only.


---

## Surrogate fits on Chinchilla targets

Two surrogates map 7-domain mixture weights → six Chinchilla-extrapolated curve losses
(step 5806, tpp = 20) from 24 pilot mixtures:

| | Mixing law | LightGBM |
|---|---|---|
| Model | Regularized Ye et al. law | One gradient-boosted tree regressor per family |
| Artifact | `mixlaw_fit_chinchilla.json` | `mixlaw_fit_lightgbm_chinchilla.json` |
| Fit script | `fit_chinchilla.py` | `fit_lightgbm_chinchilla.py` |
| LOO artifact | `mixlaw_fit_chinchilla_loo.json` | in-fit LOO per family |
| Pilot runs | 24 mixtures | 24 mixtures |
| Chinchilla step | 5806 | 5806 |

### Mixing law

```
L_i(r) = c_i + k_i * exp( clip( sum_j t_ij * r_j, -60, 60 ) )
```

- `r` = mixture weights on the 7-domain simplex (sum to 1)
- More negative `t_ij` → increasing domain j lowers task family i loss

**Regularization** (among multi-start solutions within 1.35× best RMSE, pick parsimonious):

| Parameter | Value |
|-----------|-------|
| `t_soft` | 4.0 |
| `t_hard` | 8.0 |
| `k_ratio_soft` | 20.0 |
| `lambda_t` | 0.02 |
| `lambda_k` | 0.05 |
| `rmse_slack` | 1.35 |
| `n_starts` | 128 |

### LightGBM

Features = 7 mixture weights; target = per-family Chinchilla loss. Predicted macro = mean
over families. Hyperparameters chosen by a **small LOO grid search** (144 configs) minimizing
macro LOO RMSE. Mixture optima use **50k random simplex samples + SLSQP polish** (non-convex
surrogate); constrained optima seed the uncapped search.

| Parameter | Value |
|-----------|-------|
| `objective` | regression |
| `metric` | rmse |
| `verbosity` | -1 |
| `feature_fraction` | 1.0 |
| `bagging_fraction` | 1.0 |
| `seed` | 0 |
| `num_leaves` | 7 |
| `max_depth` | 3 |
| `min_data_in_leaf` | 2 |
| `learning_rate` | 0.05 |
| `lambda_l2` | 0.1 |
| `num_boost_round` | 100 |

**Hyperparameter search** (objective: minimize macro LOO RMSE):

| | Baseline (hand-picked) | Selected (LOO grid) |
|---|---|---|
| mean LOO RMSE | 0.0735 | 0.0746 |
| macro LOO RMSE | 0.0454 | 0.0421 |
| `num_leaves` | 7 | 7 |
| `max_depth` | 3 | 3 |
| `min_data_in_leaf` | 3 | 2 |
| `learning_rate` | 0.05 | 0.05 |
| `lambda_l2` | 1.0 | 0.1 |
| `num_boost_round` | 200 | 100 |

### Observed Chinchilla targets (fit y)

Shared across both fits — per-family Chinchilla-extrapolated loss at step 5806.

| family | min | max | std |
|--------|-----|-----|-----|
| arc_challenge | 1.5036 | 1.7731 | 0.0819 |
| arc_easy | 1.7400 | 2.0903 | 0.0783 |
| mmlu_humanities | 1.5853 | 1.8464 | 0.0712 |
| mmlu_other | 2.2227 | 2.6473 | 0.1154 |
| mmlu_social_sciences | 1.3983 | 1.6531 | 0.0634 |
| mmlu_stem | 2.3975 | 2.6974 | 0.0872 |

### Leave-one-out cross-validation

| Metric | Mixing law | LightGBM |
|--------|------------|----------|
| Mean LOO RMSE | 0.0736 | 0.0746 |
| Mean LOO RMSE / std | 85.8% | 88.2% |
| Macro LOO RMSE | 0.0399 | 0.0421 |

| family | ML LOO | ML % std | LGB LOO | LGB % std | ML in-sample | LGB in-sample |
|--------|--------|----------|---------|-----------|--------------|---------------|
| arc_challenge | 0.0766 | 93.4% | 0.0786 | 96.0% | 0.0491 | 0.0072 |
| arc_easy | 0.1178 | 150.4% | 0.1041 | 132.9% | 0.0734 | 0.0134 |
| mmlu_humanities | 0.0271 | 38.1% | 0.0352 | 49.4% | 0.0164 | 0.0027 |
| mmlu_other | 0.1185 | 102.7% | 0.0966 | 83.7% | 0.0708 | 0.0082 |
| mmlu_social_sciences | 0.0324 | 51.2% | 0.0329 | 52.0% | 0.0238 | 0.0045 |
| mmlu_stem | 0.0689 | 79.0% | 0.1003 | 114.9% | 0.0616 | 0.0128 |

### Mixing-law parameters (c_i, k_i, t_ij)

| family | c_i | k_i | k/std | max\|t\| | in-sample RMSE |
|--------|-----|-----|-------|---------|----------------|
| arc_challenge | 1.5036 | 0.0819 | 1.00 | 3.62 | 0.0491 |
| arc_easy | 1.7400 | 0.0783 | 1.00 | 1.36 | 0.0734 |
| mmlu_humanities | 1.5805 | 0.0712 | 1.00 | 2.58 | 0.0164 |
| mmlu_other | 2.2010 | 0.1154 | 1.00 | 2.65 | 0.0708 |
| mmlu_social_sciences | 1.3930 | 0.0634 | 1.00 | 1.97 | 0.0238 |
| mmlu_stem | 2.3975 | 0.0872 | 1.00 | 2.84 | 0.0616 |

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---|---|---|---|---|---|---|
| arc_challenge | -3.62 | 1.92 | 1.06 | 0.50 | 0.16 | 0.97 | 0.13 |
| arc_easy | 1.36 | 0.53 | 0.52 | -0.47 | -0.73 | 1.27 | 0.95 |
| mmlu_humanities | -2.58 | 1.52 | 1.45 | 0.10 | 1.44 | 1.88 | -2.58 |
| mmlu_other | -1.35 | 2.04 | 1.66 | -0.45 | 0.43 | 0.84 | -2.65 |
| mmlu_social_sciences | -1.97 | 1.47 | 1.25 | 0.09 | 1.96 | 1.14 | -1.15 |
| mmlu_stem | -0.96 | 1.53 | 1.44 | -0.84 | 0.92 | 1.95 | -2.84 |

### LightGBM feature importance (gain)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---|---|---|---|---|---|---|
| arc_challenge | 0.78 | 0.11 | 0.04 | 0.25 | 0.26 | 0.09 | 0.11 |
| arc_easy | 0.09 | 0.36 | 0.24 | 0.40 | 0.19 | 0.14 | 0.04 |
| mmlu_humanities | 1.01 | 0.01 | 0.00 | 0.13 | 0.02 | 0.03 | 0.05 |
| mmlu_other | 1.52 | 0.30 | 0.07 | 0.76 | 0.26 | 0.30 | 0.05 |
| mmlu_social_sciences | 0.69 | 0.01 | 0.02 | 0.18 | 0.03 | 0.04 | 0.03 |
| mmlu_stem | 0.29 | 0.04 | 0.53 | 0.26 | 0.24 | 0.25 | 0.22 |

### Mixture optima and near-optimal candidates

Constrained optima (`uncapped`, `pilot_caps`, `min1pct`) plus sampled near-optimal
mixtures. Near-opt rows: **≥ 1%** on every domain, within **+0.04 bpb** of each model's
uncapped optimum (mixing law 1.8452, LightGBM 1.8513), and **≥ 8 pp**
(L∞) away from every optimum row above. None exactly match a pilot point.

**Mixing law**

| label | pred macro | max_w | dclm | arxiv | star | pes2o | owm | alg | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 1.8452 | 0.390 | 0.340 | — | — | 0.390 | — | — | 0.270 |
| pilot_caps | 1.8452 | 0.390 | 0.340 | — | — | 0.390 | — | — | 0.270 |
| min1pct | 1.8471 | 0.341 | 0.332 | 0.010 | 0.010 | 0.341 | 0.010 | 0.010 | 0.286 |
| near-opt 1 | 1.8493 | 0.375 | 0.375 | 0.012 | 0.024 | 0.210 | 0.049 | 0.015 | 0.314 |
| near-opt 2 | 1.8494 | 0.467 | 0.467 | 0.013 | 0.019 | 0.267 | 0.022 | 0.016 | 0.196 |
| near-opt 3 | 1.8496 | 0.379 | 0.217 | 0.011 | 0.013 | 0.379 | 0.057 | 0.012 | 0.311 |
| near-opt 4 | 1.8498 | 0.412 | 0.412 | 0.012 | 0.015 | 0.334 | 0.064 | 0.018 | 0.145 |
| near-opt 5 | 1.8499 | 0.463 | 0.307 | 0.016 | 0.017 | 0.140 | 0.040 | 0.017 | 0.463 |
| near-opt 6 | 1.8500 | 0.416 | 0.275 | 0.015 | 0.010 | 0.148 | 0.115 | 0.020 | 0.416 |
| near-opt 7 | 1.8502 | 0.375 | 0.339 | 0.034 | 0.013 | 0.189 | 0.030 | 0.020 | 0.375 |
| near-opt 8 | 1.8502 | 0.371 | 0.236 | 0.011 | 0.028 | 0.264 | 0.070 | 0.019 | 0.371 |


**LightGBM**

| label | pred macro | max_w | dclm | arxiv | star | pes2o | owm | alg | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 1.8513 | 0.606 | 0.606 | 0.222 | 0.042 | 0.074 | 0.041 | 0.014 | — |
| pilot_caps | 1.8536 | 0.538 | 0.538 | 0.229 | 0.074 | 0.060 | 0.046 | 0.031 | 0.022 |
| min1pct | 1.8536 | 0.511 | 0.511 | 0.273 | 0.063 | 0.069 | 0.035 | 0.013 | 0.035 |
| near-opt 1 | 1.8570 | 0.525 | 0.525 | 0.178 | 0.089 | 0.058 | 0.022 | 0.019 | 0.109 |
| near-opt 2 | 1.8572 | 0.490 | 0.490 | 0.182 | 0.073 | 0.148 | 0.044 | 0.026 | 0.037 |
| near-opt 3 | 1.8595 | 0.723 | 0.723 | 0.079 | 0.088 | 0.068 | 0.020 | 0.012 | 0.011 |
| near-opt 4 | 1.8612 | 0.571 | 0.571 | 0.097 | 0.098 | 0.149 | 0.031 | 0.043 | 0.011 |
| near-opt 5 | 1.8617 | 0.331 | 0.331 | 0.234 | 0.134 | 0.064 | 0.041 | 0.021 | 0.176 |
| near-opt 6 | 1.8617 | 0.399 | 0.399 | 0.276 | 0.070 | 0.068 | 0.042 | 0.023 | 0.121 |
| near-opt 7 | 1.8619 | 0.359 | 0.359 | 0.087 | 0.099 | 0.328 | 0.030 | 0.014 | 0.084 |
| near-opt 8 | 1.8621 | 0.489 | 0.489 | 0.226 | 0.014 | 0.088 | 0.025 | 0.039 | 0.119 |

### Random-simplex plausibility

**1000** mixtures sampled uniformly on the 7-simplex (Dirichlet(1,…,1), seed 42). Pilot observed macro range: **2.0444 – 2.2888 bpb**.

| Metric | Mixing law | LightGBM |
|--------|------------|----------|
| Macro min | 1.8526 | 1.8634 |
| Macro p50 | 1.9131 | 1.9553 |
| Macro p95 | 2.0166 | 2.0285 |
| Macro p99 | 2.0602 | 2.0464 |
| Macro max | 2.1395 | 2.0629 |
| Macro mean ± std | 1.9231 ± 0.0467 | 1.9597 ± 0.0448 |
| % inside pilot macro range | 2.3% | 1.1% |
| Mixtures with macro > 3 bpb | 0 | 0 |
| Mixtures with macro > 5 bpb | 0 | 0 |
| Any family > 10 bpb | 0 | 0 |
| mmlu_other max | 2.8288 | 2.6386 |

Sources: `mixlaw_random_simplex_plausibility.json` (mixing law); `mixlaw_fit_lightgbm_chinchilla.json` (LightGBM).

### Pilot mixtures ranked by predicted macro

Sorted by mixing-law prediction; LightGBM rank shown for comparison.

| ML rank | LGB rank | mix | tag | ML pred | LGB pred | max_w | measured curve-6 |
|---------|----------|-----|-----|---------|----------|-------|------------------|
| 1 | 9 | mix18 | C1 | 1.8626 | 1.9085 | 0.425 | 2.1089 |
| 2 | 1 | mix07 | C1-dclm60 | 1.8742 | 1.8595 | 0.600 | 2.0444 |
| 3 | 2 | mix06 | C1-dclm55 | 1.8781 | 1.8650 | 0.550 | 2.0963 |
| 4 | 5 | mix05 | C1-dclm50 | 1.8829 | 1.8950 | 0.500 | 2.0830 |
| 5 | 12 | mix23 | C1 | 1.8868 | 1.9164 | 0.475 | 2.0981 |
| 6 | 7 | mix22 | C1 | 1.8885 | 1.9010 | 0.237 | 2.0899 |
| 7 | 8 | mix17 | C1 | 1.8964 | 1.9055 | 0.475 | 2.0835 |
| 8 | 16 | mix19 | C1 | 1.8998 | 1.9840 | 0.325 | 2.1456 |
| 9 | 6 | mix01 | base | 1.9004 | 1.8965 | 0.375 | 2.0727 |
| 10 | 3 | mix16 | C1 | 1.9103 | 1.8740 | 0.400 | 2.1125 |
| 11 | 4 | mix21 | C1 | 1.9153 | 1.8880 | 0.350 | 2.1364 |
| 12 | 11 | mix04 | C0-dclm0 | 1.9191 | 1.9154 | 0.425 | 2.1174 |

### 370M validation plan

Eight mixtures selected for OLMo-370M scale-up: the natural **olmo-mix-1124** corpus mix (~95% DCLM), three pilot anchors (mix01, mix07, mix18), plus min1pct and one near-opt per surrogate (ML near-opt 3, LGB near-opt 5).

| run | source | dclm | arxiv | starcoder | pes2o | open-web-math | alg-stack | wiki |
|-----|--------|---:|---:|---:|---:|---:|---:|---:|
| olmo-mix-1124 | reference | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| mix01 | pilot | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| mix07 | pilot | 0.600 | 0.160 | 0.090 | 0.060 | 0.041 | 0.039 | 0.010 |
| mix18 | pilot | 0.237 | 0.041 | 0.041 | 0.425 | 0.100 | 0.044 | 0.113 |
| ML-min1pct | mixing-law | 0.332 | 0.010 | 0.010 | 0.341 | 0.010 | 0.010 | 0.286 |
| ML-near-opt-3 | mixing-law | 0.217 | 0.011 | 0.013 | 0.379 | 0.057 | 0.012 | 0.311 |
| LGB-min1pct | lightgbm | 0.511 | 0.273 | 0.063 | 0.069 | 0.035 | 0.013 | 0.035 |
| LGB-near-opt-5 | lightgbm | 0.331 | 0.234 | 0.134 | 0.064 | 0.041 | 0.021 | 0.176 |

### Key takeaways

1. **mix07** is the best measured pilot point; both models rank it first or near-first.
2. Mixing-law optima favor **dclm + pes2o**; LightGBM optima favor **dclm + arxiv**.
3. Claimed gains over mix07 (~0.03 bpb for mixing law) are **not distinguishable** from LOO error.
4. Mean LOO RMSE is similar across models; mixing law has slightly lower macro LOO.
5. Off-hull random mixtures stay bounded under both surrogates.

### Reproduce

```bash
cd experiments/skill-dag/mixlaw
py -3 extrapolate_chinchilla.py
py -3 fit_chinchilla.py
py -3 loo_chinchilla.py
py -3 sample_random_simplex.py
py -3 fit_lightgbm_chinchilla.py
py -3 reoptimize_constraints.py --constraints-only --lightgbm
py -3 generate_readme.py   # refresh this file
```

---

## 370M validation corpora (built)

Materialized **10B dolma2 tokens per mixture** for OLMo-370M scale-up. Recipe: `validation_mixtures_10b.json` (canonical copy on S3 at `mixlaw/validation_mixtures_10b.json`). Published **2026-07-29** — `s3://edullm-datasets/mixlaw/READY`.

| Mixture | S3 prefix |
|---------|-----------|
| `olmo-mix-1124` | `mixlaw/mixes/olmo-mix-1124/` |
| `mix01` | `mixlaw/mixes/mix01/` |
| `mix07` | `mixlaw/mixes/mix07/` |
| `mix18` | `mixlaw/mixes/mix18/` |
| `ML-min1pct` | `mixlaw/mixes/ML-min1pct/` |
| `ML-near-opt-3` | `mixlaw/mixes/ML-near-opt-3/` |
| `LGB-min1pct` | `mixlaw/mixes/LGB-min1pct/` |
| `LGB-near-opt-5` | `mixlaw/mixes/LGB-near-opt-5/` |

**mix01 policy:** server-side copy *from* `s3://edullm-datasets/regmix/regmix-10b/` into `mixlaw/mixes/mix01/` only. `regmix/regmix-10b` is **read-only** and must never be modified.

The other seven mixes are sliced offline from a peak-sized olmohq working pool (`olmo100b/olmo-mix-1124-30b` tokenized shards). Slices are contiguous 2048-token chunks with largest-remainder domain allocation (same convention as the 60M pilot).

### Build pipeline

1. **pool** — `build_working_pool_from_shards.py` downloads olmohq `.npy` shards and concatenates a peak-sized pool per domain.
2. **slice** — `build_mixture_data.py plan` + `build` materializes each non-reuse mixture under `slices/<run_name>/`.
3. **upload** — `finalize_mixlaw_upload.py` syncs slices to `mixlaw/mixes/`, copies mix01 from regmix server-side, writes `mixlaw_upload_receipt.json`, and publishes `READY` last.

### Validation build scripts

| Script | Role |
|--------|------|
| `write_validation_mixtures.py` | Emit `validation_mixtures_10b.json` from fit JSON + pilot mixtures |
| `build_working_pool_from_shards.py` | Peak pool from olmohq `tokenized_manifest.json` |
| `build_mixture_data.py` | Plan + materialize per-mixture slices from the working pool |
| `finalize_mixlaw_upload.py` | Upload to `mixlaw/`; mix01 = regmix server-side copy |
| `validation_mixtures_10b.json` | Eight-mix recipe (`reuse_s3` only on mix01) |

Regenerate the recipe locally after refitting:

```bash
cd experiments/skill-dag/mixlaw
py -3 write_validation_mixtures.py
```

