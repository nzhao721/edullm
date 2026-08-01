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

**Chinchilla step-law fitting:** fits use **in-run** `task_loss.jsonl` points only (steps 120–1440,
`eval_subset_batches=4`). The post-hoc `task_loss_final.json` at step 1451 is retained for
reporting but **not** appended to the step law — it uses full eval and often disagrees with the
last in-run point (see `plot_mixlaw_spike_examples.py`). `extrapolate_chinchilla.py` extrapolates
from the jsonl curve to Chinchilla step **5806** (tpp = 20).

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

Training realizes those weights **only** via **`DomainMixtureStream`** over a
shared peak-sized working pool staged from `edullm-data` (domain-stratified
sampling; see `domain_stream.py` / `olmo_domain_stream_patch.py`). Peak
per-domain pool sizes use largest-remainder allocation so every recipe mix
fits. Exact per-mix slice materialization (`build_mixture_data.py`) is
**not supported** for new work — do not re-materialize slices to bit-match
checked-in `pilot_runs/` curves.

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
| `prepare_data.sh` | Write recipe sidecars after edullm-data pool exists (`prepare_mixlaw_pilot_data.py`) |
| `submit_mixlaw_pilot_pool.sh` | FarmShare: stage peak pool from edullm-data into ephemeral RUN_DIR |
| `stage_working_pool_from_edullm_data.py` | Download+concat domain memmaps; write `edullm_data_source.json` |
| `prepare_mixlaw_pilot_data.py` | Per-mix `mix_weights.json` from `mixtures.json` |
| `recipe_data.py` / `domain_stream.py` | Shared recipe sidecars + olmohq domain sampler |
| `olmo_domain_stream_patch.py` | OLMo classic 60M trainer streaming hook |
| `select_and_fetch_shards.py` | Legacy raw-shard draw (pre-edullm-data); not used by training |
| `tokenize_working_pool.py` | Legacy tokenize path; not used by training |
| `build_mixture_data.py` | **Deprecated / do-not-use** — legacy slice materialization; not a supported training path |
| `budget_calculator.py` | GPU-hour / token budget vs olmohq availability |
| `train_datadecide_60m.py` | Train one mixture (DataDecide 60M, in-run curve eval) |
| `run_mixture.sh` | Single-GPU worker: train + full eval + S3 upload-before-end |
| `eval_task_loss.py` | Task-loss eval on the six curve labels (or `--full-suite` for all 20) |
| `run_task_loss_eval.py` | Batch re-eval helper for finished checkpoints |
| `extrapolate_chinchilla.py` | Extrapolate in-run jsonl curves to Chinchilla step (tpp = 20); excludes step-1451 final anchor |
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
| `prepare_validation_370m_data.py` | Per-arm `mix_weights.json` sidecars from recipe |
| `train_mixlaw_validation_370m.py` | OLMo2-370M CE trainer (streams at recipe weights) |
| `launch_validation_370m.sh` | One OLMo2-370M CE arm on a recipe mix |
| `mixlaw_runtime.py` | Pure recovery, dependency, and production contracts |
| `preflight_validation_370m.py` | OLMES/dependency preflight + version metadata |
| `submit_mixlaw_validation_pool.sh` | FarmShare: stage peak pool from edullm-data |
| `stage_validation_pool_from_edullm_data.py` | 370M peak pool download+concat from edullm-data |
| `submit_mixlaw_validation_370m.sh` | Slurm array over all recipe mixes (`TRAIN_VENV` required) |
| `build_working_pool_from_shards.py` | **Deprecated** — peak pool from `tokenized_manifest.json` |
| `check_validation_pool.py` | Peak demand vs olmohq inventory |
| `validation_mixtures_10b.json` | Eight 10B-mix recipe for 370M scale-up |
| `generate_readme.py` | Regenerate this README from JSON artifacts |

---

## S3 artifact paths

| Artifact | Location | Notes |
|----------|----------|-------|
| **Training corpus** | `s3://edullm-data/pretrain/olmo-127b` | Published+validated; staged via `stage_*_from_edullm_data.py` |
| **Pilot results** | `s3://edullm-checkpoints/mixlaw/60m-pilot/mix01` … `mix24` | Checkpoints + progress + logs (upload-before-end) |
| **370M validation results** | `s3://edullm-checkpoints/mixlaw/370m-validation/<mix>/` | Fail-closed sync from trainer / `train_one.sh` |
| **370M validation recipe** | `validation_mixtures_10b.json` | Domain weights per arm; training streams from edullm-data pool |
| **Working pool** | ephemeral `POOL_DIR` (job scratch) | Peak-sized memmap pool staged each job; provenance required |
| Per-mix progress | `…/mixNN/progress/` | `run_meta.json`, `task_loss_final.json`, `task_loss.jsonl` |
| Per-mix logs | `…/mixNN/logs/` | `train.log`, `eval.log` |
| **Local mirror** | `pilot_runs/mixNN/progress/` | Historical progress JSON from the completed (pre-streaming) pilot; not a replay target |

`run_mixture.sh` defaults `RESULTS_S3=s3://edullm-checkpoints/mixlaw/60m-pilot` and
syncs checkpoints + progress + logs before exit. Set `ALLOW_LOCAL_ONLY=1` only for
smoke tests on durable local disks.

### Weights & Biases

Trainers log to W&B project **`mixlaw`** (SmolLM-style) when enabled:

- CLI: `--wandb-project mixlaw --wandb-mode online|offline|disabled`
- FarmShare: push `wandb-session.env` via `scripts/farmshare/push_wandb_session_to_farmshare.sh $RUN_DIR`
- `run_mixture.sh` / `launch_validation_370m.sh` auto-enable `online` when the session
  file or `WANDB_API_KEY` is present; otherwise mode stays `disabled` (S3-only).
- Run names: 60M pilot → `mixNN`; 370M validation → `mixlaw-370m-<MIX_NAME>`
  (group `60m-pilot` / `370m-validation`).
- W&B is **additive** — fail-closed S3 export is unchanged.


---

## Surrogate fits on Chinchilla targets

Two surrogates map 7-domain mixture weights → six Chinchilla-extrapolated curve losses
(step 5806, tpp = 20) from 24 pilot mixtures. Mixture optima cap **wiki ≤ 30%**
(olmohq inventory binds below unconstrained optima):

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
| mean LOO RMSE | 0.0770 | 0.0751 |
| macro LOO RMSE | 0.0439 | 0.0366 |
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
| arc_challenge | 1.5043 | 1.7831 | 0.0790 |
| arc_easy | 1.8108 | 2.1452 | 0.0878 |
| mmlu_humanities | 1.6810 | 1.9250 | 0.0678 |
| mmlu_other | 2.2812 | 2.7868 | 0.1331 |
| mmlu_social_sciences | 1.2668 | 1.4775 | 0.0608 |
| mmlu_stem | 2.1722 | 2.4410 | 0.0805 |

### Leave-one-out cross-validation

| Metric | Mixing law | LightGBM |
|--------|------------|----------|
| Mean LOO RMSE | 0.0766 | 0.0751 |
| Mean LOO RMSE / std | 84.4% | 84.1% |
| Macro LOO RMSE | 0.0443 | 0.0366 |

| family | ML LOO | ML % std | LGB LOO | LGB % std | ML in-sample | LGB in-sample |
|--------|--------|----------|---------|-----------|--------------|---------------|
| arc_challenge | 0.0774 | 97.9% | 0.0757 | 95.8% | 0.0471 | 0.0079 |
| arc_easy | 0.1269 | 144.5% | 0.1163 | 132.5% | 0.0887 | 0.0159 |
| mmlu_humanities | 0.0271 | 40.0% | 0.0288 | 42.4% | 0.0163 | 0.0028 |
| mmlu_other | 0.1432 | 107.6% | 0.1306 | 98.1% | 0.0951 | 0.0111 |
| mmlu_social_sciences | 0.0269 | 44.2% | 0.0317 | 52.1% | 0.0198 | 0.0037 |
| mmlu_stem | 0.0581 | 72.2% | 0.0674 | 83.7% | 0.0433 | 0.0090 |

### Mixing-law parameters (c_i, k_i, t_ij)

| family | c_i | k_i | k/std | max\|t\| | in-sample RMSE |
|--------|-----|-----|-------|---------|----------------|
| arc_challenge | 1.5043 | 0.0790 | 1.00 | 3.72 | 0.0471 |
| arc_easy | 1.8108 | 0.0878 | 1.00 | 2.16 | 0.0887 |
| mmlu_humanities | 1.6022 | 0.0678 | 1.00 | 1.69 | 0.0163 |
| mmlu_other | 2.1336 | 0.1331 | 1.00 | 1.77 | 0.0951 |
| mmlu_social_sciences | 1.2571 | 0.0608 | 1.00 | 2.77 | 0.0198 |
| mmlu_stem | 2.1427 | 0.0805 | 1.00 | 1.88 | 0.0433 |

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---|---|---|---|---|---|---|
| arc_challenge | -3.72 | 2.07 | 1.20 | -0.07 | 0.49 | 1.03 | -1.50 |
| arc_easy | 1.31 | 0.69 | 0.84 | 0.72 | -2.16 | 1.48 | -1.92 |
| mmlu_humanities | -0.67 | 1.63 | 1.69 | 0.72 | 1.63 | 1.61 | -1.27 |
| mmlu_other | 0.19 | 1.77 | 1.22 | 0.70 | 1.41 | 1.58 | -1.23 |
| mmlu_social_sciences | -2.77 | 1.69 | 1.34 | 0.17 | 1.50 | 1.82 | -1.64 |
| mmlu_stem | -0.07 | 1.88 | 1.43 | -1.66 | 1.18 | 1.59 | -0.49 |

### LightGBM feature importance (gain)

| family | dclm | arxiv | starcoder | pes2o | open-web-math | algebraic-stack | wiki |
|--------|---|---|---|---|---|---|---|
| arc_challenge | 0.75 | 0.06 | 0.02 | 0.15 | 0.32 | 0.05 | 0.18 |
| arc_easy | 0.27 | 0.32 | 0.26 | 0.08 | 0.53 | 0.19 | 0.19 |
| mmlu_humanities | 0.85 | 0.00 | 0.05 | 0.06 | 0.04 | 0.01 | 0.12 |
| mmlu_other | 1.76 | 0.25 | 0.59 | 0.96 | 0.30 | 0.06 | 0.40 |
| mmlu_social_sciences | 0.67 | 0.02 | 0.03 | 0.14 | 0.01 | 0.02 | 0.01 |
| mmlu_stem | 0.22 | 0.06 | 0.24 | 0.71 | 0.02 | 0.01 | 0.31 |

### Mixture optima and near-optimal candidates

Constrained optima (`uncapped`, `pilot_caps`, `min1pct`) plus sampled near-optimal
mixtures. Near-opt rows: **≥ 1%** on every domain, within **+0.04 bpb** of each model's
uncapped optimum (mixing law 1.7965, LightGBM 1.8316), and **≥ 8 pp**
(L∞) away from every optimum row above. None exactly match a pilot point.

**Mixing law**

| label | pred macro | max_w | dclm | arxiv | star | pes2o | owm | alg | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 1.7965 | 0.568 | 0.568 | — | — | 0.097 | 0.035 | — | 0.300 |
| pilot_caps | 1.7965 | 0.568 | 0.568 | — | — | 0.097 | 0.035 | — | 0.300 |
| min1pct | 1.7984 | 0.556 | 0.556 | 0.010 | 0.010 | 0.092 | 0.023 | 0.010 | 0.300 |
| near-opt 1 | 1.7995 | 0.440 | 0.440 | 0.011 | 0.011 | 0.207 | 0.013 | 0.018 | 0.300 |
| near-opt 2 | 1.8016 | 0.435 | 0.435 | 0.018 | 0.011 | 0.134 | 0.069 | 0.032 | 0.300 |
| near-opt 3 | 1.8022 | 0.354 | 0.354 | 0.020 | 0.012 | 0.258 | 0.027 | 0.029 | 0.300 |
| near-opt 4 | 1.8024 | 0.440 | 0.440 | 0.012 | 0.012 | 0.043 | 0.181 | 0.012 | 0.300 |
| near-opt 5 | 1.8026 | 0.346 | 0.346 | 0.012 | 0.012 | 0.180 | 0.137 | 0.014 | 0.300 |
| near-opt 6 | 1.8032 | 0.309 | 0.274 | 0.011 | 0.021 | 0.309 | 0.073 | 0.011 | 0.300 |
| near-opt 7 | 1.8036 | 0.397 | 0.223 | 0.010 | 0.010 | 0.397 | 0.051 | 0.010 | 0.300 |
| near-opt 8 | 1.8041 | 0.638 | 0.638 | 0.024 | 0.036 | 0.010 | 0.016 | 0.010 | 0.267 |


**LightGBM**

| label | pred macro | max_w | dclm | arxiv | star | pes2o | owm | alg | wiki |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 1.8316 | 0.441 | 0.354 | 0.083 | 0.061 | 0.441 | 0.030 | 0.022 | 0.009 |
| pilot_caps | 1.8316 | 0.441 | 0.354 | 0.083 | 0.061 | 0.441 | 0.030 | 0.022 | 0.009 |
| min1pct | 1.8335 | 0.553 | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| near-opt 1 | 1.8335 | 0.558 | 0.558 | 0.199 | 0.073 | 0.083 | 0.033 | 0.042 | 0.012 |
| near-opt 2 | 1.8380 | 0.491 | 0.491 | 0.167 | 0.071 | 0.087 | 0.046 | 0.073 | 0.064 |
| near-opt 3 | 1.8383 | 0.495 | 0.495 | 0.167 | 0.072 | 0.179 | 0.015 | 0.055 | 0.017 |
| near-opt 4 | 1.8391 | 0.502 | 0.502 | 0.172 | 0.095 | 0.128 | 0.048 | 0.022 | 0.033 |
| near-opt 5 | 1.8397 | 0.505 | 0.505 | 0.190 | 0.125 | 0.080 | 0.021 | 0.024 | 0.055 |
| near-opt 6 | 1.8398 | 0.520 | 0.520 | 0.188 | 0.083 | 0.078 | 0.049 | 0.044 | 0.037 |
| near-opt 7 | 1.8398 | 0.583 | 0.583 | 0.171 | 0.082 | 0.069 | 0.029 | 0.048 | 0.018 |
| near-opt 8 | 1.8410 | 0.377 | 0.182 | 0.182 | 0.076 | 0.377 | 0.020 | 0.037 | 0.127 |

### Random-simplex plausibility

**1000** mixtures sampled uniformly on the 7-simplex (Dirichlet(1,…,1), seed 42). Pilot observed macro range: **2.0444 – 2.2888 bpb**.

| Metric | Mixing law | LightGBM |
|--------|------------|----------|
| Macro min | 1.8526 | 1.8408 |
| Macro p50 | 1.9131 | 1.9297 |
| Macro p95 | 2.0166 | 2.0022 |
| Macro p99 | 2.0602 | 2.0217 |
| Macro max | 2.1395 | 2.0339 |
| Macro mean ± std | 1.9231 ± 0.0467 | 1.9393 ± 0.0441 |
| % inside pilot macro range | 2.3% | 0.0% |
| Mixtures with macro > 3 bpb | 0 | 0 |
| Mixtures with macro > 5 bpb | 0 | 0 |
| Any family > 10 bpb | 0 | 0 |
| mmlu_other max | 2.8288 | 2.7716 |

Sources: `mixlaw_random_simplex_plausibility.json` (mixing law); `mixlaw_fit_lightgbm_chinchilla.json` (LightGBM).

### Pilot mixtures ranked by predicted macro

Sorted by mixing-law prediction; LightGBM rank shown for comparison.

| ML rank | LGB rank | mix | tag | ML pred | LGB pred | max_w | measured curve-6 |
|---------|----------|-----|-----|---------|----------|-------|------------------|
| 1 | 4 | mix18 | C1 | 1.8368 | 1.8700 | 0.425 | 2.1089 |
| 2 | 2 | mix07 | C1-dclm60 | 1.8618 | 1.8451 | 0.600 | 2.0444 |
| 3 | 7 | mix22 | C1 | 1.8666 | 1.8810 | 0.237 | 2.0899 |
| 4 | 1 | mix06 | C1-dclm55 | 1.8668 | 1.8370 | 0.550 | 2.0963 |
| 5 | 5 | mix05 | C1-dclm50 | 1.8726 | 1.8714 | 0.500 | 2.0830 |
| 6 | 14 | mix23 | C1 | 1.8733 | 1.9107 | 0.475 | 2.0981 |
| 7 | 17 | mix19 | C1 | 1.8752 | 1.9628 | 0.325 | 2.1456 |
| 8 | 13 | mix17 | C1 | 1.8897 | 1.9093 | 0.475 | 2.0835 |
| 9 | 10 | mix01 | base | 1.8913 | 1.8951 | 0.375 | 2.0727 |
| 10 | 8 | mix04 | C0-dclm0 | 1.8964 | 1.8888 | 0.425 | 2.1174 |
| 11 | 6 | mix21 | C1 | 1.8969 | 1.8764 | 0.350 | 2.1364 |
| 12 | 3 | mix16 | C1 | 1.8992 | 1.8527 | 0.400 | 2.1125 |

### 370M validation plan

Eight mixtures selected for OLMo-370M scale-up: the natural **olmo-mix-1124** corpus mix (~95% DCLM), three pilot anchors (mix01, mix07, mix18), plus mixing-law pilot_caps + near-opt 4; LightGBM min1pct + near-opt 8).

| run | source | dclm | arxiv | starcoder | pes2o | open-web-math | alg-stack | wiki |
|-----|--------|---:|---:|---:|---:|---:|---:|---:|
| olmo-mix-1124 | reference | 0.951 | 0.005 | 0.021 | 0.015 | 0.003 | 0.003 | 0.001 |
| mix01 | pilot | 0.375 | 0.250 | 0.141 | 0.094 | 0.064 | 0.061 | 0.016 |
| mix07 | pilot | 0.600 | 0.160 | 0.090 | 0.060 | 0.041 | 0.039 | 0.010 |
| mix18 | pilot | 0.237 | 0.041 | 0.041 | 0.425 | 0.100 | 0.044 | 0.113 |
| ML-pilot_caps | mixing-law | 0.568 | 0.000 | 0.000 | 0.097 | 0.035 | 0.000 | 0.300 |
| ML-near-opt-4 | mixing-law | 0.440 | 0.012 | 0.012 | 0.043 | 0.181 | 0.012 | 0.300 |
| LGB-min1pct | lightgbm | 0.553 | 0.212 | 0.087 | 0.082 | 0.042 | 0.014 | 0.011 |
| LGB-near-opt-8 | lightgbm | 0.182 | 0.182 | 0.076 | 0.377 | 0.020 | 0.037 | 0.127 |

### Key takeaways

1. **mix07** has the lowest measured 6-family macro at the last in-run eval (2.0444 bpb); mixing-law top pilot by prediction is **mix18** (1.8368 bpb @ Chinchilla).
2. Mixing-law uncapped optimum is **1.7965** bpb (max_w=0.568); LightGBM uncapped is **1.8316** (max_w=0.441).
3. Claimed gains over the best measured pilot are **not distinguishable** from LOO error.
4. Mean LOO RMSE: mixing law 0.0766, LightGBM 0.0751.
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

## 370M validation (olmohq stream + recipe)

Eight **10B-token** OLMo-370M arms. Domain weights live in `validation_mixtures_10b.json` (schema 2). Training **streams from a shared olmohq working pool** at those weights — no per-mix corpora under `mixlaw/mixes/`.

| Mixture | Role |
|---------|------|
| `olmo-mix-1124` | Natural olmo-mix-1124 reference weights |
| `mix01` | RegMix base weights (same proportions as pilot mix01) |
| `mix07`, `mix18` | Pilot grid points |
| `ML-pilot_caps`, `ML-near-opt-4` | Mixing-law surrogates |
| `LGB-min1pct`, `LGB-near-opt-8` | LightGBM surrogates |

**Data source:** `s3://edullm-data/pretrain/olmo-127b/` (published+validated). Stage one peak-sized working pool from edullm-data, then train every arm from it.

### Pipeline

1. **pool** — `submit_mixlaw_validation_pool.sh` → `stage_validation_pool_from_edullm_data.py` (peak demand from recipe @ 10B).
2. **recipe sidecars** — `prepare_validation_370m_data.py` writes per-arm `mix_weights.json`.
3. **train** — `submit_mixlaw_validation_370m.sh` → `train_mixlaw_validation_370m.py` (`DomainMixtureStream`).

### Platform seven-arm array

The platform array runs indices `0..6` as `olmo-mix-1124`, `mix07`, `mix18`, `ML-pilot_caps`, `ML-near-opt-4`, `LGB-min1pct`, and `LGB-near-opt-8`. `mix01` is deliberately excluded because its control run is already separate.

- The image is built from `.edullm/Dockerfile` with the platform-supplied digest and `.edullm/requirements-linux-cu128.lock`. Publication is manual through the `Publish platform research image` workflow; implementation does not dispatch it.
- Submit repository `edullm-p1`, workload `mixlaw-validation-370m-8xa100`, dataset `olmo-127b-v1`, team `pre-training`, W&B project `mixlaw`, and fan-out size/parallelism `7`.
- Set experiment/fan-out parameter to `370m-validation`/`mixture` and use command `bash -lc 'EDULLM_LAUNCH_CHECK=waived exec python /opt/edullm/experiments/skill-dag/mixlaw/platform_array_entrypoint.py --checkpoint-prefix "$EDULLM_CHECKPOINT_DIR"'`. The explicit launcher waiver is required because admission can inspect the submitted shell command but cannot see that this entrypoint deterministically starts eight `torchrun` ranks.
- Every child requires the platform-injected `EDULLM_DATASET_ID=pretrain/olmo-127b` and `EDULLM_DATASET_VERSION=v1`; `latest` is refused. It uses `/scratch/$AWS_BATCH_JOB_ID`, stages only its selected arm, and runs eight ranks with two CPU threads each (16 vCPU total).
- Checkpoints publish beneath `$EDULLM_CHECKPOINT_DIR/array/<index>-<mix>/stepN/`; progress and task loss publish beneath `$EDULLM_OUTPUT_PREFIX/array/<index>-<mix>/`. Checkpoint contents land before `_COMPLETE.json`.
- Use the platform status/cancel workflow against the parent array or a specific child. Do not use the FarmShare submit scripts to manage Batch.
- Batch attempts remain `1`. Checkpoints are durable, but automatic restore from platform S3 is not implemented; a replacement run must first restore a checkpoint to local scratch and pass an explicit `--load-path`.

### Recovery, eval, and durability contract

- Set `RECOVERY_MODE=fresh|resume|fail` for every launch. The default is `fail`: clean scratch starts at step 0, but any leftover checkpoint aborts. `fresh` explicitly ignores leftovers. `resume` requires an explicit `--load-path` or a validated `last_durable_step.json`; no mode scans for the newest checkpoint.
- Production (platform S3 or online W&B durability) requires synchronous all-rank evaluation of all 20 OLMES `*_bpb` labels at every permanent checkpoint. Evaluation failures abort only after the train module has been rebuilt and reloaded. Local smoke runs may explicitly disable task loss.
- A platform checkpoint becomes durable only after its files and completion sentinel upload successfully; progress and task-loss JSON then publish under the isolated arm output prefix. FarmShare retains its existing local/W&B path.
- `launch_validation_370m.sh` runs `preflight_validation_370m.py` before training. It requires the complete OLMES label map, a ladder base config, the evaluator, and current dependencies; resolved package and dataset versions are recorded in `run_meta.json`.
- Install `edullm-data` from the newest release wheel or GitHub `main`; old pins such as `v0.2.0` are rejected.

The checked-in RunPod source bundle is deterministic and local-only:

```bash
python scripts/runpod/build_mixlaw_bundle.py --sync   # refresh local tree
python scripts/runpod/build_mixlaw_bundle.py          # parity check only
```

The builder does not create or upload a tarball and does not touch a pod.

### Validation scripts

| Script | Role |
|--------|------|
| `write_validation_mixtures.py` | Emit `validation_mixtures_10b.json` from fit JSON + pilot mixtures |
| `submit_mixlaw_validation_pool.sh` | FarmShare: stage peak pool from edullm-data |
| `stage_validation_pool_from_edullm_data.py` | Download+concat domain memmaps for 370M peak demand |
| `prepare_validation_370m_data.py` | Per-arm `mix_weights.json` sidecars from recipe |
| `train_mixlaw_validation_370m.py` | OLMo2-370M CE trainer (domain-stratified stream) |
| `launch_validation_370m.sh` | Train one recipe arm |
| `mixlaw_runtime.py` | Pure recovery, dependency, and production contracts |
| `preflight_validation_370m.py` | OLMES/dependency preflight + version metadata |
| `platform_array_entrypoint.py` | Deterministic seven-arm Batch mapping and platform paths |
| `platform_artifacts.py` | Fail-closed checkpoint/progress/task-loss S3 publication |
| `ladder_base_config.yaml` | Checked-in OLMo2-370M OLMES base config |
| `submit_mixlaw_validation_370m.sh` | Slurm array over all recipe arms |
| `domain_stream.py` | Shared olmohq domain sampler (also used by skillit) |
| `check_validation_pool.py` | Peak demand vs olmohq inventory |

**Unsupported for new work:** `submit_mixlaw_validation_10b.sh` + `build_mixture_data.py` + `build_working_pool_from_shards.py` + `finalize_mixlaw_upload.py` (per-mix slices / `edullm-datasets/mixlaw/`). Use the streaming peak-pool pipeline above only.

Regenerate the recipe after refitting:

```bash
cd experiments/skill-dag/mixlaw
py -3 write_validation_mixtures.py
bash submit_mixlaw_validation_pool.sh
TRAIN_VENV=/path/to/gpu-venv POOL_DIR=$RUN_DIR/pool SAVE_ROOT=... PROGRESS_ROOT=... bash submit_mixlaw_validation_370m.sh
```

