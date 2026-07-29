#!/usr/bin/env python3
"""Regenerate README.md from fit artifacts and mixtures.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
FIT = json.loads((ROOT / "mixlaw_fit_chinchilla.json").read_text(encoding="utf-8"))
LOO = json.loads((ROOT / "mixlaw_fit_chinchilla_loo.json").read_text(encoding="utf-8"))
LGB = json.loads((ROOT / "mixlaw_fit_lightgbm_chinchilla.json").read_text(encoding="utf-8"))
MIX = json.loads((ROOT / "mixtures.json").read_text(encoding="utf-8"))
VALIDATION_PLAN = json.loads((ROOT / "validation_mixtures_10b.json").read_text(encoding="utf-8"))
SIMPLEX = json.loads((ROOT / "mixlaw_random_simplex_plausibility.json").read_text(encoding="utf-8"))
DOMAINS = MIX["domain_order"]
FAMILIES = sorted(FIT["targets"].keys())


def fmt(x: float, n: int = 4) -> str:
    if abs(x) >= 1000 or (0 < abs(x) < 1e-3):
        return f"{x:.3g}"
    return f"{x:.{n}f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


CONSTRAINT_ORDER = ("uncapped", "pilot_caps", "min1pct")
DOMAIN_HDR = ("dclm", "arxiv", "star", "pes2o", "owm", "alg", "wiki")
VALIDATION_DOMAIN_COLS = (
    "dclm",
    "arxiv",
    "starcoder",
    "pes2o",
    "open-web-math",
    "alg-stack",
    "wiki",
)


def fmt_weight(x: float) -> str:
    if x < 0.001:
        return "—"
    return fmt(x, 3)


def bullet_field(label: str, value: str) -> str:
    return f"- **{label}**: {value}"


def weights_list_to_dict(weights: list[float]) -> dict[str, float]:
    return dict(zip(DOMAINS, weights))


def validation_table_lines() -> list[str]:
    domain_hdr = " | ".join(VALIDATION_DOMAIN_COLS)
    out = [
        "### 370M validation plan",
        "",
        "Eight mixtures selected for OLMo-370M scale-up: the natural "
        "**olmo-mix-1124** corpus mix (~95% DCLM), three pilot anchors "
        "(mix01, mix07, mix18), plus min1pct and one near-opt per surrogate "
        "(ML near-opt 3, LGB near-opt 5).",
        "",
        f"| run | source | {domain_hdr} |",
        "|-----|--------|" + "|".join(["---:"] * len(VALIDATION_DOMAIN_COLS)) + "|",
    ]
    for m in VALIDATION_PLAN["mixtures"]:
        w = weights_list_to_dict(m["weights"])
        out.append(
            f"| {m['run_name']} | {m['source']} | "
            + " | ".join(f"{w[d]:.3f}" for d in DOMAINS)
            + " |"
        )
    return out


def validation_corpora_lines() -> list[str]:
    """Published 370M validation corpora (materialized on S3)."""
    runs = (
        "olmo-mix-1124",
        "mix01",
        "mix07",
        "mix18",
        "ML-min1pct",
        "ML-near-opt-3",
        "LGB-min1pct",
        "LGB-near-opt-5",
    )
    mix_rows = "\n".join(f"| `{name}` | `mixlaw/mixes/{name}/` |" for name in runs)
    return [
        "---",
        "",
        "## 370M validation corpora (built)",
        "",
        "Materialized **10B dolma2 tokens per mixture** for OLMo-370M scale-up. Recipe: "
        "`validation_mixtures_10b.json` (canonical copy on S3 at "
        "`mixlaw/validation_mixtures_10b.json`). Published **2026-07-29** — "
        "`s3://edullm-datasets/mixlaw/READY`.",
        "",
        "| Mixture | S3 prefix |",
        "|---------|-----------|",
        mix_rows,
        "",
        "**mix01 policy:** server-side copy *from* "
        "`s3://edullm-datasets/regmix/regmix-10b/` into `mixlaw/mixes/mix01/` only. "
        "`regmix/regmix-10b` is **read-only** and must never be modified.",
        "",
        "The other seven mixes are sliced offline from a peak-sized olmohq working pool "
        "(`olmo100b/olmo-mix-1124-30b` tokenized shards). Slices are contiguous "
        "2048-token chunks with largest-remainder domain allocation (same convention as "
        "the 60M pilot).",
        "",
        "### Build pipeline",
        "",
        "1. **pool** — `build_working_pool_from_shards.py` downloads olmohq `.npy` shards "
        "and concatenates a peak-sized pool per domain.",
        "2. **slice** — `build_mixture_data.py plan` + `build` materializes each "
        "non-reuse mixture under `slices/<run_name>/`.",
        "3. **upload** — `finalize_mixlaw_upload.py` syncs slices to `mixlaw/mixes/`, "
        "copies mix01 from regmix server-side, writes `mixlaw_upload_receipt.json`, "
        "and publishes `READY` last.",
        "",
        "### Validation build scripts",
        "",
        "| Script | Role |",
        "|--------|------|",
        "| `write_validation_mixtures.py` | Emit `validation_mixtures_10b.json` from fit JSON + pilot mixtures |",
        "| `build_working_pool_from_shards.py` | Peak pool from olmohq `tokenized_manifest.json` |",
        "| `build_mixture_data.py` | Plan + materialize per-mixture slices from the working pool |",
        "| `finalize_mixlaw_upload.py` | Upload to `mixlaw/`; mix01 = regmix server-side copy |",
        "| `validation_mixtures_10b.json` | Eight-mix recipe (`reuse_s3` only on mix01) |",
        "",
        "Regenerate the recipe locally after refitting:",
        "",
        "```bash",
        "cd experiments/skill-dag/mixlaw",
        "py -3 write_validation_mixtures.py",
        "```",
    ]


def mixture_table_for_model(fit: dict, title: str) -> list[str]:
    weight_hdr = " | ".join(DOMAIN_HDR)
    out = [
        f"**{title}**",
        "",
        "| label | pred macro | max_w | " + weight_hdr + " |",
        "|---|---:|---:|" + "|".join(["---:"] * len(DOMAINS)) + "|",
    ]
    for name in CONSTRAINT_ORDER:
        if name not in fit["optimization"]:
            continue
        row = fit["optimization"][name]
        w = row["weights"]
        out.append(
            f"| {name} | {fmt(row['predicted_macro'])} | {fmt(row['max_w'], 3)} | "
            + " | ".join(fmt_weight(w[d]) for d in DOMAINS)
            + " |"
        )
    for i, row in enumerate(fit.get("near_optimal_balanced_samples", []), 1):
        w = row["weights"]
        out.append(
            f"| near-opt {i} | {fmt(row['predicted_macro'])} | {fmt(row['max_w'], 3)} | "
            + " | ".join(fmt_weight(w[d]) for d in DOMAINS)
            + " |"
        )
    return out


lgb_simplex = LGB["random_simplex_plausibility"]
obs_lo, obs_hi = SIMPLEX["observed_pilot_macro_range"]
macro_ml = SIMPLEX["macro"]
macro_lgb = lgb_simplex["macro"]
lgb_rank = {row["run_name"]: i for i, row in enumerate(LGB["pilot_ranked"], 1)}


lines: list[str] = [
    "# Skill-DAG mixing-law pilot",
    "",
    "24-mixture probe over 7 OLMoHQ domains, trained with **DataDecide-60M** proxy models,",
    "evaluated on OLMo-ladder **task loss** (bits-per-byte). Fitted with a regularized",
    "Ye et al. mixing law and a per-family **LightGBM** model on **Chinchilla-extrapolated**",
    "targets (step 5806, tpp = 20).",
    "",
    "---",
    "",
    "## Proxy model architecture",
    "",
    "Each pilot run trains the exact **DataDecide 60M** geometry from",
    "[allenai/DataDecide-dolma1_7-60M](https://huggingface.co/allenai/DataDecide-dolma1_7-60M)",
    "(Ye et al., arXiv:2403.16952), with dolma2 tokenization to match the olmohq corpus.",
    "",
    bullet_field("Hidden size (`d_model`)", "384"),
    bullet_field("Layers", "16"),
    bullet_field("Heads", "12"),
    bullet_field("MLP ratio", "8"),
    bullet_field("Sequence length", "2048"),
    bullet_field("Global batch", "96 sequences"),
    bullet_field("Learning rate", "5.8e-3"),
    bullet_field("Tokenizer", "dolma2 (`allenai/dolma2-tokenizer`)"),
    bullet_field("Embedding rows", "100,352 (vocab 100,278 + specials)"),
    bullet_field("LM head", "Untied"),
    bullet_field("Body params", "37.8M"),
    bullet_field("Non-embedding params (tokens/param denominator)", "**57.1M**"),
    bullet_field("Total params (this run)", "**114.8M**"),
    "",
    "**Pilot budget:** tokens/param = 5 → **285M tokens / 1451 steps** per mixture (~30 min on one B200).",
    "",
    "**Evaluation:** OLMo-ladder task loss — bits-per-byte on six in-run **ARC + MMLU**",
    "curve families (val splits). Final eval and mixing-law fits use only these six.",
    "",
    "---",
    "",
    "## Mixture sampling (probe domain weights)",
    "",
    "Mixtures are defined in `mixtures.json` using **Algorithm 2** (double-diminishing grid)",
    "from Ye et al. (arXiv:2403.16952):",
    "",
    "1. Set **base mixture** to RegMix proportions (mix01).",
    "2. Compute **r_max** per domain from olmohq pool availability at a 30B target corpus.",
    "3. Sample a **double-diminishing grid** with step **δ = 0.05** and **seed 42**.",
    "4. Apply constraints: wiki floor 0.5% (except wiki-ablation tags), dclm cap 60%,",
    "   other domains cap 70%.",
    "5. **Inject 3 extra points** at 50% / 55% / 60% DCLM (tags `C1-dclm50/55/60`) —",
    "   mid-high DCLM weights the coarse grid cannot reach with all domains positive.",
    "",
    "Result: **24 designed probe points** (not random uniform samples). Tags:",
    "",
    "| Tag | Meaning |",
    "|-----|---------|",
    "| `base` | RegMix reference mixture |",
    "| `C0-wiki0` | Wiki ablation (wiki = 0) |",
    "| `C0-dclm0` | DCLM ablation (dclm = 0) |",
    "| `C1-dclm50/55/60` | Injected high-DCLM points |",
    "| `C1` | Standard grid points |",
    "",
    "Mixture proportions are realized as **exact per-domain sequence counts** on disk",
    "(largest-remainder allocation); one epoch shuffles non-overlapping 2048-token chunks.",
    "",
    "---",
    "",
    "## Pilot mixture domain weights",
    "",
    "Columns follow `domain_order` in `mixtures.json`. Weights sum to 1.",
    "",
    "| mix | tag | dclm | arxiv | starcoder | pes2o | open-web-math | alg-stack | wiki |",
    "|-----|-----|------|-------|-----------|-------|---------------|-----------|------|",
]

for row in MIX["mixtures"]:
    w = row["weights"]
    lines.append(
        f"| mix{row['id']:02d} | {row['tag']} | "
        + " | ".join(f"{x:.3f}" for x in w)
        + " |"
    )

lines += [
    "",
    "---",
    "",
    "## Script inventory",
    "",
    "| Script | Role |",
    "|--------|------|",
    "| `prepare_data.sh` | One-time pipeline: fetch olmohq shards, tokenize working pool, build 24 mixture slices |",
    "| `select_and_fetch_shards.py` | Random shard draw per domain from S3 inventory (overshoot-aware) |",
    "| `tokenize_working_pool.py` | Tokenize fetched raw shards into per-domain memmaps |",
    "| `build_mixture_data.py` | Plan + materialize per-mixture random subsamples from the working pool |",
    "| `budget_calculator.py` | GPU-hour / token budget vs olmohq availability |",
    "| `train_datadecide_60m.py` | Train one mixture (DataDecide 60M, in-run curve eval) |",
    "| `run_mixture.sh` | Single-GPU worker: train + full eval for one `mixNN` |",
    "| `eval_task_loss.py` | Task-loss eval on the six curve labels (or `--full-suite` for all 20) |",
    "| `run_task_loss_eval.py` | Batch re-eval helper for finished checkpoints |",
    "| `extrapolate_chinchilla.py` | Extrapolate in-run curves to Chinchilla step (tpp = 20) |",
    "| `fit_mixing_law.py` | Baseline mixing-law fit + simplex optimization |",
    "| `fit_chinchilla.py` | Regularized fit on Chinchilla targets + near-optimal sampling |",
    "| `fit_lightgbm_chinchilla.py` | LightGBM fit on Chinchilla targets + near-optimal sampling |",
    "| `loo_chinchilla.py` | Leave-one-out cross-validation |",
    "| `sample_random_simplex.py` | Random-simplex plausibility check |",
    "| `preflight_checks.py` | Sanity checks on data layout, step law, and simplex optimizer |",
    "| `mixlaw_common.py` | Shared constants (domains, DataDecide geometry, task labels) |",
    "| `mixtures.json` | 24 probe mixtures (Algorithm 2 grid + injected points) |",
    "| `reoptimize_constraints.py` | Re-run optima / near-optimal sampling on saved fits |",
    "| `write_validation_mixtures.py` | Emit 370M validation recipe (`validation_mixtures_10b.json`) |",
    "| `build_working_pool_from_shards.py` | Peak olmohq pool for 370M validation slices |",
    "| `finalize_mixlaw_upload.py` | Publish validation corpora to `s3://edullm-datasets/mixlaw/` |",
    "| `validation_mixtures_10b.json` | Eight 10B-mix recipe for 370M scale-up |",
    "| `generate_readme.py` | Regenerate this README from JSON artifacts |",
    "",
    "---",
    "",
    "## S3 artifact paths",
    "",
    "| Artifact | Location | Notes |",
    "|----------|----------|-------|",
    "| **Training corpus** | `s3://edullm-datasets/olmo100b/olmo-mix-1124-30b` | Raw olmohq json.gz shards; only a subset is fetched locally |",
    "| **Pilot results** | `s3://edullm-checkpoints/token-selection/mixlaw-pilot/mix01` … `mix24` | Logs + progress only (no weight checkpoints on S3) |",
    "| **370M validation corpora** | `s3://edullm-datasets/mixlaw/` | 10B tokens × 8 mixes; `READY` + `validation_mixtures_10b.json` |",
    "| Per-mix corpus | `…/mixlaw/mixes/<run_name>/` | Tokenized slices; mix01 copied from regmix-10b |",
    "| Per-mix progress | `…/mixNN/progress/` | `run_meta.json`, `task_loss_final.json`, `task_loss.jsonl` |",
    "| Per-mix logs | `…/mixNN/logs/` | `train.log`, `eval.log` |",
    "| **Local mirror** | `pilot_runs/mixNN/progress/` | Checked-in progress JSON from the completed pilot |",
    "",
    "Set `RESULTS_S3=s3://edullm-checkpoints/token-selection/mixlaw-pilot` in `run_mixture.sh`",
    "to sync progress/logs after each mix finishes. Weight checkpoints stay on local NVMe only.",
    "",
]

lines += [
    "",
    "---",
    "",
    "## Surrogate fits on Chinchilla targets",
    "",
    "Two surrogates map 7-domain mixture weights → six Chinchilla-extrapolated curve losses",
    "(step 5806, tpp = 20) from 24 pilot mixtures:",
    "",
    "| | Mixing law | LightGBM |",
    "|---|---|---|",
    "| Model | Regularized Ye et al. law | One gradient-boosted tree regressor per family |",
    "| Artifact | `mixlaw_fit_chinchilla.json` | `mixlaw_fit_lightgbm_chinchilla.json` |",
    "| Fit script | `fit_chinchilla.py` | `fit_lightgbm_chinchilla.py` |",
    "| LOO artifact | `mixlaw_fit_chinchilla_loo.json` | in-fit LOO per family |",
    f"| Pilot runs | {FIT['n_runs']} mixtures | {LGB['n_runs']} mixtures |",
    f"| Chinchilla step | {FIT['extrapolate_to_step']} | {LGB['extrapolate_to_step']} |",
    "",
    "### Mixing law",
    "",
    "```",
    "L_i(r) = c_i + k_i * exp( clip( sum_j t_ij * r_j, -60, 60 ) )",
    "```",
    "",
    "- `r` = mixture weights on the 7-domain simplex (sum to 1)",
    "- More negative `t_ij` → increasing domain j lowers task family i loss",
    "",
    "**Regularization** (among multi-start solutions within 1.35× best RMSE, pick parsimonious):",
    "",
    "| Parameter | Value |",
    "|-----------|-------|",
]
for k, v in FIT["regularization"].items():
    lines.append(f"| `{k}` | {v} |")

lines += [
    "",
    "### LightGBM",
    "",
    "Features = 7 mixture weights; target = per-family Chinchilla loss. Predicted macro = mean",
    "over families. Hyperparameters chosen by a **small LOO grid search** (144 configs) minimizing",
    "macro LOO RMSE. Mixture optima use **50k random simplex samples + SLSQP polish** (non-convex",
    "surrogate); constrained optima seed the uncapped search.",
    "",
    "| Parameter | Value |",
    "|-----------|-------|",
]
for k, v in LGB["lgb_params"].items():
    lines.append(f"| `{k}` | {v} |")

search = LGB.get("hyperparam_search", {})
if search:
    base = search["baseline_defaults"]
    sel = search["selected"]
    lines += [
        "",
        "**Hyperparameter search** (objective: minimize macro LOO RMSE):",
        "",
        "| | Baseline (hand-picked) | Selected (LOO grid) |",
        "|---|---|---|",
        f"| mean LOO RMSE | {fmt(base['mean_loo_rmse'])} | {fmt(sel['mean_loo_rmse'])} |",
        f"| macro LOO RMSE | {fmt(base['macro_loo_rmse'])} | {fmt(sel['macro_loo_rmse'])} |",
        f"| `num_leaves` | {base['params']['num_leaves']} | {sel['params']['num_leaves']} |",
        f"| `max_depth` | {base['params']['max_depth']} | {sel['params']['max_depth']} |",
        f"| `min_data_in_leaf` | {base['params']['min_data_in_leaf']} | {sel['params']['min_data_in_leaf']} |",
        f"| `learning_rate` | {base['params']['learning_rate']} | {sel['params']['learning_rate']} |",
        f"| `lambda_l2` | {base['params']['lambda_l2']} | {sel['params']['lambda_l2']} |",
        f"| `num_boost_round` | {base['num_boost_round']} | {sel['num_boost_round']} |",
    ]

lines += [
    "",
    "### Observed Chinchilla targets (fit y)",
    "",
    "Shared across both fits — per-family Chinchilla-extrapolated loss at step 5806.",
    "",
    "| family | min | max | std |",
    "|--------|-----|-----|-----|",
]
for fam in FAMILIES:
    o = FIT["targets"][fam]["observed"]
    lines.append(f"| {fam} | {fmt(o['min'])} | {fmt(o['max'])} | {fmt(o['std'])} |")

lines += [
    "",
    "### Leave-one-out cross-validation",
    "",
    "| Metric | Mixing law | LightGBM |",
    "|--------|------------|----------|",
    f"| Mean LOO RMSE | {fmt(LOO['summary']['mean_loo_rmse'])} | {fmt(LGB['summary']['mean_loo_rmse'])} |",
    f"| Mean LOO RMSE / std | {pct(LOO['summary']['mean_loo_rmse_over_std'])} | {pct(LGB['summary']['mean_loo_rmse_over_std'])} |",
    f"| Macro LOO RMSE | {fmt(LOO['macro']['loo_rmse'])} | {fmt(LGB['summary']['macro_loo_rmse'])} |",
    "",
    "| family | ML LOO | ML % std | LGB LOO | LGB % std | ML in-sample | LGB in-sample |",
    "|--------|--------|----------|---------|-----------|--------------|---------------|",
]
for fam in FAMILIES:
    ml = LOO["families"][fam]
    lg = LGB["family_models"][fam]
    lines.append(
        f"| {fam} | {fmt(ml['loo_rmse'])} | {pct(ml['loo_rmse_over_std'])} "
        f"| {fmt(lg['loo_rmse'])} | {pct(lg['loo_rmse_over_std'])} "
        f"| {fmt(ml['in_sample_rmse'])} | {fmt(lg['in_sample_rmse'])} |"
    )

lines += [
    "",
    "### Mixing-law parameters (c_i, k_i, t_ij)",
    "",
    "| family | c_i | k_i | k/std | max\\|t\\| | in-sample RMSE |",
    "|--------|-----|-----|-------|---------|----------------|",
]
for fam in FAMILIES:
    t = FIT["targets"][fam]
    lines.append(
        f"| {fam} | {fmt(t['c'])} | {fmt(t['k'])} | {fmt(t['k_over_std'], 2)} "
        f"| {fmt(t['max_abs_t'], 2)} | {fmt(t['fit_rmse'])} |"
    )

lines += [
    "",
    "| family | " + " | ".join(DOMAINS) + " |",
    "|--------|" + "|".join(["---"] * len(DOMAINS)) + "|",
]
for fam in FAMILIES:
    t = FIT["targets"][fam]["t"]
    row = " | ".join(fmt(t[d], 2) for d in DOMAINS)
    lines.append(f"| {fam} | {row} |")

lines += [
    "",
    "### LightGBM feature importance (gain)",
    "",
    "| family | " + " | ".join(DOMAINS) + " |",
    "|--------|" + "|".join(["---"] * len(DOMAINS)) + "|",
]
for fam in FAMILIES:
    imp = LGB["family_models"][fam]["feature_importance_gain"]
    row = " | ".join(fmt(imp[d], 2) for d in DOMAINS)
    lines.append(f"| {fam} | {row} |")

lines += [
    "",
    "### Mixture optima and near-optimal candidates",
    "",
    "Constrained optima (`uncapped`, `pilot_caps`, `min1pct`) plus sampled near-optimal",
    "mixtures. Near-opt rows: **≥ 1%** on every domain, within **+0.04 bpb** of each model's",
    f"uncapped optimum (mixing law {fmt(FIT['optimization']['uncapped']['predicted_macro'])}, "
    f"LightGBM {fmt(LGB['optimization']['uncapped']['predicted_macro'])}), and **≥ 8 pp**",
    "(L∞) away from every optimum row above. None exactly match a pilot point.",
    "",
]
lines += mixture_table_for_model(FIT, "Mixing law")
lines += ["", ""]
lines += mixture_table_for_model(LGB, "LightGBM")

lines += [
    "",
    "### Random-simplex plausibility",
    "",
    f"**1000** mixtures sampled uniformly on the 7-simplex (Dirichlet(1,…,1), seed 42). "
    f"Pilot observed macro range: **{fmt(obs_lo)} – {fmt(obs_hi)} bpb**.",
    "",
    "| Metric | Mixing law | LightGBM |",
    "|--------|------------|----------|",
    f"| Macro min | {fmt(macro_ml['min'])} | {fmt(macro_lgb['min'])} |",
    f"| Macro p50 | {fmt(macro_ml['p50'])} | {fmt(macro_lgb['p50'])} |",
    f"| Macro p95 | {fmt(macro_ml['p95'])} | {fmt(macro_lgb['p95'])} |",
    f"| Macro p99 | {fmt(macro_ml['p99'])} | {fmt(macro_lgb['p99'])} |",
    f"| Macro max | {fmt(macro_ml['max'])} | {fmt(macro_lgb['max'])} |",
    f"| Macro mean ± std | {fmt(macro_ml['mean'])} ± {fmt(macro_ml['std'])} | "
    f"{fmt(macro_lgb['mean'])} ± {fmt(macro_lgb['std'])} |",
    f"| % inside pilot macro range | {macro_ml['pct_in_pilot_range']:.1f}% | "
    f"{macro_lgb['pct_in_pilot_range']:.1f}% |",
    f"| Mixtures with macro > 3 bpb | {macro_ml['n_gt_3']} | {macro_lgb['n_gt_3']} |",
    f"| Mixtures with macro > 5 bpb | {macro_ml['n_gt_5']} | {macro_lgb['n_gt_5']} |",
    f"| Any family > 10 bpb | {SIMPLEX['any_family_gt_10']} | {lgb_simplex['any_family_gt_10']} |",
    f"| mmlu_other max | {fmt(SIMPLEX['mmlu_other_max'])} | {fmt(lgb_simplex['mmlu_other_max'])} |",
    "",
    "Sources: `mixlaw_random_simplex_plausibility.json` (mixing law); "
    "`mixlaw_fit_lightgbm_chinchilla.json` (LightGBM).",
    "",
    "### Pilot mixtures ranked by predicted macro",
    "",
    "Sorted by mixing-law prediction; LightGBM rank shown for comparison.",
    "",
    "| ML rank | LGB rank | mix | tag | ML pred | LGB pred | max_w | measured curve-6 |",
    "|---------|----------|-----|-----|---------|----------|-------|------------------|",
]
for i, row in enumerate(FIT["pilot_ranked"][:12], 1):
    lg_pred = next(
        r["predicted_macro"] for r in LGB["pilot_ranked"] if r["run_name"] == row["run_name"]
    )
    lines.append(
        f"| {i} | {lgb_rank[row['run_name']]} | {row['run_name']} | {row['tag']} "
        f"| {fmt(row['predicted_macro'])} | {fmt(lg_pred)} | {fmt(row['max_w'], 3)} "
        f"| {fmt(row['measured_curve_6'])} |"
    )

lines += [""]
lines += validation_table_lines()

lines += [
    "",
    "### Key takeaways",
    "",
    "1. **mix07** is the best measured pilot point; both models rank it first or near-first.",
    "2. Mixing-law optima favor **dclm + pes2o**; LightGBM optima favor **dclm + arxiv**.",
    "3. Claimed gains over mix07 (~0.03 bpb for mixing law) are **not distinguishable** from LOO error.",
    "4. Mean LOO RMSE is similar across models; mixing law has slightly lower macro LOO.",
    "5. Off-hull random mixtures stay bounded under both surrogates.",
    "",
    "### Reproduce",
    "",
    "```bash",
    "cd experiments/skill-dag/mixlaw",
    "py -3 extrapolate_chinchilla.py",
    "py -3 fit_chinchilla.py",
    "py -3 loo_chinchilla.py",
    "py -3 sample_random_simplex.py",
    "py -3 fit_lightgbm_chinchilla.py",
    "py -3 reoptimize_constraints.py --constraints-only --lightgbm",
    "py -3 generate_readme.py   # refresh this file",
    "```",
    "",
]
lines += validation_corpora_lines()
lines += [""]

(ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {ROOT / 'README.md'} ({len(lines)} lines)")
