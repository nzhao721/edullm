# Token-selection experiments

**The code that produced the reported runs lives in
[`olmo_core_token_selection/`](olmo_core_token_selection/)** — a vendored copy of the
`.edullm/` tree from the `edullm/token-selection-370m` branch of the OLMo-core fork.
See [`olmo_core_token_selection/PROVENANCE.md`](olmo_core_token_selection/PROVENANCE.md)
for exactly what was copied, what was left out, and why the branch's pinned commit
cannot reproduce four of the seven arms.

**The code that built the corpora from Hugging Face sources lives in
[`datasets/`](datasets/)** — the training corpus (`pretrain/regmix-10b`), the HQ
reference, and the Instruct reference, each vendored from this same repo's top-level
`datasets/`. See [`datasets/README.md`](datasets/README.md) for what was copied and
what was left out.

The per-arm directories below (`attention/`, `middle-ppl-token/`, `rel-ema-exp/`,
`rho-1/`) are **documentation only**: a README describing each method and the YAML
recording its intended contract. Their launch scripts have been removed because they
targeted a superseded in-repo trainer that produced none of the reported runs; the real
launch paths are `olmo_core_token_selection/farmshare/` and
`olmo_core_token_selection/runpod/`.

Reference architecture source of truth: [`reference/`](reference/) (RefHQ CE, leave as-is).
This one *is* live: `reference/train_olmo3_370m_refhq.py` trained the HQ 5.5B reference
(`edullm-370M-refhq-5p5b`, checkpoints 125…1125 plus `step1315`) that the Perplexity arm
scores against.

| Arm | Directory | Selection | Status |
|-----|-----------|-----------|--------|
| Full-loss control | `arms.py` `full-loss-control` | none (full CE on every valid target token) | `full-loss-control-regmix10b-v3` — W&B [`eduLLM/token-selection/349f144dc23ee52d18396be695d6b6b0`](https://wandb.ai/eduLLM/token-selection/runs/349f144dc23ee52d18396be695d6b6b0); matched to the selection arms, see below |
| Control (random 60%) | `arms.py` `random-control` | uniform random keep 60% | **Two seeds**: `random-control-regmix10b-v1` (seed 42, W&B `fa841187ff07e9164da282efd353c217`) and `random-control-regmix10b-seed69-v1` (seed 69, W&B `123189f79a722b3481d06bc48b61fad9`). Reported as a single two-run fit, see below |
| BLADE | `arms.py` `blade` | top-60% `L_proxy − L_ref` | RegMix proxy/penalty stream + pinned `pretrain/refhq-instruct/v3` updates; syncs 500/875/1250/1625/2000; K=75, τ=375, γ=0.6, λ=1.0; blade_start=500; pre/post-sync checkpoints |
| RHO-1 | [`rho-1/`](rho-1/) | top-60% `L_curr − L_ref` | Frozen refhq-instruct v3 step940; `t0=0`; YAML spine |
| REL exp-α | [`rel-ema-exp/`](rel-ema-exp/) | top-60% `L_curr − L_hist` | Bias-corrected EMA from zero; `α(t)=1−e^(−t/300)`; `t0=0` |
| Middle PPL (token) | [`middle-ppl-token/`](middle-ppl-token/) | middle-60% by frozen RefHQ `L_ref` | Offline precomputed masks; `t0=0` |
| Attention | [`attention/`](attention/) | top-60% attn-received | `attention_topk`; FA-safe hook+recompute; `t0=0` |
| Reference (RefHQ) | [`reference/`](reference/) | — | Frozen HQ 5.5B; arch / refs only |

### Shared contracts

- **Architecture:** `TransformerConfig.olmo2_370M` (full attn, no SWA), GBS `4_194_304`, seq 2048, peak LR `4e-4`, warmup 24, `alpha_f=0.1` (match RefHQ).
- **Permanent checkpoints:** step 0, every 125 (skip last grid point if within 125 of final), plus final — e.g. `{0,125,…,2125,2360}` (omit 2250). Helper: `token_selection.olmo_ext.checkpoint_ladder`.
- **Token budget:** one epoch of published `pretrain/regmix-10b` **v1**, whose realized size is
  **10,004,807,041 tokens**. YAML/standalone defaults use `9900000000` → **2360** steps at GBS
  `4_194_304`. All seven arms (full-loss control, random control, RHO-1, BLADE, Attention,
  Middle-PPL, REL-EMA) train on this same published dataset, all for **2360** steps
  (9,898,557,440 tokens), which does not wrap into a second epoch — it stays under
  10,004,807,041.

  Realized per-domain token counts (uint32 `.npy` bytes / 4, verified on FarmShare):

  | Domain | Tokens |
  | --- | --- |
  | dclm | 3,752,801,841 |
  | arxiv | 2,500,162,905 |
  | starcoder | 1,406,986,385 |
  | pes2o | 938,157,310 |
  | open-web-math | 635,098,778 |
  | algebraic-stack | 615,239,017 |
  | wiki | 156,360,805 |
  | **total** | **10,004,807,041** |

- **Online selection warmup:** `t0_steps=0` / `t0_frac=0` for all online scorers. BLADE keeps its separate 500-step proxy warmup.
- **Task loss:** full 20-label OLMo-ladder `task_loss_bpb` (RC 5-shot) via `task_loss_hook` / `TaskLossEvalCallback` on each permanent save. Evaluator: `scripts/farmshare/task_loss/eval_task_loss_olmo_core.py`.
  The **20 labels are (task, split) pairs over 10 OLMES benchmarks**, not 20 distinct benchmarks:
  ARC-Challenge, ARC-Easy, BoolQ, CSQA, HellaSwag, MMLU, OpenBookQA, PIQA, SocialIQA, WinoGrande.
  Two consequences worth stating explicitly when reading the macro number:
  - **MMLU contributes 8 of the 20 labels** (`stem` / `humanities` / `social_sciences` / `other`
    × `val` / `test`), i.e. **40% of the macro weight** sits on a single benchmark.
  - **7 of the 20 labels are `test` splits** (ARC-Challenge, ARC-Easy, OpenBookQA, and the four
    MMLU categories), so calling the aggregate "validation macro bpb" is a **misnomer**; it is a
    mixed val/test macro.
- **Hardware:** discover world size from `torchrun` / env; no hardcoded GPU count, device pins, or host paths as required defaults.

### Full-loss control (reported)

| Field | Value |
| --- | --- |
| Run | `full-loss-control-regmix10b-v3` |
| W&B | [`eduLLM/token-selection/349f144dc23ee52d18396be695d6b6b0`](https://wandb.ai/eduLLM/token-selection/runs/349f144dc23ee52d18396be695d6b6b0) |
| Hardware | 4×L40S (FarmShare job 1719708) |
| Steps | **2360**, same as every other arm |
| Eval grid | the **125**-step permanent ladder, same as every other arm |
| Method | `method="full"` with `keep_fraction=1.0`, routed through the stock train module (not the selection path) |

This run replaced `full-loss-control-regmix10b-v2` (W&B `hh19uatg`), which was cloned from
`eduLLM/hpo-ladder` and was **not** a matched sibling of the selection arms: dataloader seed 6199
rather than 42, 2384 steps rather than 2360, a ~119-step eval grid, and an init seed that logged as
0 because `init_seed` never propagated into `TransformerConfig`. v3 is matched on init (confirmed
via the step-0 eval fingerprint), data seed, step count and eval grid, so that confound is gone.

**Initialization is not uniform across the arms.** The step-0 evaluation splits the eight runs
into three groups: **4.4610** bpb (random control seed 69), **4.4662** bpb (full-loss control,
random control seed 42, REL-EMA) and **4.4838** bpb (RHO-1, Attention, BLADE, Middle-PPL) — a
0.0228 bpb spread before any training. Two consequences:

- The full-loss control and the **seed-42** random control share an initialization, but the
  **seed-69** replicate does not, so the pooled two-run random control mixes two initializations.
  That is a fair estimate of run-to-run variance (seed 69 re-rolls init as well as selection), but
  it means the 0.0082 bpb seed spread is *not* attributable to the selection RNG alone.
- RHO-1 — the one arm that finishes below the random control — sits in the third group, so its
  comparison against either control is confounded by initialization.

### Random control (reported as a two-run fit)

The random-60% arm was run at two seeds:

| Run | Seed | W&B | Fitted final | 95% CI |
| --- | --- | --- | --- | --- |
| `random-control-regmix10b-v1` | 42 | `fa841187ff07e9164da282efd353c217` | 1.6857 | [1.6775, 1.6920] |
| `random-control-regmix10b-seed69-v1` | 69 | `123189f79a722b3481d06bc48b61fad9` | 1.6939 | [1.6865, 1.6993] |

The two differ by **0.0082 bpb** (95% CI [0.0033, 0.0126], two-sided **p = 0.0002**), which is
larger than several of the between-arm gaps, so a single run of this arm cannot carry the baseline.
It is therefore **reported as one two-run fit**: a single power law fitted to the union of both
runs' fit-window points (11 each, 22 total), with the bootstrap taken over the pooled residuals, so
the interval carries seed-to-seed spread as well as within-run noise. Pooled: **1.6900**, 95% CI
**[1.6841, 1.6947]**. Every comparison involving the random control uses this fit.

The seed-69 run's step-1500 evaluation was recovered from its on-disk `task_loss` artifact; a resume
collided with W&B's monotonic-step rule and dropped it from the logged history.

---

Each arm’s `README.md` describes its method and hyperparameters. Its *launch* sections are historical: the scripts they name were removed with the superseded trainer (see above). The launch path that produced the reported runs is `olmo_core_token_selection/farmshare/` and `olmo_core_token_selection/runpod/`. Do not submit AWS workloads unless explicitly authorized.

---

Full experiment plan: [README.md](README.md).  
Contamination audit (paper Section 2.3): [contamination/](contamination/).  
