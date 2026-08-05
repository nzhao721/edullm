# Token selection (Mixing Laws Dataset 10B × OLMo-2 370M)

**Question.** Under a matched one-epoch Mixing Laws Dataset budget, can selecting a subset of tokens per sequence beat full-token CE on macro task-loss?

**Answer.** No. Every selection arm finished **worse** than the full-CE baseline. Excess-loss ρ-1 was the least harmful; middle-perplexity and relative-EMA selection collapsed performance.

---

## Setup


| Knob                         | Value                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Architecture                 | OLMo-2 370M (full attention): d=1024, 16L/16H, FFN 4096, RoPE \theta=5\times10^5, vocab 100,352 |
| Train corpus                 | Mixing Laws Dataset 10B (~9.989B tokens), flat shuffle                                          |
| Global batch / seq / LR      | 4,194,304 / 2048 / 4\times10^{-4} cosine (warmup 24, \alpha_f=0.1)                              |
| Steps                        | 2360 ≈ one epoch under published train length                                                   |
| FLOPs / arm                  | 2.63\times10^{19} (measured from W&B)                                                           |
| Keep rate (where applicable) | top / middle **60%** of valid target tokens per sequence                                        |
| Primary metric               | Macro mean CE bits-per-byte over 20 OLMES-style labels                                          |


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


dolma2 tokenizer. Training FLOPs \approx 1.45\times10^{19}. Used as the frozen scorer for ρ-1 (reference A) and for Middle-PPL late-average scores (mean of late checkpoints).

**Reference B — instruct mix.** Plain CE on a one-pass **English-filtered instruct** corpus (no upsampling to a fixed 5.5B cap; realized size is whatever the filtered unique pass yields). Sources are instruction / chat / math / code SFT-style collections with safety / tool / IFEval-oriented subsets dropped, then Dolma English document-score ≥ 0.5:


| Source family  | Kept (summary)                                                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tulu-v2        | all                                                                                                                                                                              |
| OpenHermes-2.5 | all (optional non-English metadata drop)                                                                                                                                         |
| Tulu-3         | FLAN, WildChat, math/code/science personas, Numina-TIR, Evol CodeAlpaca, SciRIFF, TableGPT, No Robots, OASST, etc.; drop WildGuard / WildJailbreak / CoCoNot / Aya / IF personas |
| Hermes-3       | all                                                                                                                                                                              |
| SmolTalk       | all configs except API-gen / constraint suites                                                                                                                                   |
| Dolci          | keep non-safety / non-tool / non-IF / non-Aya rows                                                                                                                               |


Domains labeled general / math / code / science / chat from metadata. dolma2 tokenizer. Training FLOPs \approx 1.04\times10^{19}. Used as the frozen scorer for ρ-1 (reference B).

### What each method manipulates

**ρ-1 (excess loss).** For each target token, score L_{\mathrm{curr}} - L_{\mathrm{ref}} under the live student vs a **frozen** reference model of the same architecture. Keep the top 60% (largest excess loss — tokens where the student is worst relative to the reference). Selection is active from step 0. Two arms use Reference A vs Reference B checkpoints (same scoring rule). Method from [RHO-1](https://arxiv.org/abs/2404.07965).

**Attention top-k.** No external reference. Score each token by **causal attention received** on the last transformer layer (mean over heads of attention mass from later positions). Keep the top 60%. Active from step 0. Manipulation: prefer tokens that the model currently “attends to” as context for later predictions. Method from [ssToken](https://arxiv.org/abs/2510.18250) (attention-based score).

**Middle-PPL.** Score tokens by **late-checkpoint average** token CE under **Reference A**. Keep the **middle 60%** of the per-sequence score distribution (drop both easiest and hardest tails). Masks can be precomputed. Manipulation: train on “medium difficulty” tokens under that frozen scorer, excluding extremes. Method from [Marion et al., Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/abs/2309.04564).

**BLADE.** Bi-level setup with a **proxy** (trained student) and a **dynamic reference** that is periodically reset from the proxy. Steps 0–499: full CE on the proxy only (no selection). At sync steps 500, 875, 1250, 1625, 2000: copy proxy → reference, run K=75 reference updates, then keep proxy tokens with largest L_{\mathrm{proxy}}-L_{\mathrm{ref}} at keep-rate \gamma=0.6 (\tau=375). After the last sync, hold that reference to the end. Manipulation: select tokens where the fast proxy outruns a lagged copy of itself. Method from [BLADE](https://arxiv.org/abs/2606.18650).

**REL-EMA (exponential).** Online relative loss vs an **EMA of the student** (bias-corrected from zero; no external seed). EMA rate \alpha(t)=1-e^{-t/300}. Score \mathrm{REL}=L_{\mathrm{hist}}-L_{\mathrm{curr}}; keep top 60%. Active from step 0. Manipulation: prefer tokens where the live model is worse than its own exponential history. Method from [ssToken](https://arxiv.org/abs/2510.18250) (retrospective excess loss / REL).

### Arms actually run


| Arm                        | A100-h    | FLOPs                                                                                   |
| -------------------------- | --------- | --------------------------------------------------------------------------------------- |
| REL-EMA (exponential)      | 59.40     | 2.63\times10^{19}                                                                       |
| Attention top-k            | 58.55     | 2.63\times10^{19}                                                                       |
| Middle-PPL (training)      | 42.99     | 2.63\times10^{19}                                                                       |
| Middle-PPL mask precompute | 2.0       | —                                                                                       |
| ρ-1 (reference A)          | 56.56     | 2.63\times10^{19}                                                                       |
| ρ-1 (reference B)          | 56.56     | 2.63\times10^{19}                                                                       |
| BLADE                      | 70.5      | 3.04\times10^{19}                                                                       |
| **Total**                  | **346.6** | **\approx 1.62\times10^{20}** (6× Mixing Laws Dataset arms; mask precompute negligible) |


Control is the Data Mixing Laws paper full-CE run (every token kept). A random-60% selection control was **not** run. A100-hours are the no-waste totals measured for these training arms. Task-loss curves below report the ρ-1 (reference B) arm.

---

## Evaluation and uncertainty

Same as MixLaw: power law y = a + b/\mathrm{step}^{\alpha} on steps ≥ 1000; fitted final as center; residual bootstrap (10k) 95% CI. p-values vs control are omitted because **no selection arm beat control**.

---

## Results

### Fitted final macro task-loss (bpb)


| Arm                              | Fitted final | Observed | 95% CI           |
| -------------------------------- | ------------ | -------- | ---------------- |
| Data Mixing Laws paper (control) | **1.6518**   | 1.6518   | [1.6483, 1.6552] |
| ρ-1 (reference B)                | 1.6824       | 1.6843   | [1.6771, 1.6858] |
| Attention top-k                  | 1.6991       | 1.7005   | [1.6935, 1.7039] |
| BLADE                            | 1.7090       | 1.7115   | [1.7028, 1.7182] |
| Middle-PPL                       | 1.9047       | 1.9001   | [1.9007, 1.9091] |
| REL-EMA (exponential)            | 2.6228       | 2.6510   | [2.5487, 2.6791] |


Lower is better. All selection CIs sit above control.

### Takeaways

1. **Full-token CE wins.** Dropping ~40% of tokens never improved macro task-loss at this budget.
2. **Ranking among failures.** ρ-1 < Attention top-k < BLADE ≪ Middle-PPL ≪ REL-EMA. Excess loss vs a strong frozen reference is least damaging; EMA-relative selection is catastrophic (~+1.0 bpb).
3. **BLADE’s extra machinery did not pay off** — worse than simple ρ-1 despite sync overhead (70.5 A100-h).
4. **Cost.** Token selection was the most expensive lever among the four (346.6 A100-hours for the Mixing Laws Dataset arms; \approx 1.58\times10^{20} FLOPs) and the weakest scientific return.

---

## Conclusions

Under the P1 Mixing Laws Dataset × 370M one-epoch contract, **token selection is a negative result**: every tested scorer underperforms full CE. Prefer mixture optimization (MixLaw) or, secondarily, difficulty curricula over token masking for this setup.