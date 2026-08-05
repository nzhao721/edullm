# P1 Experiments Runbook

Fifteen full-scale OLMo2-370M training runs across four research levers, reduced from an original 35+ arm grid for compute constraints. All runs use **8× A100**, **~5 h 20 m** wall time each (**~42.7 GPU-hours**). Total: **~640 GPU-hours**.

**Already complete (not in the 15-run budget):** MixLaw 60M pilot (24 mixtures), SkillIt 60M probes (7 domains), RefHQ reference model.

---

## OLMo-core baseline

All four branches train through **OLMo-core’s standard Trainer stack**. The **control** is full-token cross-entropy on a default 10B corpus (OLMo's dataset with domain weights chosen as one paper's claimed optimum).

Each lever then changes **one axis** while holding architecture, global batch size (4.19M tokens/step), and (within a lever) optimizer settings fixed.

---

### MixLaw: find a better *static* domain mix

Pretraining corpora are blends of domains (web crawl, code, math, papers, etc.). The default OLMo mix (`olmo-mix-1124`) reflects one published recipe. **MixLaw** asks: given a fixed 10B-token budget, is there a better *fixed* blend?

The pipeline:

1. Train **24 small proxy models** (DataDecide-60M) on different 7-domain mixtures.
2. Measure each proxy on six **downstream task-loss curves** (ARC + MMLU families).
3. Fit a **mixing law**: a parametric model that predicts task loss from domain weights, and a **LightGBM surrogate** on Chinchilla-extrapolated targets.
4. **Optimize** mixture weights offline under feasibility constraints

Full 370M runs then train once at the chosen static weights.

### SkillIt: *adapt* the domain mix during training

MixLaw picks weights upfront. **Skill-It** (Chen et al.) updates them **during** training using checkpoint eval signals.

At steps 500, 875, 1250, 1625, and 2000 the trainer:

1. Runs the standard 20-benchmark task-loss eval on the current checkpoint.
2. Maps eval losses and an **adjacency matrix A** (7 domains × 6 task families) into new domain sampling weights via a softmax update (η = 0.2).
3. Continues training with the updated weights — same tokens and pool, different domain proportions.

**A** encodes which domains are believed to help which task families. The two arms differ only in how **A** is obtained: fixed from 60M one-hot probe runs, or recomputed from mixing-law derivatives at the *current* weights. Both start from RegMix base weights.

### Curriculum: order chunks by difficulty

Instead of shuffling RegMix 10B uniformly, **curriculum learning** pre-ranks every 2048-token chunk by a **difficulty score**, then controls **when** each difficulty level is sampled.

- **Difficulty metrics** (ablated): Flesch reading ease (text readability), MTLD (lexical diversity), or learnability (gap between early and late model loss on the chunk for a reference model).
- **Pacing schedules** (ablated): linear progression through N=10 difficulty buckets over 250-step segments; **naive sequential** easy→hard for a warmup prefix then shuffle; or interleaved mini-curricula within each segment.

Only the **sampling order** over the same pre-tokenized corpus differs from a **random shuffle** (flat-shuffle) control. Learning rate is constant and checkpoints are merged via EMA over late steps to reduce noise in the final model.

### Token selection: train on 60% of tokens

**Token selection** masks **40%** of tokens from the loss each step and backprops only on the remaining **60%**: same model and corpus, sparser gradient signal.

Arms differ in **which** tokens to keep:


| Strategy        | Intuition                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------ |
| **BLADE**       | Keep tokens where a small proxy model is most wrong relative to the main model (dynamic, synced proxy).      |
| **ρ-1 (rho-1)** | Keep tokens with highest excess loss vs. a frozen reference model.                                           |
| **REL-EMA**     | Keep tokens where current loss dropped most vs. an EMA of past losses (emphasize "newly learned" positions). |
| **Middle-PPL**  | Keep the *middle* 60% by reference model perplexity.                                                         |
| **Attention**   | Keep tokens that receive the most causal attention mass (ssToken-style).                                     |


---

## Status summary


| Lever           | Done                  | To run                             |
| --------------- | --------------------- | ---------------------------------- |
| MixLaw          | 60M pilot + fits      | 3 arms (+ `mix01` baseline)        |
| SkillIt         | Probes + offline A    | 2 arms                             |
| Curriculum      | Trainer + pacing code | 5 arms (needs label/index publish) |
| Token selection | RefHQ ref + trainers  | 5 arms                             |


