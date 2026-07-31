---
name: Skill-It Probe and Train
overview: Run 7 DataDecide-60M one-hot Skill-It probes at 5 tpp, build a Chinchilla-extrapolated 7x6 A matrix vs RegMix (mixlaw fit), then train two OLMo2-370M x 10B arms that start at RegMix weights and apply the same Skill-It update (eta=0.2, w=1) at five checkpoints—differing only in whether A is the offline probe matrix or online mixing-law derivatives.
todos:
  - id: probes-json
    content: Add skillit/probes.json (7 one-hot domains) and wire mixlaw/build_mixture_data materialization from olmohq
    status: completed
  - id: run-probes
    content: Add launch_probe.sh; run 7 one-hot DataDecide-60M @ 5 tpp with same 6-curve eval cadence as mixlaw pilots
    status: completed
  - id: build-A
    content: "build_adjacency.py: Chinchilla extrapolate to step 5806; A_ij=max(0,L_regmix-L_i); publish artifacts"
    status: completed
  - id: skillit-math
    content: "skillit_math.py: eta=0.2 w=1 softmax update; online A from mixlaw/mixlaw_fit_chinchilla.json as max(0,-dL/dr)"
    status: completed
  - id: domain-stream
    content: "domain_stream.py: olmohq domain-stratified sampler with set_weights(p) for mid-run reweighting"
    status: completed
  - id: train-370m
    content: "train_skillit_370m.py + launch_arm.sh: OLMo2-370M 10B curriculum contract; updates at 500/875/1250/1625/2000; two A_MODE arms"
    status: completed
  - id: docs-s3
    content: "README: architectures, schedules, local paths, S3 prefixes for probes/A/arms; platform-agnostic launch notes"
    status: completed
isProject: false
---

# Skill-It Probe + Dual-Arm 370M Plan

## Locked decisions

- **Probe model:** DataDecide-60M (same as mixlaw pilots)
- **Probe budget:** 5 tpp → **1451 steps / 285,278,208 tokens** (training ends at 1451; step-law fits use in-run evals through **1440** only)
- **Step-law / Chinchilla:** `extrapolate_chinchilla.py` fits on `task_loss.jsonl` only; `task_loss_final.json` @ 1451 is **not** used (different eval protocol; see mixlaw README)
- **A offline:** `A_ij = max(0, L_j(r_RegMix,chin) - L_j(i,chin))` using Chinchilla-extrapolated (tpp=20, step **5806**) family losses; `L_j(r_RegMix)` from `mixlaw_fit_chinchilla.json` at RegMix base weights (same as mixlaw mix01 / `DOMAIN_BASE_WEIGHTS`)
- **A online:** recompute `A(r)` from `[mixlaw/mixlaw_fit_chinchilla.json](experiments/skill-dag/mixlaw/mixlaw_fit_chinchilla.json)` via the mixing-law derivative (see **Mixing-law derivative** below; not LightGBM)
- **Skill-It:** eta=0.2, w=1; start at RegMix weights; updates at steps **500, 875, 1250, 1625, 2000**
- **Full runs:** OLMo2-370M, 10B tokens, curriculum-matching hparams; stream from **olmohq** with time-varying domain weights
- **Platform:** no AWS/FarmShare/GPU SKU hardcoding; `NPROC=1` → python, `NPROC>1` → `torchrun`

## Mixing-law derivative (Skill-It-compatible)

Parametric mixing law (one row per task family `j`, domain weights `r`):

```text
L_j(r) = c_j + k_j * exp( sum_i t_ij * r_i )
```

Partial derivative with respect to domain weight `r_i` (chain rule on the exponential):

```text
dL_j / dr_i = t_ij * k_j * exp( sum_k t_kj * r_k )
            = t_ij * ( L_j(r) - c_j )
```

Skill-It treats `A_ij > 0` as “domain i helps task j.” Helpful domains have **negative** `t_ij` (increasing `r_i` lowers loss). Therefore both arms use:

```text
A_ij = max( 0, -(dL_j / dr_i) ) = max( 0, -t_ij * (L_j(r) - c_j) )
```

evaluated at the **current** domain weights `r` for the online (`skillit-deriv`) arm. (Raw `dL/dr` would invert Skill-It.) Fitted parameters `(c_j, k_j, t_ij)` come from `mixlaw_fit_chinchilla.json`; `skillit_math.py::online_A_from_mixlaw_fit` implements the formula above.

---

## Phase 0 — Repo layout

`[experiments/skill-dag/](experiments/skill-dag/)` is the parent experiment directory. After the mixlaw reorganization it contains:

- `[mixlaw/](experiments/skill-dag/mixlaw/)` — existing 24-mixture pilot (DataDecide-60M, mixing-law fits, validation tooling)
- `[skillit/](experiments/skill-dag/skillit/)` — **new** Skill-It probes + 370M arms (sibling of `mixlaw/`, not nested inside it)

New code under `[experiments/skill-dag/skillit/](experiments/skill-dag/skillit/)`:

- `skillit/probes.json` — 7 one-hot mixture definitions
- `skillit/build_adjacency.py` — Chinchilla-extrapolate probes → write `A_offline.npy` + JSON
- `skillit/skillit_math.py` — Shared: softmax update, offline A load, online derivative A from mixlaw fit
- `skillit/domain_stream.py` — Domain-stratified chunk sampler with time-varying `p`
- `skillit/train_skillit_370m.py` — OLMo2-370M trainer (fork of curriculum control path)
- `skillit/launch_probe.sh` — Platform-agnostic probe launcher
- `skillit/launch_arm.sh` — Platform-agnostic 370M arm launcher
- `skillit/README.md` — Contract, schedules, S3 layout

Reuse from sibling `[experiments/skill-dag/mixlaw/](experiments/skill-dag/mixlaw/)` (import or subprocess; do not duplicate):

- `mixlaw_common.py` — domains, curve families, token budget math, `DOMAINS`, `CURVE_FAMILIES`
- `build_mixture_data.py` — materialize probe slices from olmohq
- `train_datadecide_60m.py` — 60M probe training
- `eval_task_loss.py` — post-run 6-family eval
- `extrapolate_chinchilla.py` — step-law extrapolation to Chinchilla step 5806
- `run_mixture.sh` — launch pattern reference
- `mixlaw_fit_chinchilla.json` — parametric mixing-law fit for online-derivative arm

Update `[experiments/skill-dag/README.md](experiments/skill-dag/README.md)` to list `skillit/` alongside `mixlaw/`.

---

## Phase 1 — Probe runs (7× DataDecide-60M one-hot)

### Architecture (exact pilot match)

From `[mixlaw/mixlaw_common.py](experiments/skill-dag/mixlaw/mixlaw_common.py)`:

- d_model / layers / heads: **384 / 16 / 12**
- mlp_ratio / seq_len: **8 / 2048**
- Global batch: **96 sequences** (196,608 tok/step)
- LR: **5.8e-3** (CosWithWarmupAndDecay)
- Non-emb params: **57,078,144**
- Tokenizer: dolma2 (100,352 embed rows)
- Init: random (`InitFnType.normal`, std=0.02)
- tpp: **5** → 1451 steps

### Mixtures (`skillit/probes.json`)

Domain order: `dclm, arxiv, starcoder, pes2o, open-web-math, algebraic-stack, wiki` (same as `mixlaw/DOMAINS`).

- `probe_dclm` … `probe_wiki` — one-hot per domain

No uniform 1/7 probe: offline **A** compares each one-hot domain to **RegMix** via the mixing-law reference `L_j(r_RegMix)` (not `probe_uni`).

### Data

- Source pool: `s3://edullm-datasets/olmo100b/olmo-mix-1124-30b` (olmohq), same working-pool tokenize path as mixlaw pilots
- Materialize via `mixlaw/build_mixture_data.py` with `--mixtures ../skillit/probes.json` (or copy probes.json into mixlaw work dir)
- Local work dir (env `SKILLIT_PROBE_WORK`): `$WORK/skillit-probes/`
- Published probe artifacts: `s3://edullm-datasets/skillit/probes/` (slice plans + `paths_train.txt` per probe)
- Progress/logs: `s3://edullm-checkpoints/skillit/probes/<probe_id>/` (optional sync; weight ckpts stay local)

### Evals (exact mixlaw pilot match)

Skill-It probes call [`mixlaw/train_datadecide_60m.py`](experiments/skill-dag/mixlaw/train_datadecide_60m.py) with the same defaults as [`mixlaw/run_mixture.sh`](experiments/skill-dag/mixlaw/run_mixture.sh) — no overrides.

**In-run curve eval (for Chinchilla step-law fitting):**

- **Labels:** 6 `CURVE_TASK_LOSS_LABELS` (ARC challenge/easy val + 4 MMLU val families)
- **`eval_interval`:** **120** training steps
- **`eval_subset_batches`:** **4** (max batches per label per eval; keeps in-run eval cheap)
- **`device_eval_batch_size`:** **32**
- **Points per 1451-step run:** **~12** curve points (steps 120, 240, …, **1440**)
- **Output:** `progress/task_loss.jsonl` (one row per eval step; scraped by `TaskLossHandler`)

Do **not** pass `--full-task-suite-in-run` or `--skip-eval` on probe runs.

**Post-training eval (reporting only — not used for step-law fit):**

- Once at end of training via [`mixlaw/eval_task_loss.py`](experiments/skill-dag/mixlaw/eval_task_loss.py) on the final checkpoint (step **1451**)
- Same 6 curve labels (default; not `--full-suite`)
- Output: `progress/task_loss_final.json`
- **Not** appended to the Chinchilla step-law fit: it uses a different eval protocol (full eval vs `eval_subset_batches=4` in-run) and often disagrees with the last in-run point at step 1440 (notably `mmlu_stem`). Step laws use **jsonl points only**, then extrapolate to step **5806** (tpp=20) via [`mixlaw/extrapolate_chinchilla.py`](experiments/skill-dag/mixlaw/extrapolate_chinchilla.py).

**Not run on probes:** the full 20-label ladder suite (that is reserved for 370M arms at checkpoint saves).

### Launch

`skillit/launch_probe.sh` mirrors curriculum style: env `PROBE_ID`, `SAVE_FOLDER`, `NPROC` (1 or >1), no cluster SKU. Calls `mixlaw/train_datadecide_60m.py` against materialized slices.

---

## Phase 2 — Build offline A (7x6)

```mermaid
flowchart LR
  probes[7 one-hot probe curves] --> stepLaw["Fit L(s)=L_inf+A*s^(-alpha) on jsonl only"]
  stepLaw --> chin["Extrapolate to step 5806"]
  fit["mixlaw_fit_chinchilla.json L_j(r_RegMix)"] --> delta["A_ij = max(0, L_regmix_j - L_i_j)"]
  chin --> delta
  delta --> Aout["A_offline.npy + adjacency.json"]
```

1. Reuse `[mixlaw/extrapolate_chinchilla.py](experiments/skill-dag/mixlaw/extrapolate_chinchilla.py)` logic per probe / per `CURVE_FAMILIES` member → losses at step **5806** (tpp=20).
2. Evaluate `L_j(r_RegMix)` from `[mixlaw/mixlaw_fit_chinchilla.json](experiments/skill-dag/mixlaw/mixlaw_fit_chinchilla.json)` at RegMix base weights (`base_weights` in `probes.json`).
3. `skillit/build_adjacency.py`:
  - `A_ij = max(0, L_j(r_RegMix) - L_j(i))` for domains i, families j
  - Write `skillit/artifacts/A_offline.npy` (shape 7x6), `adjacency.json` (named rows/cols + reference losses)
4. Publish: `s3://edullm-checkpoints/skillit/artifacts/A_offline.json` (+ npy)

### Measured A matrices (7 one-hot probes, Chinchilla step 5806)

From the July 2026 FarmShare probe array (`skillit-probes-20260729-112123`). Rows = source domain **i** (one-hot probe); columns = task family **j**. Positive **A_ij** means domain **i** is projected to beat the reference on family **j** at Chinchilla scale (tpp=20, step **5806**).

**RegMix reference losses** `L_j(r_RegMix)` from `mixlaw_fit_chinchilla.json` at `base_weights`:

| Family | L_j(RegMix) |
|--------|-------------|
| arc_challenge | 1.546 |
| arc_easy | 2.001 |
| mmlu_humanities | 1.731 |
| mmlu_other | 2.467 |
| mmlu_social_sciences | 1.305 |
| mmlu_stem | 2.297 |

#### Offline A (`skillit-probe` arm)

One-hot probes extrapolated to step 5806; reference = RegMix mixing-law losses.

```text
A_ij = max(0, L_j(r_RegMix) - L_j(i,chin))
```

| domain ↓ / family → | arc_ch | arc_easy | mmlu_hum | mmlu_oth | mmlu_soc | mmlu_stem |
|---------------------|--------|----------|----------|----------|----------|-----------|
| dclm | 0.341 | 0.149 | 0.368 | 0 | 0.079 | 0.486 |
| arxiv | 0.006 | 0 | 0 | 0 | 0 | 0.340 |
| starcoder | 0 | 0 | 0 | 0 | 0 | 0 |
| pes2o | 0.252 | 0.084 | 0 | 0 | 0 | 0.450 |
| open-web-math | 0 | 0 | 0 | 0 | 0 | 0.232 |
| algebraic-stack | 0.017 | 0 | 0 | 0 | 0 | 0.242 |
| wiki | 0.227 | 0 | 0.470 | 0.013 | 0 | 0.352 |

Artifacts: `skillit/artifacts/probes_full/A_offline.npy`, `A_offline.json`. Curves: `task_loss_chinchilla_by_family.png`. Recompute: `python skillit/build_adjacency.py` or `python skillit/plot_probe_chinchilla_results.py`.

---

## Phase 3 — Full experimental arms (2x OLMo2-370M)

### Shared training contract (match curriculum)

From `[train_curriculum_regmix_370m.py](experiments/curriculum/train_curriculum_regmix_370m.py)` / control:

- Model: `TransformerConfig.olmo2_370M` (d=1024, 16L, 16H)
- Seq / GBS: 2048 / **4,194,304** tokens
- Microbatch: 32 seqs/rank (`--device-batch-size`)
- Steps: **2384** (10,000,058,051 // GBS)
- Optim: SkipStepAdamW, LR **4e-4**, warmup **24**, `alpha_f=1.0` (constant after warmup)
- Seed: **42**
- Checkpoints: every **125** steps + 0 + 2384; omit 2375
- Task loss: **all 20** ladder bpb labels at every permanent checkpoint

### Domain sampling (required for mid-run reweight)

Do **not** use fixed RegMix-10B concat shuffle (weights baked into corpus sizes).

- Stream from olmohq tokenized pool with **domain-stratified** sampling: at each step draw domain ~ `p_t`, then a random 2048-token chunk from that domain’s memmap (`domain_stream.py`).
- Materialize / cache a working pool sized for 10B with headroom (reuse `mixlaw/prepare_data.sh` / working-pool tooling); S3:
  - Pool: `s3://edullm-datasets/skillit/train-pool/` (or reuse existing olmohq tokenized working pool if already present)
  - Per-arm logs: `s3://edullm-checkpoints/skillit/<arm_id>/`

### Arms (only A differs)

- `skillit-probe` — Fixed offline A from Phase 2
- `skillit-deriv` — Recompute `A(r_t)` from `mixlaw/mixlaw_fit_chinchilla.json` at current `r_t` (= `p_before`) before each update, using the mixing-law derivative:

```text
dL_j / dr_i = t_ij * ( L_j(r) - c_j )
A_ij        = max( 0, -(dL_j / dr_i) )
```

Both:

1. **Steps 0–499:** `p` = RegMix base weights
  `(0.375, 0.25, 0.1406, 0.0938, 0.0635, 0.0615, 0.0156)`
2. **At steps 500, 875, 1250, 1625, 2000** (after that step’s 20-label eval):
  - Read 6 curve-family losses `L_j(f_t)` from the checkpoint eval
  - Build / load A (7x6)
  - Update (w=1):

```text
p_i(t+1)  proportional to  exp( eta * sum_j A_ij * L_j(f_t) )
eta = 0.2
```

- Softmax over 7 domains; apply from the **next** step onward

1. **Persist every update** (both arms; required for post-hoc review):
  - Append one JSONL record to `progress/skillit_updates.jsonl` with:
    - `step`, `arm_id`, `a_mode` (`probe` | `derivative`)
    - `losses`: dict of the 6 curve-family bpb values used in the update
    - `A`: full 7x6 matrix as nested lists, plus `domain_order` and `family_order`
    - `p_before`: domain weights in effect before this update
    - `p_after`: domain weights after softmax (used from next step)
    - For `derivative` only: `r` at which `dL/dr` was evaluated (equals `p_before`)
  - Also write a per-update snapshot file for easy inspection:
    - `progress/skillit_updates/step{N}_A.json` — full A with named rows/cols
    - `progress/skillit_updates/step{N}_weights.json` — `p_before`, `p_after`, losses
  - At step 0 (train start), write the initial RegMix `p` once to the same JSONL / `step0_weights.json` (A may be the offline matrix or the derivative at RegMix `r`, recorded for baseline comparison even though no weight change occurs yet)
2. Sync `progress/skillit_updates.jsonl` and `progress/skillit_updates/` with other arm artifacts to `s3://edullm-checkpoints/skillit/<arm_id>/` when live S3 upload is enabled (see **S3 export** below)

### Trainer hook

`train_skillit_370m.py` forks the curriculum trainer’s model/optim/checkpoint/task_loss path; replace data stream with `DomainMixtureStream(p)` that accepts `set_weights(p)` at update steps. Single- and multi-GPU via same grad-accum / HSDP pattern as curriculum (`NPROC` in `launch_arm.sh`).

### Launch

```bash
# single GPU
NPROC=1 ARM_ID=skillit-probe A_MODE=probe \
  SAVE_FOLDER=... PROGRESS_DIR=... POOL_DIR=... \
  bash experiments/skill-dag/skillit/launch_arm.sh

# multi-GPU
NPROC=8 ARM_ID=skillit-deriv A_MODE=derivative ...
```

No Slurm/AWS submit baked in. Optional post-save S3 sync follows the token-selection pattern (see **S3 export** below).

---

## Phase 4 — Analysis artifacts

- Per-arm: checkpoint ladder task_loss JSONs (20 labels), weight trajectories, final macro curve-6
- Per-arm Skill-It logs (required):
  - `progress/skillit_updates.jsonl` — one record per update with full `A` (7x6) and `p_before` / `p_after`
  - `progress/skillit_updates/step{N}_A.json` and `step{N}_weights.json` — same data as discrete files
  - Offline arm: A is constant across updates but still re-saved each time so both arms have identical log schemas
  - Derivative arm: A changes with `r`; each update’s A is independently recoverable
- Compare `skillit-probe` vs `skillit-deriv` vs existing RegMix control (`curriculum` control / token-selection control) on the 6 Skill-It families and full 20-label suite
- Small script `skillit/plot_weights.py` for `p_t` trajectories (reads `skillit_updates.jsonl`)

---

## Implementation order

1. `skillit/probes.json` + materialize 7 slices via `mixlaw/build_mixture_data.py`
2. Run 7 one-hot probes via `mixlaw/train_datadecide_60m.py`; extrapolate via `mixlaw/extrapolate_chinchilla.py`; `skillit/build_adjacency.py` → offline A vs RegMix
3. `skillit/skillit_math.py` (update + derivative A from `mixlaw/mixlaw_fit_chinchilla.json`) unit-tested against toy numbers
4. `skillit/domain_stream.py` + `skillit/train_skillit_370m.py` + launch scripts
5. Update `experiments/skill-dag/README.md`; launch two 370M arms; sync artifacts to S3 prefixes above

---

## S3 export (`S3_EXPORT`)

`S3_EXPORT` is **not** a Skill-It-specific concept — it is an env var used by the token-selection trainers ([`token_selection/olmo_ext/s3_export.py`](experiments/token-selection/token_selection/olmo_ext/s3_export.py)) to control whether the training host automatically runs `aws s3 sync` after checkpoint saves.

- **unset or `1` (default):** Sync checkpoints/progress to S3 when `aws` CLI is on PATH and credentials exist
- **`0`, `false`, `no`, `off`:** Disable live S3 uploads (artifacts stay local only)

Also disabled when `SKIP_S3_UPLOAD=1`. The Skill-It 370M trainer should reuse this helper (or equivalent) for optional artifact sync — it is **not required** for training to work.

Probe runs: progress/logs may sync to `s3://edullm-checkpoints/skillit/probes/` via `RESULTS_S3` in `run_mixture.sh` style, or stay local; 60M weight checkpoints are not uploaded (same as mixlaw pilot).

---

## Out of scope

- LightGBM surrogates for A (`mixlaw/mixlaw_fit_lightgbm_chinchilla.json`)
- Pairwise (i,j) vs j-only probing
- Behavioral clustering
- Changing LR schedule, GBS, or checkpoint ladder relative to curriculum

