# Curriculum learning experiment plan (OLMo2-370M / RegMix 10B)

## Goal

Train from scratch on the RegMix 10B corpus with **fixed token budget** and **one shared training script** parameterized by `--pacing` and `--difficulty-metric`. Only the **data sampler / batch stream** differs across arms. All arms use **warmup + constant LR** and **post-hoc EMA** over checkpoints at steps **2000, 2125, 2250, 2384** (α=0.8).

## Execution environment

**Ephemeral runtime.** Training assumes job-scoped scratch starts empty and is wiped after the job. Stage train/curriculum bytes from validated `s3://edullm-data/` into a job-local cache. Checkpoints, progress, metrics, and task-loss outputs remain on runtime scratch and upload to Weights & Biases project `curriculum`; no run artifact is written to S3. At each permanent step the order is local checkpoint → synchronous all-rank 20-label eval → awaited W&B eval/checkpoint/runtime-state uploads → `last_wandb_step.json`. A production upload failure aborts all ranks. Local smoke only: `--wandb-mode disabled --allow-local-only`, and explicitly disable task loss if the eval stack is unavailable. Every launch selects recovery explicitly: `FRESH=1` / `--fresh`, or `LOAD_PATH` / `--load-path` pointing at a local step dir or `wandb-artifact://entity/project/name:version`.

Instance type, GPU model, and GPU count are **not chosen in this plan** — scripts must work on any CUDA host with 1–N GPUs via `torchrun`.

- Data: resolve+stage from `s3://edullm-data/` (`pretrain/regmix-10b`, `curriculum/regmix-370m`)
- Checkpoints / progress / metrics / task_loss: job-local scratch write → W&B artifacts only
- Production requires online W&B; each checkpoint artifact upload is awaited and fail-closed
- Resume via `--load-path` (local step dir or `wandb-artifact://entity/project/name:version`)
- No slurm/FarmShare-specific launch paths, queue names, or GPU SKU assumptions in required code paths
- `--device-batch-size` and grad accumulation are CLI parameters; derive steps from global batch `4_194_304` and discovered `world_size`
- Curriculum index build is a CPU job; runnable locally, on AWS, or any host with S3 access — not tied to training hardware

Label **generation** lives in **`datasets/regmix/`**; publish token corpora / curriculum orders via the `edullm-data` package. Training code in `experiments/curriculum/` consumes published `edullm-data` artifacts only (legacy dataset buckets are refused).

## Training contract

Fork [`experiments/token-selection/control/train_ce_regmix_olmo_370m.py`](experiments/token-selection/control/train_ce_regmix_olmo_370m.py) and reuse shared helpers from [`experiments/token-selection/token_selection/olmo_ext/`](experiments/token-selection/token_selection/olmo_ext/) for the checkpoint ladder and task-loss wiring. Curriculum artifact publication is W&B-only.

### Architecture

- **Config factory**: `TransformerConfig.olmo2_370M` (not `olmo3_370M` / SWA)
- **`d_model` / layers / heads**: 1024 / 16 / 16
- **Block**: `reordered_norm`
- **MLP**: gated SiLU FFN, hidden 4096
- **Attention**: full (no sliding window); QK-RMSNorm; RoPE θ = 500_000
- **Vocab / tokenizer**: 100_352 / `allenai/dolma2-tokenizer`
- **Sequence length**: 2048
- **Global batch tokens**: `4_194_304`
- **Rank microbatch tokens**: `65_536` default (32 seqs at seq 2048); override via `--device-batch-size`
- **Optim**: SkipStepAdamW
- **`z_loss_multiplier`**: `1e-5`
- **`max_grad_norm`**: 1.0
- **Compile**: `compile_model=True`
- **Init**: from scratch
- **DP**: HSDP bf16 train module

Reference implementation: [`experiments/token-selection/reference/train_olmo3_370m_refhq.py`](experiments/token-selection/reference/train_olmo3_370m_refhq.py).

### Hyperparameters

- **Peak LR**: `4e-4`
- **LR warmup**: 24 steps
- **LR after warmup**: constant peak (`alpha_f=1.0`)
- **Token budget**: ~10B → **2384 steps**
- **Train corpus**: published `pretrain/regmix-10b` on `s3://edullm-data/`
- **Curriculum orders**: published `curriculum/regmix-370m` with groups `compression`, `flesch`, `mtld`, `learnability` (fail closed if unpublished)

Control arm reads flat token memmaps from `pretrain/regmix-10b`; curriculum arms read the matching order group from `curriculum/regmix-370m` and apply in-process pacing.

### Checkpoint saving

Ladder via `permanent_checkpoint_steps()` from [`token_selection.olmo_ext.checkpoint_ladder`](experiments/token-selection/token_selection/olmo_ext/checkpoint_ladder.py):

- Step **0** (pre-train snapshot)
- Every **125** steps: 125, 250, …, `125 * floor(total_steps / 125)`
- Always save the **true final step** (2384)
- Omit last on-grid step when within one interval of final → `{0, 125, …, 2250, 2384}` — omit 2375
- `max_checkpoints=None` — permanent saves only; no ephemeral pruning
- Checkpoint format `model_and_optim` / `full_state_dict_v1`
- After each permanent save: synchronous, fail-closed W&B model artifact upload; progress/eval/metrics snapshots are W&B run-state artifacts

### Task-loss eval

- Metric: `task_loss_bpb` = `-log2 p(gold continuation | context) / utf8_bytes(continuation)`
- Full **20-label** `*_rc_5shot_bpb` suite (ARC, BoolQ, CSQA, HellaSwag, OpenBookQA, PIQA, SocialIQA, WinoGrande, MMLU×4)
- Trigger synchronously on every permanent checkpoint save (step 0 + each ladder step + final)
- All ranks pause, release the HSDP train module, evaluate in lockstep through the shared `pause_eval_reload_distributed` helper, then rebuild/reload the saved checkpoint
- Production is strict and requires `LADDER_BASE_CONFIG`; local smoke may explicitly use `--no-task-loss-on-save`
- Evaluator script: [`scripts/farmshare/task_loss/eval_task_loss_olmo_core.py`](scripts/farmshare/task_loss/eval_task_loss_olmo_core.py) (repo path only; runs on the training/eval worker)
- Outputs: job-local `$PROGRESS_DIR/task_loss_results/step{N}_task_loss.json` → W&B eval and run-state artifacts

Post-hoc EMA merge runs the 20-label eval on the merged artifact in addition to per-checkpoint evals.

### Hardware / launch

- World size from `torchrun` / `LOCAL_RANK` / `WORLD_SIZE` (or 1 if unset)
- No hardcoded GPU indices, `CUDA_VISIBLE_DEVICES`, instance types, or host paths as required defaults
- Global batch `4_194_304`; per-rank microbatch / grad-accum from `world_size` and `--device-batch-size`; fail-fast if not divisible
- Eval scripts: 1+ GPU compatible on the same worker policy as training

### Artifact layout

Runtime scratch contains `checkpoints/`, `progress/`, `metrics/`, and
`progress/task_loss_results/`. W&B receives one model artifact per permanent
checkpoint, one eval artifact per completed task-loss suite, and versioned
run-state artifacts containing progress, task-loss, and train-metric files.
Production calls `Artifact.wait()` before advancing past each checkpoint.

## Local repo layout (`C:\alpha_ai\edullm`)

### `datasets/regmix/` — labeling and publish inputs

Document labeling stays in the regmix dataset package; publish validated corpora into `edullm-data` (not a legacy raw-datasets bucket):

```
datasets/regmix/
├── submit_regmix_labeling.sh
├── submit_regmix_doc_lm_labeling.sh
├── label_regmix_shard.sbatch
├── label_regmix_doc_lm.py
├── finalize_regmix_lm_labels.py
├── finalize_regmix_upload.py
├── finalize_regmix_labels_upload.py
└── submit_regmix_labels_upload.sh
```

### `experiments/curriculum/` — training experiment code

```
experiments/curriculum/
├── train_curriculum_regmix_370m.py
├── curriculum_pacing.py
├── ema_merge_checkpoints.py
├── scripts/
│   └── build_curriculum_index.py    # local index staging → publish via edullm-data
├── launch/
│   ├── launch_arm.sh
│   └── submit_matrix.sh
└── tests/
    ├── test_pacing.py
    ├── test_ema_merge.py
    ├── test_index_join.py
    └── test_training_defaults.py
```

**Dependencies** (existing code, imported — not duplicated):

- [`experiments/token-selection/control/train_ce_regmix_olmo_370m.py`](experiments/token-selection/control/train_ce_regmix_olmo_370m.py) — trainer fork source
- [`experiments/token-selection/token_selection/olmo_ext/`](experiments/token-selection/token_selection/olmo_ext/) — checkpoint ladder and task_loss
- [`datasets/regmix/`](datasets/regmix/) — labeling (no duplicate upload scripts under `experiments/curriculum/`)

Published training inputs are `edullm-data` dataset IDs (`pretrain/regmix-10b`, `curriculum/regmix-370m`), not paths under a legacy datasets bucket.

## Data on S3 (`edullm-data`)

- **`pretrain/regmix-10b`**: tokenized parent pool (control + curriculum arms)
- **`curriculum/regmix-370m`**: four `token-order/v1` groups (`compression`, `flesch`, `mtld`, `learnability`) over `pretrain/regmix-10b` (fail closed if missing from `_catalog/`)

| `--difficulty-metric` | order group |
|-----------------------|-------------|
| `compression_ratio`   | `compression` |
| `flesch`              | `flesch` |
| `mtld`                | `mtld` |
| `learnability`        | `learnability` |

Trainers resolve `curriculum/regmix-370m` with `dataset_paths(..., group=<name>, split=train)`. The group defaults from `--difficulty-metric`; override with `--curriculum-order-group` when needed. The order group must bind the exact staged parent dataset version and `manifest_sha256`; a same-length order for another parent version is rejected.

### Parent-pool flat chunk coordinates

Production order vectors are complete permutations of the exact chunks exposed by the published parent:

1. Walk `pretrain/regmix-10b` train shards in `dataset_paths()` order.
2. Each shard contributes `(shard_tokens - 1) // 2048` independently mapped chunks.
3. Concatenate those shard-local ranges into flat IDs `0..N-1`.
4. Map each flat chunk to the labeled document containing the chunk's first token; sort by that document's metric rank, breaking ties by flat ID.

`build_curriculum_index.py` requires a local parent-layout descriptor captured from the published parent. It must include `dataset_id`, pinned `version`, `manifest_sha256`, `seq_len`, `tokenizer_id`, `eos_token_id`, `source_total_tokens`, and ordered `shards` with `path`, `source`, `source_token_start`, and `count`. Label rows must retain one unambiguous `source_path` per source, cover contiguous `source_doc` ordinals, and match exact `n_tokens + EOS` totals. Missing offsets, incomplete metric coverage, tokenizer/hash/version mismatch, or a non-permutation fails closed. The legacy document-local `--curriculum-index` path is rejected.

**17 arms:** 1 control + 4 pacing × 4 metrics (`compression_ratio`, `flesch`, `mtld`, `learnability`).

**Pacing:** `linear_n10`, `expanding_25_1000`, `warmup_1000`, `interleave_i10_linear` — 250-step segments aligned to checkpoint ladder (last segment 134 steps).

## Phases

### Phase 0: Label + publish (`datasets/regmix/` → `edullm-data`)

- Label runs produce `labels/` and `lm_labels/`
- Publish validated corpora / curriculum orders into `s3://edullm-data/` via the `edullm-data` package

### Phase 1: Build curriculum index (`experiments/curriculum/`)

- `experiments/curriculum/scripts/build_curriculum_index.py` — merge labels and map ranks into a pinned published-parent flat chunk layout (no production re-tokenization)
- `datasets/regmix/publish_regmix_curriculum_edullm_data.py` — stage four order groups → `publish()` as `curriculum/regmix-370m`
- `datasets/regmix/submit_publish_regmix_curriculum_edullm_data.sh` — FarmShare Slurm publish (after index build)
- CPU job; local staging only by default; publish resulting token-order dataset into `edullm-data`

### Phase 2: Pacing library

- `experiments/curriculum/curriculum_pacing.py` + `tests/test_pacing.py`

### Phase 3: Shared trainer

- `experiments/curriculum/train_curriculum_regmix_370m.py` — fork from control trainer; `--lr-alpha-f 1.0`
- Checkpoint ladder via `permanent_checkpoint_steps()`; fail-closed W&B checkpoint artifacts; task loss via shared `pause_eval_reload_distributed`
- `experiments/curriculum/tests/test_training_defaults.py` — ladder `{0,125,…,2250,2384}` omits 2375; GBS/microbatch defaults; edullm-data binding

### Phase 4: Post-hoc EMA

- `experiments/curriculum/ema_merge_checkpoints.py` — merge steps 2000/2125/2250/2384, α=0.8
- Use checkpoints still on job scratch or download the required W&B model artifact versions into a scratch work dir first
- Production uploads the merged checkpoint and EMA task-loss result to W&B and awaits both artifact commits; `--allow-local-only` is for local smoke only

### Phase 5: Launch matrix

- `experiments/curriculum/launch/launch_arm.sh` — `python` (1 rank) or `torchrun` (N ranks); all paths via CLI/env; job-scoped scratch
- `experiments/curriculum/launch/submit_matrix.sh` — thin wrapper for job submission (service chosen at launch time)
- Arm matrix: see [Arm matrix (17 arms)](#arm-matrix-17-arms) below

### Phase 6: Smoke test → full runs

- Short smoke (single rank) for control + one curriculum arm on empty scratch
- Full 17-arm matrix after `pretrain/regmix-10b` and curriculum orders are published on `edullm-data`

## Arm matrix (17 arms)

- **`control`**: pacing `control`; difficulty metric —
- **`linear10-cr`**: pacing `linear_n10`; difficulty metric `compression_ratio`
- **`linear10-flesch`**: pacing `linear_n10`; difficulty metric `flesch`
- **`linear10-mtld`**: pacing `linear_n10`; difficulty metric `mtld`
- **`linear10-learn`**: pacing `linear_n10`; difficulty metric `learnability`
- **`expand-cr`**: pacing `expanding_25_1000`; difficulty metric `compression_ratio`
- **`expand-flesch`**: pacing `expanding_25_1000`; difficulty metric `flesch`
- **`expand-mtld`**: pacing `expanding_25_1000`; difficulty metric `mtld`
- **`expand-learn`**: pacing `expanding_25_1000`; difficulty metric `learnability`
- **`warmup-cr`**: pacing `warmup_1000`; difficulty metric `compression_ratio`
- **`warmup-flesch`**: pacing `warmup_1000`; difficulty metric `flesch`
- **`warmup-mtld`**: pacing `warmup_1000`; difficulty metric `mtld`
- **`warmup-learn`**: pacing `warmup_1000`; difficulty metric `learnability`
- **`interleave-cr`**: pacing `interleave_i10_linear`; difficulty metric `compression_ratio`
- **`interleave-flesch`**: pacing `interleave_i10_linear`; difficulty metric `flesch`
- **`interleave-mtld`**: pacing `interleave_i10_linear`; difficulty metric `mtld`
- **`interleave-learn`**: pacing `interleave_i10_linear`; difficulty metric `learnability`

Print the matrix: `bash experiments/curriculum/launch/submit_matrix.sh --print-only`

Example launches (job-scoped scratch; W&B project `curriculum` is the only production artifact backend):

```bash
RUN_DIR="${TMPDIR:-/tmp}/curriculum-job-$$"
mkdir -p "$RUN_DIR"/{ckpts,progress,cache}

# FarmShare: mint on laptop, push session envs into RUN_DIR (SmolLM protocol)
# bash scripts/farmshare/push_aws_session_to_farmshare.sh "$RUN_DIR"
# bash scripts/farmshare/push_wandb_session_to_farmshare.sh "$RUN_DIR"
export WANDB_PROJECT=curriculum WANDB_MODE=online

# Control
ARM_ID=control PACING=control \
  FRESH=1 \
  SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
  DATA_CACHE_DIR=$RUN_DIR/cache \
  bash experiments/curriculum/launch/launch_arm.sh

# Curriculum example (linear + compression_ratio)
ARM_ID=linear10-cr PACING=linear_n10 DIFFICULTY_METRIC=compression_ratio \
  FRESH=1 LADDER_BASE_CONFIG=/path/to/ladder-config.yaml \
  SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
  DATA_CACHE_DIR=$RUN_DIR/cache \
  bash experiments/curriculum/launch/launch_arm.sh

# Local smoke (no durable sink)
# WANDB_MODE=disabled ALLOW_LOCAL_ONLY=1 bash …/launch_arm.sh

# Optional post-hoc EMA (from scratch checkpoints or downloaded W&B artifacts)
python experiments/curriculum/ema_merge_checkpoints.py \
  --checkpoints-root /path/to/staged/linear10-cr/checkpoints \
  --arm-id linear10-cr
```

### W&B naming

| Field | Default |
| --- | --- |
| Project | `curriculum` |
| Entity | unset (W&B account default; same as SmolLM — set `WANDB_ENTITY` only if needed) |
| Run name | `ARM_ID`, or `ARM_ID-SLURM_JOB_ID` when Slurm sets `SLURM_JOB_ID` |

Logged: `train/loss`, `train/lr`, throughput; task-loss eval metrics + artifacts; checkpoint artifacts on each permanent ladder save.

## Implementation order

1. `datasets/regmix/`: label + publish into `edullm-data`
2. Create `experiments/curriculum/` tree
3. `build_curriculum_index.py` + publish curriculum orders
4. `curriculum_pacing.py` + tests
5. `train_curriculum_regmix_370m.py` (ephemeral scratch + fail-closed W&B artifacts)
6. `ema_merge_checkpoints.py` + launch scripts
7. Smoke → full 17-arm matrix
