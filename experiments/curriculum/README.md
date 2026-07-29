# Curriculum learning experiment plan (OLMo2-370M / RegMix 10B)

## Goal

Train from scratch on the RegMix 10B corpus with **fixed token budget** and **one shared training script** parameterized by `--pacing` and `--difficulty-metric`. Only the **data sampler / batch stream** differs across arms. All arms use **warmup + constant LR** and **post-hoc EMA** over checkpoints at steps **2000, 2125, 2250, 2384** (α=0.8).

## Execution environment

**Training and task-loss evals run on AWS.** Instance type, GPU model, and GPU count are **not chosen in this plan** — scripts must work on any CUDA host with 1–N GPUs via `torchrun`.

- Data and checkpoints: S3 (`edullm-datasets`, `edullm-checkpoints`); sync or stream to local scratch on the AWS worker as needed
- No slurm/FarmShare-specific launch paths, queue names, or GPU SKU assumptions in required code paths
- `--device-batch-size` and grad accumulation are CLI parameters; derive steps from global batch `4_194_304` and discovered `world_size`
- Curriculum index build is a CPU job; runnable locally, on AWS, or any host with S3 access — not tied to training hardware

Label **generation** and **S3 upload** live in **`datasets/regmix/`** (see below); training code in `experiments/curriculum/` consumes the published S3 artifacts only.

## Training contract

Fork [`experiments/token-selection/control/train_ce_regmix_olmo_370m.py`](experiments/token-selection/control/train_ce_regmix_olmo_370m.py) and reuse shared helpers from [`experiments/token-selection/token_selection/olmo_ext/`](experiments/token-selection/token_selection/olmo_ext/) for checkpoint ladder and task-loss wiring.

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
- **Train corpus**: `s3://edullm-datasets/regmix/regmix-10b/`

Control arm reads flat `tokenized/`; curriculum arms read the `curriculum/` index.

### Checkpoint saving

Ladder via `permanent_checkpoint_steps()` from [`token_selection.olmo_ext.checkpoint_ladder`](experiments/token-selection/token_selection/olmo_ext/checkpoint_ladder.py) (manual equivalent of `checkpointer_kwargs_for_ladder()`):

- Step **0** (pre-train snapshot)
- Every **125** steps: 125, 250, …, `125 * floor(total_steps / 125)`
- Always save the **true final step** (2384)
- Omit last on-grid step when within one interval of final → `{0, 125, …, 2250, 2384}` — omit 2375
- `max_checkpoints=None` — permanent saves only; no ephemeral pruning
- Checkpoint format `model_and_optim` / `full_state_dict_v1`

### Task-loss eval

- Metric: `task_loss_bpb` = `-log2 p(gold continuation | context) / utf8_bytes(continuation)`
- Full **20-label** `*_rc_5shot_bpb` suite (ARC, BoolQ, CSQA, HellaSwag, OpenBookQA, PIQA, SocialIQA, WinoGrande, MMLU×4)
- Trigger on every permanent checkpoint save (step 0 + each ladder step + final); async/subprocess OK
- [`token_selection.olmo_ext.task_loss_hook.trigger_task_loss_eval`](experiments/token-selection/token_selection/olmo_ext/task_loss_hook.py); rank 0 only; `--task-loss-on-save` default on
- Evaluator script: [`scripts/farmshare/task_loss/eval_task_loss_olmo_core.py`](scripts/farmshare/task_loss/eval_task_loss_olmo_core.py) (repo path only; runs on the AWS training/eval worker)
- Outputs: `task_loss_results/<arm>/step{N}_task_loss.json`

Post-hoc EMA merge runs the 20-label eval on the merged artifact in addition to per-checkpoint evals.

### Hardware / launch

- World size from `torchrun` / `LOCAL_RANK` / `WORLD_SIZE` (or 1 if unset)
- No hardcoded GPU indices, `CUDA_VISIBLE_DEVICES`, instance types, or host paths as required defaults
- Global batch `4_194_304`; per-rank microbatch / grad-accum from `world_size` and `--device-batch-size`; fail-fast if not divisible
- Eval scripts: 1+ GPU compatible on the same AWS worker policy as training

### S3 export layout

```text
s3://edullm-checkpoints/curriculum/<arm_id>/
  checkpoints/
  task_loss_results/
  metrics/
  progress/
```

Use `arm_s3_prefix(arm_id)` in the trainer (`curriculum/<arm_id>`); same layout as `token_selection.olmo_ext.s3_layout` patterns.

## Local repo layout (`C:\alpha_ai\edullm`)

### `datasets/regmix/` — labeling and S3 upload

All document labeling and upload to `s3://edullm-datasets/regmix/regmix-10b/` stay in the regmix dataset package:

```
datasets/regmix/
├── submit_regmix_labeling.sh          # heuristic metrics (existing)
├── submit_regmix_doc_lm_labeling.sh   # learnability LM labels (existing)
├── label_regmix_shard.sbatch          # ...
├── label_regmix_doc_lm.py             # ...
├── finalize_regmix_lm_labels.py       # ...
├── finalize_regmix_upload.py          # mix corpus upload (existing pattern)
├── finalize_regmix_labels_upload.py   # NEW: labels/ + lm_labels/ → S3
└── submit_regmix_labels_upload.sh     # NEW: driver for label upload
```

`finalize_regmix_labels_upload.py` mirrors [`finalize_regmix_upload.py`](datasets/regmix/finalize_regmix_upload.py):

- Inputs: `--run-dir`, `--dst-bucket edullm-datasets`, `--dst-prefix regmix/regmix-10b`
- Upload `labels/` and `lm_labels/` trees; write `labels_upload_manifest.json`
- Update [`S3_DATASETS.md`](S3_DATASETS.md) and [`datasets/regmix/README.md`](datasets/regmix/README.md)

### `experiments/curriculum/` — training experiment code

```
experiments/curriculum/
├── train_curriculum_regmix_370m.py
├── curriculum_pacing.py
├── ema_merge_checkpoints.py
├── scripts/
│   └── build_curriculum_index.py    # reads S3 labels → curriculum/ index
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
- [`experiments/token-selection/token_selection/olmo_ext/`](experiments/token-selection/token_selection/olmo_ext/) — checkpoint ladder, task_loss, S3 layout
- [`datasets/regmix/`](datasets/regmix/) — labeling + label upload (no duplicate upload scripts under `experiments/curriculum/`)

**S3 `regmix-10b/curriculum/`** is the published training index (built by `build_curriculum_index.py`), not the source tree.

## Data on S3

- **`tokenized/<domain>/`**: On S3 — produced by `datasets/regmix/` mix pipeline
- **`labels/`**: To upload — produced by `datasets/regmix/` heuristic labeling + `finalize_regmix_labels_upload.py`
- **`lm_labels/`**: To upload — produced by `datasets/regmix/` LM labeling + `finalize_regmix_labels_upload.py`
- **`curriculum/`**: To build — produced by `experiments/curriculum/scripts/build_curriculum_index.py`

**17 arms:** 1 control + 4 pacing × 4 metrics (`compression_ratio`, `flesch`, `mtld`, `learnability`).

**Pacing:** `linear_n10`, `expanding_25_1000`, `warmup_1000`, `interleave_i10_linear` — 250-step segments aligned to checkpoint ladder (last segment 134 steps).

## Phases

### Phase 0: Label upload (`datasets/regmix/`)

- Add `datasets/regmix/finalize_regmix_labels_upload.py` + `submit_regmix_labels_upload.sh`
- Upload completed FarmShare label runs: `labels/` and `lm_labels/` → S3
- Update [`S3_DATASETS.md`](S3_DATASETS.md) and [`datasets/regmix/README.md`](datasets/regmix/README.md)

### Phase 1: Build curriculum index (`experiments/curriculum/`)

- `experiments/curriculum/scripts/build_curriculum_index.py` — merge labels from S3, per-metric ranks, doc/chunk index, tokenize docs
- CPU job; upload to `s3://edullm-datasets/regmix/regmix-10b/curriculum/`

### Phase 2: Pacing library

- `experiments/curriculum/curriculum_pacing.py` + `tests/test_pacing.py`

### Phase 3: Shared trainer

- `experiments/curriculum/train_curriculum_regmix_370m.py` — fork from control trainer; `--lr-alpha-f 1.0`
- Checkpoint ladder via `permanent_checkpoint_steps()`; S3 prefix via `arm_s3_prefix()`; task-loss via `trigger_task_loss_eval`
- `experiments/curriculum/tests/test_training_defaults.py` — ladder `{0,125,…,2250,2384}` omits 2375; GBS/microbatch defaults

### Phase 4: Post-hoc EMA

- `experiments/curriculum/ema_merge_checkpoints.py` — merge steps 2000/2125/2250/2384, α=0.8
- Checkpoints: `s3://edullm-checkpoints/curriculum/<arm_id>/`

### Phase 5: Launch matrix

- `experiments/curriculum/launch/launch_arm.sh` — `python` (1 rank) or `torchrun` (N ranks); all paths via CLI/env
- `experiments/curriculum/launch/submit_matrix.sh` — thin wrapper for AWS job submission (service chosen at launch time)
- Arm matrix: see [Arm matrix (17 arms)](#arm-matrix-17-arms) below

### Phase 6: Smoke test → full runs

- Short smoke on AWS (single rank) for control + one curriculum arm
- Full 17-arm matrix on AWS after labels and curriculum index are on S3

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

Example launches (paths are local scratch; S3 layout is `curriculum/<arm_id>/checkpoints` and `…/progress`):

```bash
# Control
ARM_ID=control PACING=control TRAIN_PATHS_FILE=/data/regmix/paths_train.txt \
  SAVE_FOLDER=/scratch/curriculum/control/checkpoints \
  PROGRESS_DIR=/scratch/curriculum/control/progress \
  bash experiments/curriculum/launch/launch_arm.sh

# Curriculum example (linear + compression_ratio)
ARM_ID=linear10-cr PACING=linear_n10 DIFFICULTY_METRIC=compression_ratio \
  CURRICULUM_INDEX=/data/regmix/curriculum \
  SAVE_FOLDER=/scratch/curriculum/linear10-cr/checkpoints \
  PROGRESS_DIR=/scratch/curriculum/linear10-cr/progress \
  bash experiments/curriculum/launch/launch_arm.sh

# Optional post-hoc EMA (task-loss default on)
python experiments/curriculum/ema_merge_checkpoints.py \
  --checkpoints-root /scratch/curriculum/linear10-cr/checkpoints \
  --arm-id linear10-cr
```

## Implementation order

1. `datasets/regmix/`: label upload script + submit driver + docs
2. Create `experiments/curriculum/` tree
3. `build_curriculum_index.py` + S3 upload
4. `curriculum_pacing.py` + tests
5. `train_curriculum_regmix_370m.py`
6. `ema_merge_checkpoints.py` + launch scripts
7. AWS smoke → full 17-arm matrix
