# Learnability (document-level)

Offline corpus filter + plain CE on RegMix docs with the **largest early→late RefHQ improvement**, keeping the top **60% of tokens** (token-weighted), then upsampling to the shared one-epoch budget (**9.9B / 2360 steps**).

**Train data:** published+validated `pretrain/learnability-doc-top60` on `s3://edullm-data/` (fail-closed if missing — never falls back to unfiltered `regmix-10b` or legacy `edullm-datasets`).

**Artifact durability:** runtime scratch + W&B after each permanent save and at end of run; production online checkpoint uploads fail closed.

Near-clone of **control** CE stack (same arch, ladder, eval hook); independent variable is the offline doc filter. Differs from **learnability-token** (online dual-ref scorer) by filtering documents offline only.

## Ephemeral empty-scratch

Jobs must run on a clean scratch that starts empty and may be wiped after the job:

1. Stage train shards from `s3://edullm-data/` into `STAGE_DIR` (trainer does this via `edullm_data.read`).
2. Write checkpoints/progress under job-local `SAVE_FOLDER` / `PROGRESS_DIR`.
3. Upload those artifacts to W&B before the job ends.
4. Resume with `WANDB_RESUME_ARTIFACT` or a local `LOAD_PATH`; `SAVE_FOLDER` is not auto-resumed.

## Label dependency (publish path)

Building/publishing the filtered corpus **requires** finalized RegMix document LM labels from `datasets/regmix/`:

1. `submit_regmix_doc_lm_labeling.sh` → per-chunk metrics
2. `finalize_regmix_lm_labels.py` → `metrics_index.jsonl.gz` + `READY`

Typical layout:

```text
$LABELS_ROOT/
  metrics_index.jsonl.gz
  READY
  docs/<domain>/*.jsonl.gz
  SCHEMA.json
```

Filter fails clearly until `READY` + `metrics_index.jsonl.gz` exist. Pass `--allow-incomplete` / `ALLOW_INCOMPLETE=1` only for debugging partial indexes. Then publish via `edullm_data.publish` as `pretrain/learnability-doc-top60` before training.

## Filter polarity

| Field | Meaning |
|-------|---------|
| Stored metric | `learnability_late_minus_early_avg_nll` = late − early |
| Improvement | `early − late` = **−**`learnability_late_minus_early_avg_nll` |
| Keep | **Top 60% of tokens** by largest improvement |

Negative stored values = late model improved (lower NLL). Ranking uses the negated metric so the most-improved docs are kept first; selection is cumulative over `n_loss_tokens` (fallback `n_tokens`).

## Build + publish filtered corpus

```bash
export LABELS_ROOT=/path/to/lm_labels/labels
export WORK=/path/to/learnability-doc-work   # job-local only; not a train data source of truth
# optional: KEEP_FRACTION=0.6  ALLOW_INCOMPLETE=1
bash experiments/token-selection/learnability-doc/prepare_data.sh
# then publish $WORK/corpus → pretrain/learnability-doc-top60 via edullm_data.publish
```

Or step-by-step:

```bash
python experiments/token-selection/learnability-doc/filter_learnability_docs.py \
  --labels-root "$LABELS_ROOT" \
  --out-dir "$WORK/filter" \
  --keep-token-fraction 0.6

python experiments/token-selection/learnability-doc/build_filtered_corpus.py \
  --labels-root "$LABELS_ROOT" \
  --filter-dir "$WORK/filter" \
  --out-dir "$WORK/corpus"
```

Outputs: `$WORK/corpus/paths_train.txt`, per-domain `tokenized/<domain>/<domain>.npy`, manifests — for **publishing**, not as a persistent train assumption.

## Train (CE, RefHQ-matched olmo2_370M)

Permanent ladder: `{0, 125, …, 2125, 2360}` (omit 2250). Immediate 20-label `task_loss_bpb` via shared `token_selection.olmo_ext.task_loss_hook` on each save.

**Clean / ephemeral machine** (resolve+stage from `edullm-data`):

```bash
export STAGE_DIR=/tmp/learnability-doc-stage      # empty scratch OK
export SAVE_FOLDER=/tmp/ckpts/learnability-doc    # job-local; uploaded to W&B
export PROGRESS_DIR=/tmp/progress/learnability-doc
export NPROC=1
# FRESH defaults on; durable export defaults on
bash experiments/token-selection/learnability-doc/launch_train.sh
```

**Multi-GPU** (world size from `NPROC` / `torchrun`; no hardcoded device list):

```bash
export NPROC=4
bash experiments/token-selection/learnability-doc/launch_train.sh
```

Equivalent direct call:

```bash
export PYTHONPATH=experiments/token-selection
torchrun --standalone --nproc_per_node="$NPROC" \
  experiments/token-selection/learnability-doc/train_ce_learnability_doc_olmo_370m.py \
  --name edullm-370M-learnability-doc-10b \
  --dataset-id pretrain/learnability-doc-top60 \
  --stage-dir "$STAGE_DIR" \
  --save-folder "$SAVE_FOLDER" \
  --progress-dir "$PROGRESS_DIR" \
  --length-tokens 9900000000 \
  --fresh
```

Upsample: `--length-tokens 9900000000` (2360 steps at GBS `4_194_304`); the dataloader cycles the kept ~6B tokens.

Disable eval enqueue with `TASK_LOSS_EVAL=0` / `--no-task-loss-on-save`. Resume with `WANDB_RESUME_ARTIFACT=entity/project/name:alias`.

## Files

| File | Role |
|------|------|
| `filter_learnability_docs.py` | Token-weighted top-60% selection from `metrics_index` |
| `build_filtered_corpus.py` | Re-tokenize kept docs → uint32 memmaps (publish input) |
| `prepare_data.sh` | Filter + build orchestration |
| `train_ce_learnability_doc_olmo_370m.py` | CE trainer; permanent checkpoint → synchronous 20-label eval → strict W&B upload → durable marker |
| `launch_train.sh` | Hardware-agnostic launcher (STAGE_DIR required) |
| `enqueue_task_loss.sh` | Optional FarmShare eval wrapper (trainer uses shared hook) |
| `test_filter_polarity.py` | Polarity + ladder unit tests (no GPU) |

Does **not** launch AWS; does not modify other arms.
