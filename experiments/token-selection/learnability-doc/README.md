# Learnability (document-level)

Offline corpus filter + plain CE on RegMix docs with the **largest early→late RefHQ improvement**, keeping the top **60% of tokens** (token-weighted), then upsampling to a **10B / ~2384-step** budget.

S3 export: `s3://edullm-checkpoints/token-sel/learnability-doc/`.

Near-clone of **control** CE stack (same arch, ladder, eval hook); independent variable is the offline doc filter. Differs from **learnability-token** (online dual-ref scorer) by filtering documents offline only.

## Label dependency

This arm **requires** finalized RegMix document LM labels from `datasets/regmix/`:

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

Filter fails clearly until `READY` + `metrics_index.jsonl.gz` exist. Pass `--allow-incomplete` / `ALLOW_INCOMPLETE=1` only for debugging partial indexes. Corpus build is plan-allowed pending until labels are READY.

## Filter polarity

| Field | Meaning |
|-------|---------|
| Stored metric | `learnability_late_minus_early_avg_nll` = late − early |
| Improvement | `early − late` = **−**`learnability_late_minus_early_avg_nll` |
| Keep | **Top 60% of tokens** by largest improvement |

Negative stored values = late model improved (lower NLL). Ranking uses the negated metric so the most-improved docs are kept first; selection is cumulative over `n_loss_tokens` (fallback `n_tokens`).

## Build filtered corpus

```bash
export LABELS_ROOT=/path/to/lm_labels/labels
export WORK=/path/to/learnability-doc-work
# optional: KEEP_FRACTION=0.6  ALLOW_INCOMPLETE=1
bash experiments/token-selection/learnability-doc/prepare_data.sh
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

Outputs: `$WORK/corpus/paths_train.txt`, per-domain `tokenized/<domain>/<domain>.npy`, manifests.

## Train (CE, RefHQ-matched olmo2_370M)

Permanent ladder: `{0, 125, …, 2250, 2384}` (omit 2375). Immediate 20-label `task_loss_bpb` via shared `token_selection.olmo_ext.task_loss_hook` on each save.

**Single GPU:**

```bash
export TRAIN_PATHS_FILE=$WORK/corpus/paths_train.txt
export SAVE_FOLDER=/path/to/ckpts/learnability-doc
export PROGRESS_DIR=/path/to/progress/learnability-doc
export NPROC=1
export FRESH=1
bash experiments/token-selection/learnability-doc/launch_train.sh
```

**Multi-GPU** (world size from `NPROC` / `torchrun`; no hardcoded device list):

```bash
export NPROC=4   # or whatever is available
bash experiments/token-selection/learnability-doc/launch_train.sh
```

Equivalent direct call:

```bash
export PYTHONPATH=experiments/token-selection
torchrun --standalone --nproc_per_node="$NPROC" \
  experiments/token-selection/learnability-doc/train_ce_learnability_doc_olmo_370m.py \
  --name edullm-370M-learnability-doc-10b \
  --train-paths-file "$TRAIN_PATHS_FILE" \
  --save-folder "$SAVE_FOLDER" \
  --progress-dir "$PROGRESS_DIR" \
  --length-tokens 10000000000 \
  --fresh
```

Upsample: `--length-tokens 10000000000` (~2384 steps at GBS `4_194_304`); the dataloader cycles the kept ~6B tokens.

Disable eval enqueue with `TASK_LOSS_EVAL=0` / `--no-task-loss-on-save`.

## Files

| File | Role |
|------|------|
| `filter_learnability_docs.py` | Token-weighted top-60% selection from `metrics_index` |
| `build_filtered_corpus.py` | Re-tokenize kept docs → uint32 memmaps |
| `prepare_data.sh` | Filter + build orchestration |
| `train_ce_learnability_doc_olmo_370m.py` | CE trainer + permanent ladder + shared task_loss hook |
| `launch_train.sh` | Hardware-agnostic launcher |
| `enqueue_task_loss.sh` | Optional FarmShare eval wrapper (trainer uses shared hook) |
| `test_filter_polarity.py` | Polarity + ladder unit tests (no GPU) |

Does **not** launch AWS; does not modify other arms.
