# Middle PPL (document) arm

Offline RegMix filter: keep the **middle 60% of tokens** by late-RefHQ document
`avg_perplexity`, then train plain CE on the filtered corpus with upsample to a
**10B-token / ~2384-step** budget.

CE stack is a near-clone of [`control/`](../control/) (RefHQ-matched OLMo-2 370M,
permanent ladder, task-loss hook). Only the data path differs.

S3 export: `s3://edullm-checkpoints/token-sel/middle-ppl-doc/`.

## Dependency (do not train until labels are ready)

This arm **requires finalized RegMix LM labels**:

1. `datasets/regmix/submit_regmix_doc_lm_labeling.sh` (or equivalent) produces per-shard metrics
2. `datasets/regmix/finalize_regmix_lm_labels.py` writes:
   - `lm_labels/metrics_index.jsonl.gz`
   - `lm_labels/READY`
   - `lm_labels/docs/<domain>/*.jsonl.gz` (text + metrics)

`avg_perplexity` = `exp(avg_nll)` under the **late** RefHQ average
(steps 1000 / 1125 / 1315), as defined in `label_regmix_doc_lm.py`.

Scripts are shipped now and will run once that tree exists. Do **not** wait on
labels to land code; do **not** launch AWS from these helpers.

## Filter algorithm (token-weighted middle 60%)

1. Load every metrics row with finite `avg_perplexity` and positive token weight
   (`n_loss_tokens`, else `n_tokens`)
2. Sort ascending by `(avg_perplexity, id)` — easiest → hardest
3. Let `T = sum(n_tokens)`, `lo = 0.2 T`, `hi = 0.8 T`
4. Walk cumulative token mass; **keep** a doc iff the midpoint of its token
   interval lies in `[lo, hi)`  
   → drops easiest ~20% and hardest ~20% of tokens; documents are atomic
5. Emit `keep_manifest.jsonl.gz` + `keep_ids.txt`

Kept docs are re-tokenized with `allenai/dolma2-tokenizer` (EOS 100257) into
per-domain `uint32` memmaps. Training cycles the kept stream to hit 10B tokens.

## Architecture (control-matched)

| Field | Value |
|-------|-------|
| Config | `TransformerConfig.olmo2_370M` (full attn, no SWA) |
| Size | d_model=1024, 16 layers, 16 heads, vocab 100352 |
| Seq / GBS / microbatch | 2048 / 4_194_304 / 65_536 tokens |
| Optim | SkipStepAdamW + CosWithWarmup |
| LR | peak 4e-4, warmup 24, `alpha_f=0.1` |
| Other | `z_loss_multiplier=1e-5`, `compile_model=True`, from scratch |
| DP | HSDP bf16; world size from `torchrun` / `WORLD_SIZE` |

Default `run_id`: **`edullm-370M-middle-ppl-doc-ladder125-v1`**

## Files

| File | Role |
|------|------|
| `filter_middle_ppl_docs.py` | Rank + keep middle 60% token mass |
| `build_filtered_corpus.py` | Materialize docs + tokenize memmaps |
| `prepare_data.py` | Write `paths_train.txt` + `length_tokens.txt` |
| `train_ce_middle_ppl_doc.py` | Control-matched CE trainer + permanent ladder |
| `launch_train.sh` | 1..N GPU launcher (prefer this) |
| `run_train.sh` | Thin wrapper → `launch_train.sh` |
| `test_filter_middle_ppl.py` | Unit tests for filter + ladder contract |

Immediate eval uses shared `token_selection.olmo_ext.task_loss_hook` (default
`task_loss_results/middle-ppl-doc/`; override with `--task-loss-results-dir` /
`TASK_LOSS_OUT_DIR`).

## Launch commands

```bash
# Paths are examples — use your local labels / scratch roots.
LABELS_ROOT=/path/to/regmix/lm_labels          # must contain READY + metrics_index
FILTER_OUT=/path/to/middle_ppl_doc_filter
CORPUS_OUT=/path/to/middle_ppl_doc_corpus
WORK=/path/to/middle_ppl_doc_work
CKPT=/path/to/ckpts/edullm-370M-middle-ppl-doc-ladder125-v1
PROGRESS=/path/to/progress/middle-ppl-doc
EVAL_OUT=/path/to/task_loss_results/middle-ppl-doc

# 1) Filter (requires READY unless --allow-incomplete)
python experiments/token-selection/middle-ppl-doc/filter_middle_ppl_docs.py \
  --labels-root "$LABELS_ROOT" \
  --out-dir "$FILTER_OUT"

# 2) Build filtered tokenized corpus
python experiments/token-selection/middle-ppl-doc/build_filtered_corpus.py \
  --labels-root "$LABELS_ROOT" \
  --keep-manifest "$FILTER_OUT/keep_manifest.jsonl.gz" \
  --out-dir "$CORPUS_OUT"

# 3) Prepare train paths (10B upsample budget)
python experiments/token-selection/middle-ppl-doc/prepare_data.py \
  --work "$WORK" \
  --train-tokenized-root "$CORPUS_OUT/tokenized" \
  --length-tokens 10000000000

# 4) Train — single GPU
export PYTHONPATH=experiments/token-selection
export TRAIN_PATHS_FILE="$WORK/train_tokenized/paths_train.txt"
export SAVE_FOLDER="$CKPT"
export PROGRESS_DIR="$PROGRESS"
export TASK_LOSS_OUT_DIR="$EVAL_OUT"
export NPROC=1
bash experiments/token-selection/middle-ppl-doc/launch_train.sh

# Multi-GPU (world size = NPROC; no hardcoded device IDs)
export NPROC=4
bash experiments/token-selection/middle-ppl-doc/launch_train.sh
```

Equivalent direct `torchrun`:

```bash
export PYTHONPATH=experiments/token-selection
torchrun --standalone --nproc_per_node="${NPROC:-1}" \
  experiments/token-selection/middle-ppl-doc/train_ce_middle_ppl_doc.py \
  --name edullm-370M-middle-ppl-doc-ladder125-v1 \
  --train-paths-file "$WORK/train_tokenized/paths_train.txt" \
  --save-folder "$CKPT" \
  --progress-dir "$PROGRESS" \
  --length-tokens 10000000000 \
  --fresh
```

## Checkpoint + eval contract

- Permanent saves: `{0, 125, …, 2250, 2384}` for a 2384-step run (**omit 2375**)
- No ephemeral pruning (`max_checkpoints=None`)
- On every permanent save, rank 0 spawns `task_loss_bpb` (disable with
  `--no-task-loss-on-save` or `TASK_LOSS_EVAL=0`)
- Checkpoints + results export under `token-sel/middle-ppl-doc/` (disable with
  `S3_EXPORT=0` / `SKIP_S3_UPLOAD=1`)
