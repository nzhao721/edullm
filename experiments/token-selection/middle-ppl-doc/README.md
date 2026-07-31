# Middle PPL (document) arm

Offline RegMix filter: keep the **middle 60% of tokens** by late-RefHQ document
`avg_perplexity`, then train plain CE on the filtered corpus with upsample to the
shared one-epoch budget (**9.9B tokens / 2360 steps**).

CE stack is a near-clone of [`control/`](../control/) (RefHQ-matched OLMo-2 370M,
permanent ladder, task-loss hook). Only the data path differs.

**Train data:** published `s3://edullm-data/pretrain/middle-ppl-doc-mid60` (fail-closed
if unpublished). **Durable artifacts:**
`s3://edullm-checkpoints/token-sel/middle-ppl-doc/`.

## Ephemeral runtime

Training is meant for a machine whose scratch starts empty and is wiped after the job:

| Path | Role |
|------|------|
| `STAGE_DIR` | Job-scoped; trainer fetches shards from `edullm-data` here |
| `SAVE_FOLDER` | Job-scoped local ckpt scratch; each permanent save is uploaded to S3 before continuing |
| `PROGRESS_DIR` | Job-scoped metrics; uploaded with checkpoints and again at end |

- Do **not** assume pre-staged FarmShare/laptop corpora or leftover checkpoint trees.
- Resume only via explicit `LOAD_PATH` after staging a durable S3 checkpoint into scratch.
- Default `FRESH=1`. Local auto-resume of `SAVE_FOLDER` is refused.
- `S3_EXPORT=0` / `SKIP_S3_UPLOAD=1` are refused unless `ALLOW_LOCAL_ONLY=1` (debug only).

Does **not** read `s3://edullm-datasets/`.

## Train data gate

Default `--dataset-id` / `DATASET_ID` is **`pretrain/middle-ppl-doc-mid60`**.
Resolve uses `edullm_data.read.resolve_latest` + `dataset_paths` and exits if the
dataset is missing or unvalidated. Related published base (not a substitute):
`pretrain/regmix-10b`.

Publish the filtered corpus before training (offline helpers below → `edullm_data.publish`).

## Offline filter → publish (not the train launcher)

These helpers build the corpus that must be published into `edullm-data`. They are
not required on the train host once the dataset is published.

1. RegMix LM labels ready (`datasets/regmix/` → `READY` + `metrics_index.jsonl.gz`)
2. `filter_middle_ppl_docs.py` — middle 60% token mass by `avg_perplexity`
3. `build_filtered_corpus.py` — re-tokenize kept docs → uint32 memmaps
4. Publish as `pretrain/middle-ppl-doc-mid60` via `edullm_data.publish`

### Filter algorithm (token-weighted middle 60%)

1. Load every metrics row with finite `avg_perplexity` and positive token weight
2. Sort ascending by `(avg_perplexity, id)`
3. Let `T = sum(n_tokens)`, `lo = 0.2 T`, `hi = 0.8 T`
4. Keep a doc iff the midpoint of its token interval lies in `[lo, hi)`
5. Emit keep manifest; tokenize with `allenai/dolma2-tokenizer` (EOS 100257)

## Architecture (control-matched)

| Field | Value |
|-------|-------|
| Config | `TransformerConfig.olmo2_370M` (full attn, no SWA) |
| Size | d_model=1024, 16 layers, 16 heads, vocab 100352 |
| Seq / GBS / microbatch | 2048 / 4_194_304 / 65_536 tokens |
| Optim | SkipStepAdamW + CosWithWarmup |
| LR | peak 4e-4, warmup 24, `alpha_f=0.1` |
| Other | `z_loss_multiplier=1e-5`, `max_grad_norm=1.0`, `compile_model=True`, from scratch |
| DP | HSDP bf16; world size from `torchrun` / `WORLD_SIZE` |

Default `run_id`: **`edullm-370M-middle-ppl-doc-ladder125-v1`**

## Files

| File | Role |
|------|------|
| `filter_middle_ppl_docs.py` | Rank + keep middle 60% token mass |
| `build_filtered_corpus.py` | Materialize docs + tokenize memmaps |
| `prepare_data.py` | Offline helper for local path lists (publish pipeline) |
| `train_ce_middle_ppl_doc.py` | Control-matched CE trainer + durable S3 ladder |
| `launch_train.sh` | Ephemeral-scratch launcher (prefer this) |
| `run_train.sh` | Thin wrapper → `launch_train.sh` |
| `test_filter_middle_ppl.py` | Unit tests for filter + trainer contract |

## Launch (train host)

```bash
# Job-scoped scratch roots (may start empty).
export STAGE_DIR=/tmp/middle-ppl-doc-stage
export SAVE_FOLDER=/tmp/middle-ppl-doc-ckpts
export PROGRESS_DIR=/tmp/middle-ppl-doc-progress
export TASK_LOSS_OUT_DIR=/tmp/middle-ppl-doc-task-loss
export NPROC=1
export FRESH=1
# optional: DATASET_VERSION=v1
bash experiments/token-selection/middle-ppl-doc/launch_train.sh

# Multi-GPU
export NPROC=4
bash experiments/token-selection/middle-ppl-doc/launch_train.sh
```

Equivalent direct `torchrun`:

```bash
export PYTHONPATH=experiments/token-selection
torchrun --standalone --nproc_per_node="${NPROC:-1}" \
  experiments/token-selection/middle-ppl-doc/train_ce_middle_ppl_doc.py \
  --name edullm-370M-middle-ppl-doc-ladder125-v1 \
  --dataset-id pretrain/middle-ppl-doc-mid60 \
  --stage-dir "$STAGE_DIR" \
  --save-folder "$SAVE_FOLDER" \
  --progress-dir "$PROGRESS_DIR" \
  --length-tokens 9900000000 \
  --fresh
```

Resume (stage durable ckpt into scratch first, then):

```bash
export FRESH=0
export LOAD_PATH="$SAVE_FOLDER/step125"   # previously synced from S3
bash experiments/token-selection/middle-ppl-doc/launch_train.sh
```

## Checkpoint + eval contract

- Permanent saves: `{0, 125, …, 2125, 2360}` for a 2360-step run (**omit 2250**)
- Each permanent save uploads to `s3://edullm-checkpoints/token-sel/middle-ppl-doc/checkpoints/`
  (fail-closed unless `--allow-local-only` / `ALLOW_LOCAL_ONLY=1`)
- Progress uploaded with checkpoints and again at train end
- On every permanent save, rank 0 may spawn `task_loss_bpb` (disable with
  `--no-task-loss-on-save` or `TASK_LOSS_EVAL=0`)
