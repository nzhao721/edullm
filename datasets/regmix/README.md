# RegMix 10B

**S3:** `s3://edullm-datasets/regmix/regmix-10b/`

**Entry:** `submit_regmix_mix.sh` (source: olmohq pool on S3 or local mirror)

**AWS data prep:** `prepare_regmix_data.py`

**Difficulty labels (FarmShare):** `submit_regmix_labeling.sh` — compression ratio, Flesch reading ease, and MTLD on the seven trimmed domain shards under `trim/<domain>/`. Writes `RUN_DIR/labels/` (`READY`, `docs/`, `metrics/`, `metrics_index.jsonl.gz`).

**LM learnability labels (FarmShare):** `submit_regmix_doc_lm_labeling.sh` — document-level NLL / learnability under averaged RefHQ checkpoints. Writes `RUN_DIR/lm_labels/labels/` (nested; finalized with `READY`).

**Label upload to S3:** `submit_regmix_labels_upload.sh` → `finalize_regmix_labels_upload.py`

- `labels/` → `s3://edullm-datasets/regmix/regmix-10b/labels/`
- `lm_labels/labels/` (or flat `lm_labels/`) → `s3://edullm-datasets/regmix/regmix-10b/lm_labels/`
- Receipt: `RUN_DIR/labels_upload_manifest.json` (also copied to the corpus prefix)

**Curriculum training index:** built by `experiments/curriculum/scripts/build_curriculum_index.py`; published as `curriculum/regmix-370m` on edullm-data (four order groups).

```bash
# 1. Build local ranked index (CPU; needs labels + lm_labels on disk or S3)
python experiments/curriculum/scripts/build_curriculum_index.py \
  --labels-root ... --lm-labels-root ... --out-dir ...

# 2. Stage + publish token-order groups (requires pretrain/regmix-10b on edullm-data)
python datasets/regmix/publish_regmix_curriculum_edullm_data.py \
  --index-dir ... --stage-dir ...          # live publish
python datasets/regmix/publish_regmix_curriculum_edullm_data.py \
  --index-dir ... --stage-dir ... --dry-run  # layout check only
```

**FarmShare publish:** `sync_submit_publish_regmix_curriculum_edullm_data.sh` (from laptop) or `submit_publish_regmix_curriculum_edullm_data.sh` (on login node). Defaults: `INDEX_DIR=$REGMIX_ROOT/curriculum_index`, target `curriculum/regmix-370m`. Set `DRY_RUN=1` for layout-only; `PARENT_VERSION=` to pin `pretrain/regmix-10b`.
