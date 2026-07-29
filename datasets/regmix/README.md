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

**Curriculum training index:** built by `experiments/curriculum/scripts/build_curriculum_index.py` → `s3://edullm-datasets/regmix/regmix-10b/curriculum/` (doc/chunk ranks + tokenized memmaps for curriculum pacing arms).
