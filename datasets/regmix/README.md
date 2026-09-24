# RegMix 10B

**S3:** `s3://edullm-datasets/regmix/regmix-10b/`

**Entry:** `submit_regmix_mix.sh` (source: olmohq pool on S3 or local mirror)

**AWS data prep:** `prepare_regmix_data.py`

**Difficulty labels (FarmShare):** `submit_regmix_labeling.sh` — compression ratio, Flesch reading ease, and MTLD on the seven trimmed domain shards under `trim/<domain>/`. Writes `RUN_DIR/labels/` (`READY`, `docs/`, `metrics/`, `metrics_index.jsonl.gz`).

**Label upload to S3** (upload scripts since removed):

- `labels/` → `s3://edullm-datasets/regmix/regmix-10b/labels/`
- Receipt: `RUN_DIR/labels_upload_manifest.json` (also copied to the corpus prefix)
