# OLMo HQ ~100B upsample pool

**S3:** `s3://edullm-dataset-olmohq/olmo-mix-1124-30b/`

**Entry:** `submit_olmo_mix_upsample.sh` → `submit_olmo_pool_tokenize.sh` (in `../olmo/`)

Reuses DCLM shards from the completed 30B run; upsamples other domains from `allenai/olmo-mix-1124`.
