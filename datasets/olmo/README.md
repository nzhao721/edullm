# OLMo-mix ~30B

**S3:** `s3://edullm-dataset-olmo/olmo-mix-1124-30b/`

**Entry:** `submit_olmo_mix_sample.sh` → manual trim (`trim_olmo_overshoot.sbatch`) → `finalize_olmo_trim_upload.py` → `submit_olmo_pool_tokenize.sh`

**HF source:** `allenai/olmo-mix-1124` (seed 42, stratified sample)
