# RegMix 10B

**S3:** `s3://edullm-datasets/regmix/regmix-10b/`

**Entry:** `submit_regmix_mix.sh` (source: olmohq pool on S3 or local mirror)

**AWS data prep:** `prepare_regmix_data.py`

**Publish:** `publish_regmix_edullm_data.py` (called from `submit_publish_regmix_edullm_data.sh`) stages the trimmed, tokenized shards and calls `edullm_data.publish()` under `pretrain/regmix-10b`.

The source repo's `regmix/` also holds a second, unrelated job graph — per-document difficulty labels (compression ratio, Flesch, MTLD, LM learnability) — for a different paper. That graph reads the finished `pretrain/regmix-10b` corpus but never writes it, so it isn't part of this pipeline; see `../README.md` for why it's excluded here.
