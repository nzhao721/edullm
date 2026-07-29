# Dataset pipelines

Build scripts for corpora published on S3 (see [`S3_DATASETS.md`](../S3_DATASETS.md); bucket consolidation as of **2026-07-27** in [S3 consolidation](../S3_DATASETS.md#s3-consolidation-2026-07-27)).

| Directory | S3 bucket / prefix | Description |
|-----------|-------------------|-------------|
| [`olmo/`](olmo/) | `edullm-datasets/olmo30b/olmo-mix-1124-30b/` | ~30B trimmed sample from `allenai/olmo-mix-1124` |
| [`olmohq/`](olmohq/) | `edullm-datasets/olmo100b/olmo-mix-1124-30b/` | ~100B upsampled / rebalanced pool (feeds RegMix) |
| [`regmix/`](regmix/) | `edullm-datasets/regmix/regmix-10b/` | 10B RegMix-weighted mix from olmohq |
| [`refhq/`](refhq/) | `edullm-datasets/refhq/refhq-regmix-5p5b-v1/` | 5.5B HQ-filtered reference corpus (HF sources) |

## Shared dataset utilities (this directory)

Scripts used by more than one corpus pipeline live here (not under `olmo/`, `regmix/`, etc.):

| File | Purpose |
|------|---------|
| `olmo_shard_utils.py` | OLMo-mix shard I/O, domain token totals, doc materialization |
| `download_s3_shard.py` / `.sbatch` | Slurm array worker to fetch one manifest-indexed shard |
| `trim_olmo_overshoot.py` | Trim one overshot OLMo domain to a token budget |
| `trim_and_tokenize_regmix.py` | Trim one domain and emit dolma2 uint32 `.npy` memmaps |

FarmShare **platform** utilities (AWS session minting, bootstrap) remain in [`scripts/farmshare/`](../scripts/farmshare/).
