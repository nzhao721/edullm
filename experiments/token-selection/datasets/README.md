# Dataset build code

This is the code that built the three corpora Appendix A of the paper describes:
the 10B-token training corpus, the 5.5B-token HQ reference, and the ~3.9B-token
Instruct reference. It is a **copy**, vendored here so a reader following this
repo's "Code and data availability" link can find the pipelines that produced
the data, not just the prose description of them in the appendix tables.

## Where it came from

All of it is copied unchanged from the top level of this same repo, under
`datasets/`, as of 2026-09-19:

| Here | Source | Produces |
| --- | --- | --- |
| `olmo/` | `datasets/olmo/` | `olmo-mix-1124-30b` — a ~30B-token stratified sample of `allenai/olmo-mix-1124` (Hugging Face) |
| `olmohq/` | `datasets/olmohq/` | `olmo100b/olmo-mix-1124-30b` — an upsampled ~100B+ pool built on top of the 30B sample, sized to cover RegMix's per-domain targets |
| `regmix/` | `datasets/regmix/` | `pretrain/regmix-10b` **v1** — the paper's 10B-token training corpus, domain-weighted from the pool above |
| `refhq/` | `datasets/refhq/` | `refhq/refhq-regmix-5p5b-v1` — the paper's 5.5B-token HQ reference corpus (DCLM, StarCoder, peS2o, arXiv, OpenWebMath, Algebraic Stack, Wikipedia) |
| `refhq_new/` | `datasets/refhq_new/` | `pretrain/refhq-instruct` — the paper's ~3.9B-token Instruct reference corpus (Tulu-v2, OpenHermes-2.5, Tulu-3, Hermes-3, SmolTalk, Dolci) |
| top-level `*.py`/`*.sbatch` here | `datasets/*.py`/`*.sbatch` | Shared utilities the four pipelines above import or invoke |

The four AWS-session-minting scripts once vendored under `farmshare/` have been removed;
the deployment scripts here still `source` them from `datasets/farmshare/`.

`olmo/` and `olmohq/` are here because `regmix/` depends on them: the 10B
training corpus is a domain-weighted subsample of the `olmohq` pool, which is
itself built on top of the `olmo` 30B sample. Reproducing `pretrain/regmix-10b`
from Hugging Face means walking this whole chain.

## What is included, and why

Each pipeline is a real Slurm/FarmShare job graph with a plan → download/build
→ tokenize → finalize → publish shape. What's here is that graph's real path:
the entry scripts each pipeline's own README (or, for `refhq_new/`,
`DATASET-DESIGN.md`) names, plus the Python modules those entry scripts
directly invoke, plus the filter-spec modules that Appendix A's tables
describe in prose (`domain_configs.py`, `hq_reference_sources.py`,
`refhq_new_sources.py`, `exclusion.py` + `exclusion_rules.yaml`, `process.py`,
`math_quality.py`, `sample_datadecide_dclm.py`, and the `configs/*.yaml`
tagger/mixer configs).

## What is deliberately excluded

- **Labeling infrastructure under `regmix/`.** The source directory's
  `regmix/` holds a second job graph — difficulty labels (compression
  ratio, Flesch, MTLD) — for a different paper (the
  curriculum-difficulty project). None of it touches `pretrain/regmix-10b` itself; it only reads
  the finished corpus. Excluded: `build_regmix_label_manifest.py`,
  everything under `*_regmix_labels*`, and
  every FarmShare `check_*`/`compare_*`/`diagnose_*`/`status_*` operational
  script for that job graph.
- **The same shape under `olmo/`.** `build_label_manifest.py`,
  `finalize_olmo_labels.py`, `label_olmo_shard.py`/`.sbatch`,
  `materialize_curriculum.py`, `submit_olmo_labeling.sh`,
  `text_difficulty_metrics.py`, `test_text_difficulty_metrics.py`, and
  `estimate_olmo_domain_tokens.py` — curriculum labeling and a one-off token
  estimate, not corpus construction.
- **Smoke tests and local dry-runs**: `smoke_code_copyright_strip.py`
  (`refhq/`), `smoke_refhq_new_local.py` (`refhq_new/`). Useful for
  developing the filters; produced no reported artifact.
- **`regmix/finalize_regmix_upload.py`.** An alternate uploader that
  provisions a raw S3 bucket and copies shards directly, bypassing the
  `edullm_data.publish()` call. The regmix `README.md`'s documented chain
  ends in `publish_regmix_edullm_data.py`, which does call `publish()`; this
  script isn't part of that chain and appears to be an earlier or
  alternate path. Excluded as superseded.
- **`olmo/tokenize_olmo_shard_aws.py`.** An AWS-specific tokenize variant;
  the pipeline that actually ran used the FarmShare `tokenize_olmo_shard.py`
  in the same directory.
- **Everything else in `refhq/` and `refhq_new/`** not named above: mainly
  `tests/`, which check the filter logic but produced no reported artifact.

## Caveat

This code is a copy of what's on `main` in this repo today, not a snapshot
pinned to when each corpus was actually built (2026-07 for `olmo`/`olmohq`/
`regmix`, 2026-07–08 for `refhq`, 2026-08 for `refhq_new`, per each file's
last-modified date in the source tree). If you need the exact revision, check
this repo's git history for the commit dates in that range.
