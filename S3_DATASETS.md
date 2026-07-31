# eduLLM S3 Datasets Inventory

Read-only inventory in **sbsandbox** (`056956104102`), scan **2026-07-31 ~12:15 UTC** via sb-aws.

| Bucket | Role |
|--------|------|
| **`s3://edullm-data/`** | Canonical published datasets (validator-promoted). Prefer this for training when a matching `dataset_id` exists. |
| **`s3://edullm-datasets/`** | Working / pre-standard corpora (mixes, labels, archives, unpromoted pools). |
| **`s3://edullm-landing/`** | Staging only (14-day expiry). Not for training. |

Checkpoints: [`S3_CHECKPOINTS.md`](S3_CHECKPOINTS.md). Spec: `docs/dataset-creation/DATASET-STANDARD.md`.

---

## `s3://edullm-data/` — published

Catalog: `_catalog/<family>/<name>/<version>.json`. Layout: `s3://edullm-data/<dataset_id>/<version>/` with `dataset.json`, `README.md`, and `tokens/` or `tokenizer/` groups (`.u32le.bin` for `pretrain-tokens/v1`).

| Dataset ID | Version | Tokens (train / val) | Tokenizer | Created (UTC) | Bytes / objects |
|------------|---------|---------------------:|-----------|---------------|----------------:|
| `pretrain/olmo-150b-dolma2` | v1 | 157.237B / 0.230B | `tokenizer/dolma2-bpe` v1 | 2026-07-30T16:29:30Z | 629.9 GiB / 6911 |
| `pretrain/olmo-original-30b` | v1 | 31.287B / 0.047B | `tokenizer/dolma2-bpe` v1 | 2026-07-31T02:07:09Z | 125.3 GiB / 127 |
| `pretrain/regmix-10b` | v1 | 9.990B / 0.015B | `tokenizer/dolma2-bpe` v1 | 2026-07-30T22:49:03Z | 40.0 GiB / 48 |
| `pretrain/refhq-regmix-5p5b` | v2 | 5.509B / 0.008B | `tokenizer/dolma2-bpe` v1 | 2026-07-30T21:20:09Z | 22.1 GiB / 31 |
| `pretrain/math-memory-full` | v1 | 21.500B / 0.00012B (Lean val) | `tokenizer/bytes-utf8` v1 | 2026-07-31T01:05:45Z | 86.0 GiB / 47 |
| `pretrain/lean4-mathlib-bytes` | v3 | 0.081B / 0.00012B | `tokenizer/bytes-utf8` v1 | 2026-07-30T23:29:30Z | 0.33 GiB / 2 |
| `tokenizer/dolma2-bpe` | v1 | — | (self) | 2026-07-29T03:45:00Z | 6.8 MiB / 5 |
| `tokenizer/bytes-utf8` | v1 | — | (self) | 2026-07-30T22:44:59Z | 5.2 KiB / 4 |

### Details

- **`pretrain/olmo-150b-dolma2` v1** — Dolma2-0625 150B sample. Nested `tokens/<source>/<domain>/`. Val carved per source. Four degenerate shards excluded (~1.5M tokens).
- **`pretrain/olmo-original-30b` v1** — Same domain weights as `allenai/olmo-mix-1124`; nested `tokens/<source>/`. Val = 0.15% per source. Related working copy: `edullm-datasets/olmo30b/`.
- **`pretrain/regmix-10b` v1** — RegMix-weighted 10B; nested `tokens/<source>/`. Related working copy: `edullm-datasets/regmix/regmix-10b/`.
- **`pretrain/refhq-regmix-5p5b` v2** — HQ-filtered RegMix 5.5B; nested `tokens/<source>/`. Related working copy: `edullm-datasets/refhq/refhq-regmix-5p5b-v1/`.
- **`pretrain/math-memory-full` v1** — P3Math byte tokens (OpenWebMath + AlgebraicStack + arXiv math-pure + Lean4-Mathlib). Val is Lean-only.
- **`pretrain/lean4-mathlib-bytes` v3** — Lean4 Mathlib as UTF-8 byte tokens; single train shard + Lean val.
- **`tokenizer/dolma2-bpe` v1** — Owned `allenai/dolma2-tokenizer` (vocab ~100278, EOS 100257).
- **`tokenizer/bytes-utf8` v1** — Identity byte tokenizer (ids 0–255).

---

## `s3://edullm-landing/` — staging present

| Prefix | Notes |
|--------|-------|
| `pretrain/olmo-mix-1124-31b/v1/` | `dataset.json` + `_VALIDATED.json`; train rows = 31,334,000,834. Not in `edullm-data`. Prefer `pretrain/olmo-original-30b` v1 for training. |
| `pretrain/{lean4-mathlib-bytes,math-memory-full,olmo-150b-dolma2,olmo-original-30b,refhq-regmix-5p5b,regmix-10b}/` | Staging trees for already-promoted IDs |
| `tokenizer/{bytes-utf8,dolma2-bpe}/` | Staging for published tokenizers |
| `_dist/`, `_migrate/`, `_staging/`, `_tmp/` | Internal staging |

---

## `s3://edullm-datasets/` — working store

### Top-level prefixes

| Prefix | Contents |
|--------|----------|
| `olmo30b/` | OLMo-mix-1124 ~30B trimmed (also published as `pretrain/olmo-original-30b`) |
| `olmo100b/` | OLMo-mix-1124 rebalanced / HQ pool (~126.7B) |
| `mixlaw/` | Skill-dag 370M validation mixtures (10B × 8) |
| `regmix/` | RegMix-10B + labels / curriculum sidecars (also published as `pretrain/regmix-10b`) |
| `refhq/` | RefHQ RegMix 5.5B (also published as `pretrain/refhq-regmix-5p5b`) |
| `datamix1-jul22/` | Week-one Data Mix 1 (9.282B packed) |
| `curriculum-p1-jul23/` | Curriculum release `370m-1.25xc-static-v1` |
| `olmo-150b-dolma2/` | OLMo 150B Dolma2 sample (also published under `edullm-data`) |
| `mythos-rdt/` | FineWeb-Edu 3B shards + FarmShare broad mix (~24B) |
| `p1hypothesis/` | Packed Allen Zhu corpus archive |
| `_manifests/` | Path lists / download logs |
| `tmp/` | Scratch (`tmp/linear-attn-vs-gdn/`) |

### Summary

| Dataset | Path | Tokens | Published as | Tokenizer |
|---------|------|-------:|--------------|-----------|
| OLMo-mix ~30B | `olmo30b/olmo-mix-1124-30b/` | 31.334B | `pretrain/olmo-original-30b` v1 | dolma2 |
| OLMo-mix HQ pool | `olmo100b/olmo-mix-1124-30b/` | 126.651B | — | dolma2 |
| Mixlaw 10B×8 | `mixlaw/` | 10B each | — | dolma2 |
| RegMix 10B | `regmix/regmix-10b/` | 10.000B | `pretrain/regmix-10b` v1 | dolma2 |
| RefHQ RegMix 5.5B | `refhq/refhq-regmix-5p5b-v1/` | 5.514B | `pretrain/refhq-regmix-5p5b` v2 | dolma2 |
| Data Mix 1 | `datamix1-jul22/` | 9.282B packed | — | dolma2 @ `5292e5d…` |
| Curriculum static v1 | `curriculum-p1-jul23/releases/370m-1.25xc-static-v1/` | 9.282B | — | dolma2 @ `5292e5d…` |
| OLMo 150B sample | `olmo-150b-dolma2/` | ~155.6B (this layout) | `pretrain/olmo-150b-dolma2` v1 | dolma2 |
| Mythos FineWeb-Edu | `mythos-rdt/shards/` | 3.0B | — | cosmo2 |
| Mythos FarmShare mix | `mythos-rdt/farmshare_40b/` | 24.000B | — | cosmo2 |
| P1 hypothesis pack | `p1hypothesis/stage1-v3-fastbatch-r5/…` | not in manifest | — | — |

---

## Working-store details

### OLMo-mix ~30B — `olmo30b/olmo-mix-1124-30b/`

- Measured dolma2 tokens: **31,334,000,834** (`plan/tokenized_manifest.json`)
- Text under `data/<domain>/`; tokenized uint32 `.npy` under `tokenized/shards/` (218 shards)
- Tokenizer: `allenai/dolma2-tokenizer` (EOS `100257`)

| Domain | Shards | Tokens |
|--------|-------:|-------:|
| dclm | 212 | 29.691B |
| starcoder | 1 | 0.761B |
| pes2o | 1 | 0.461B |
| arxiv | 1 | 0.197B |
| open-web-math | 1 | 0.099B |
| algebraic-stack | 1 | 0.096B |
| wiki | 1 | 0.029B |
| **Total** | **218** | **31.334B** |

### OLMo-mix HQ pool — `olmo100b/olmo-mix-1124-30b/`

- Active measured: **126,650,704,924** (`plan/tokenized_manifest.json`, 2026-07-29)
- Text + tokenized `.npy` under `data/` / `tokenized/shards/`

| Domain | Measured | Plan | Rel err |
|--------|---------:|-----:|--------:|
| dclm | 29.691B | 28.600B | 3.7% |
| arxiv | 22.148B | 20.800B | 6.1% |
| pes2o | 26.379B | 26.300B | 0.3% |
| starcoder | 18.541B | 20.300B | 9.5% |
| open-web-math | 13.238B | 12.200B | 7.8% |
| algebraic-stack | 12.902B | 11.800B | 8.5% |
| wiki | 3.752B | 3.660B | 2.4% |
| **Total** | **126.651B** | — | — |

### Mixlaw — `mixlaw/`

- `READY` + `mixlaw_upload_receipt.json`; recipe `validation_mixtures_10b.json` (8 mixes × 10B)
- `mixes/mix01/` … under `mixlaw/mixes/`

### RegMix 10B — `regmix/regmix-10b/`

- Measured content tokens: **10,000,058,051**
- Text `data/<domain>/`; tokenized `tokenized/<domain>/<domain>.npy`
- Labels: `labels/` (heuristic), `lm_labels/` (LM learnability)
- Curriculum index: `curriculum/`

| Domain | Weight | Measured |
|--------|-------:|---------:|
| dclm | 0.3750 | 3.750B |
| arxiv | 0.2500 | 2.500B |
| starcoder | 0.1406 | 1.406B |
| pes2o | 0.0938 | 0.938B |
| open-web-math | 0.0635 | 0.635B |
| algebraic-stack | 0.0615 | 0.615B |
| wiki | 0.0156 | 0.156B |

### RefHQ RegMix 5.5B — `refhq/refhq-regmix-5p5b-v1/`

- Measured: **5,514,030,574** (`manifests/final_manifest.json`)
- Text Dolma shards + `tokenized/<domain>/<domain>.npy`

| Domain | Weight | Measured |
|--------|-------:|---------:|
| dclm | 0.3750 | 2.068B |
| arxiv | 0.2500 | 1.379B |
| starcoder | 0.1406 | 0.775B |
| pes2o | 0.0938 | 0.517B |
| open-web-math | 0.0635 | 0.350B |
| algebraic-stack | 0.0615 | 0.339B |
| wiki | 0.0156 | 0.086B |

### Data Mix 1 — `datamix1-jul22/`

- **9,281,564,672** packed tokens under `packed/` (headerless uint32; `.npy` suffix)
- Block mix: DCLM 50%, FineWeb-Edu 5%, academic/STEM 10%, code 10%, math 10%, Wikipedia 5%, StackExchange 5%, FLAN 5%

### Curriculum — `curriculum-p1-jul23/releases/370m-1.25xc-static-v1/`

- Same 9.282B population as Data Mix 1; `tokens/` + curriculum views

### OLMo 150B sample — `olmo-150b-dolma2/`

- ~155.6B in this layout under `data/preprocessed/dolma2-0625/…`
- Mix YAMLs: `configs/10b-config.yaml`, `equal-weighting-config.yaml`, `scaled-weighting-config.yaml`

| Domain | Tokens |
|--------|-------:|
| all-dressed-snazzy2 | 119.3B |
| s2pdf-redacted | 19.8B |
| stack-edu | 11.1B |
| finemath-3plus | 4.06B |
| arxiv | 1.25B |
| wikipedia | 0.064B |

### Mythos RDT

- **`shards/`** — FineWeb-Edu only, 3.0B, uint16, cosmo2 tokenizer
- **`farmshare_40b/`** — 24.000B across general / code_technical / math / science_reference / synthetic

### P1 hypothesis — `p1hypothesis/stage1-v3-fastbatch-r5/…`

- Packed archive only (`source/*.tar.gz`, `packages/*.part-*`); token count not in an S3 manifest
