# eduLLM S3 Datasets Inventory

Read-only inventory of training/data corpora in **sbsandbox** (`056956104102`), as of **2026-07-27** (last scan **~2026-07-27 18:30 UTC** / evening PDT). All corpora live under **`s3://edullm-datasets/`** as directory prefixes.

Checkpoint-only buckets and non-corpus prefixes are **not** treated as datasets below. See [`S3_CHECKPOINTS.md`](S3_CHECKPOINTS.md) for checkpoint detail.

---

## S3 consolidation (2026-07-27)

On **2026-07-27**, per-family dataset buckets were consolidated into a single bucket in **sbsandbox** (`056956104102`):

**`s3://edullm-datasets/`** — all training corpora live here as top-level directory prefixes. Legacy per-family bucket names are **deprecated**; use the canonical paths below.

**Legacy bucket deletion (2026-07-27):** Twelve eduLLM legacy buckets were deleted after consolidation: six `edullm-dataset-*`, four `memorysplit-*` (except stephen source), and two checkpoint buckets (see [`S3_CHECKPOINTS.md`](S3_CHECKPOINTS.md#s3-consolidation-2026-07-27)). Ten were removed in the first pass (afternoon UTC); `edullm-dataset-curriculum-p1-jul23` and `edullm-dataset-olmo` required a second pass (~18:30 UTC) to purge versioned delete markers. The only retained legacy **source** bucket is `memorysplit-stephen-056956104102-us-east-1` (see memorysplit table below).

### Top-level prefixes on `edullm-datasets`

| Prefix | Contents |
|--------|----------|
| `olmo30b/` | OLMo-mix-1124 ~30B trimmed corpus |
| `olmo100b/` | OLMo-mix-1124 rebalanced / HQ pool |
| `mixlaw/` | Skill-dag 370M validation mixtures (10B each) |
| `regmix/` | RegMix-optimized mixes |
| `refhq/` | RefHQ filtered corpora |
| `datamix1-jul22/` | Week-one Data Mix 1 |
| `curriculum-p1-jul23/` | Curriculum releases |
| `olmo-150b-dolma2/` | OLMo 150B Dolma2 tokenized sample |
| `mythos-rdt/` | Mythos RDT shards and FarmShare mixes |
| `p1hypothesis/` | P1 hypothesis corpus packages |
| `memorysplit/` | Memory-split experiment corpora (see below) |

### Legacy `edullm-dataset-*` → `edullm-datasets/`

| Legacy bucket | Canonical prefix | Status |
|---------------|------------------|--------|
| `edullm-dataset-regmix` | `regmix/` | **Deleted** 2026-07-27 |
| `edullm-dataset-curriculum-p1-jul23` | `curriculum-p1-jul23/` | **Deleted** 2026-07-27 |
| `edullm-dataset-datamix1-jul22` | `datamix1-jul22/` | **Deleted** 2026-07-27 |
| `edullm-dataset-olmo` | `olmo30b/` | **Deleted** 2026-07-27 |
| `edullm-dataset-olmohq` | `olmo100b/` | **Deleted** 2026-07-27 |
| `edullm-dataset-refhq` | `refhq/` | **Deleted** 2026-07-27 |

### Legacy `memorysplit-*` → `edullm-datasets/memorysplit/`

| Legacy bucket | Canonical prefix | Status |
|---------------|------------------|--------|
| `memorysplit-adarsh-056956104102` | `memorysplit/adarsh/` | **Deleted** 2026-07-27 |
| `memorysplit-corpus-056956104102-us-east-1` | `memorysplit/corpus/` | **Deleted** 2026-07-27 |
| `memorysplit-sid-056956104102` | `memorysplit/sid/` | **Deleted** 2026-07-27 |
| `memorysplit-stephen-056956104102-us-east-1` | `memorysplit/stephen/` | **Complete** at destination — source bucket **retained** (10 objects under `corpus/` still in source; copies verified at destination; source delete blocked by bucket policy on `corpus/*`) |
| `memorysplit-training-056956104102-us-east-1` | `memorysplit/training/` | **Deleted** 2026-07-27 |

**Legacy bucket cleanup (complete 2026-07-27):** All twelve targeted eduLLM legacy buckets (six `edullm-dataset-*`, four `memorysplit-*` except stephen source, and two checkpoint buckets in [`S3_CHECKPOINTS.md`](S3_CHECKPOINTS.md)) are **deleted**. Only `memorysplit-stephen-056956104102-us-east-1` remains as a source holdout (see row above).

Checkpoint bucket consolidation (`edullm-checkpoints`) is documented in [`S3_CHECKPOINTS.md`](S3_CHECKPOINTS.md#s3-consolidation-2026-07-27).

---

**Creator attribution** uses CloudTrail `CreateBucket` management events (principal = assumed role `Intern-<name>-sbsandbox` / session `broker-<name>-…`). Object-level `PutObject` uploaders are **not** available via `lookup-events` (S3 data events are not returned).

**Scan delta vs prior (~01:30 UTC 2026-07-27):** Dataset bucket consolidation into `edullm-datasets/` completed for regmix, refhq, olmo, olmohq, datamix1, and curriculum prefixes (see [S3 consolidation](#s3-consolidation-2026-07-27)). Manifest `LastModified` timestamps unchanged (`tokenized_manifest.json` 2026-07-26T15:29:11Z; RefHQ `final_manifest.json` 2026-07-26T12:13:09Z). Deleted historical buckets `edullm-dataset-skilldag` / `edullm-dataset-v1` remain absent (404).

---

## Summary

| Dataset | S3 location | Token budget | Creator (CloudTrail) | Bucket created | Text on S3 | Tokenized on S3 | Tokenizer |
|---------|-------------|-------------:|----------------------|----------------|:----------:|:---------------:|-----------|
| OLMo-mix-1124 ~30B (trimmed) | `s3://edullm-datasets/olmo30b/olmo-mix-1124-30b/` | **31.334B** measured dolma2 | **nathan.zhao** (`CreateBucket`) | 2026-07-23 00:48:49Z (us-east-1) | Yes | **Yes** (218 shards) | `allenai/dolma2-tokenizer` |
| OLMo-mix-1124 rebalanced / HQ pool | `s3://edullm-datasets/olmo100b/olmo-mix-1124-30b/` | **126.651B** active measured (post top-up trim) | **nathan.zhao** (`CreateBucket`) | 2026-07-23 18:10:00Z (us-east-1); top-up 2026-07-29 | Yes | **Yes** (active inventory) | `allenai/dolma2-tokenizer` |
| Mixlaw 370M validation (10B×8) | `s3://edullm-datasets/mixlaw/` | **10B** per mix | **nathan.zhao** | 2026-07-29 READY | Via mixes | **Yes** | `allenai/dolma2-tokenizer` |
| RegMix-optimized 10B | `s3://edullm-datasets/regmix/regmix-10b/` | 10.000B measured | **nathan.zhao** (`CreateBucket`) | 2026-07-25 19:51:19Z (us-east-1) | Yes | Yes | `allenai/dolma2-tokenizer` |
| RefHQ RegMix 5.5B v1 | `s3://edullm-datasets/refhq/refhq-regmix-5p5b-v1/` | 5.514B measured | **nathan.zhao** (`CreateBucket`) | 2026-07-26 12:11:47Z (us-east-1) | Yes | Yes | `allenai/dolma2-tokenizer` |
| Week-one Data Mix 1 (370M 1.25×C) | `s3://edullm-datasets/datamix1-jul22/` | 9.282B packed | **eric.wu** (`CreateBucket`) | 2026-07-23 02:55:47Z (us-east-2) | No | Yes (packed only) | `allenai/dolma2-tokenizer` @ `5292e5d…` |
| Curriculum release 370m-1.25xc-static-v1 | `s3://edullm-datasets/curriculum-p1-jul23/releases/370m-1.25xc-static-v1/` | 9.282B | **eric.wu** (`CreateBucket`) | 2026-07-23 08:14:32Z (us-east-2) | No | Yes (packed only) | `allenai/dolma2-tokenizer` @ `5292e5d…` |
| OLMo 150B Dolma2 sample | `s3://edullm-datasets/olmo-150b-dolma2/` | 155.6B available | Bucket: **tom.liu**; prefix uploader: **not in LookupEvents** | Bucket 2026-07-23 20:16:51Z (us-east-1) | No | Yes | `allenai/dolma2-tokenizer` |
| Mythos RDT FineWeb-Edu 3B | `s3://edullm-datasets/mythos-rdt/shards/` | 3.0B | Bucket: **tom.liu**; prefix uploader: **not in LookupEvents** | Bucket 2026-07-23 20:16:51Z | No | Yes | `HuggingFaceTB/cosmo2-tokenizer` |
| Mythos FarmShare broad mix | `s3://edullm-datasets/mythos-rdt/farmshare_40b/` | 24.000B | Bucket: **tom.liu**; prefix uploader: **not in LookupEvents** | Bucket 2026-07-23 20:16:51Z | No | Yes | Cosmo2-family vocab 49,152 |
| P1 hypothesis corpus package | `s3://edullm-datasets/p1hypothesis/stage1-v3-fastbatch-r5/` | Not published in manifest | Bucket: **tom.liu**; prefix uploader: **not in LookupEvents** | Bucket 2026-07-23 20:16:51Z | Packaged archive only | Unknown / packed | Not documented on S3 |

---

## 1. OLMo-mix-1124 ~30B (document-trimmed)

- **Path:** `s3://edullm-datasets/olmo30b/olmo-mix-1124-30b/` (consolidated from `edullm-dataset-olmo` on 2026-07-27)
- **Total token budget:** target ~30B; plan estimate ~30.04B; **measured dolma2 `total_content_tokens` = 31,334,000,834** (`plan/tokenized_manifest.json`, created 2026-07-26T15:18:29Z)
- **Creator (CloudTrail):** **nathan.zhao** — `CreateBucket` `edullm-dataset-olmo` as `Intern-nathan.zhao-sbsandbox` / `broker-nathan.zhao-…` (2026-07-23T00:48:49Z, us-east-1). FarmShare run path in README: `olmo-mix-30b-20260722`.
- **Created:** bucket 2026-07-23; README / plan objects dated 2026-07-22; tokenized upload 2026-07-26
- **Upstream:** `allenai/olmo-mix-1124`, shard sample seed `42`
- **Non-tokenized on S3:** **Yes** — Dolma JSONL / `.json.gz` / `.jsonl.zstd` under `data/<domain>/`
- **Tokenized on S3:** **Yes** — 218 flat uint32 `.npy` memmaps under `tokenized/shards/` (+ per-shard `.json` sidecars); index `plan/tokenize_index.jsonl`
- **Tokenizer:** `allenai/dolma2-tokenizer` (EOS `100257`). Earlier trim counting used `allenai/OLMo-2-0425-1B` (counting only).

### Domain splits (dolma2 measured)

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

---

## 2. OLMo-mix-1124 rebalanced corpus (olmohq)

- **Path:** `s3://edullm-datasets/olmo100b/olmo-mix-1124-30b/` (consolidated from `edullm-dataset-olmohq` on 2026-07-27)
- **Total token budget:** plan estimate was ~95.91B; **active measured dolma2 after starcoder/pes2o top-up trim = 126,650,704,924** (`plan/tokenized_manifest.json`, updated 2026-07-29). Pre-topup backup: `plan/tokenized_manifest.pre_topup.json` (100.179B / 317 shards). Overshoot inventory kept on S3 as `plan/tokenized_manifest.overshoot.json` (objects retained; excluded from active inventory).
- **Creator (CloudTrail):** **nathan.zhao** — `CreateBucket` `edullm-dataset-olmohq` (2026-07-23T18:10:00Z, us-east-1). FarmShare run path in README: `olmo-mix-rebalance-20260723`. Top-up run: `olmohq-topup-20260728-185841`.
- **Created:** bucket 2026-07-23; README 2026-07-24; tokenized upload 2026-07-26; top-up+trim 2026-07-29
- **Non-tokenized on S3:** **Yes** — text shards under `data/<domain>/`
- **Tokenized on S3:** **Yes** — flat uint32 `.npy` memmaps under `tokenized/shards/` (+ per-shard `.json`); index `plan/tokenize_index.jsonl`
- **Tokenizer:** `allenai/dolma2-tokenizer` (EOS `100257`)
- **Availability gate:** `|plan − measured| / measured ≤ 10%` for all 7 domains (`plan/availability_after_topup.json`)

### Domain splits (dolma2 measured, active inventory)

| Domain | Measured tokens | Plan | Rel err |
|--------|----------------:|-----:|--------:|
| dclm | 29.691B | 28.600B | 3.7% |
| arxiv | 22.148B | 20.800B | 6.1% |
| pes2o | 26.379B | 26.300B | 0.3% |
| starcoder | 18.541B | 20.300B | 9.5% |
| open-web-math | 13.238B | 12.200B | 7.8% |
| algebraic-stack | 12.902B | 11.800B | 8.5% |
| wiki | 3.752B | 3.660B | 2.4% |
| **Total** | **126.651B** | — | — |

This pool is the source for RegMix-10B and mixlaw validation slices. **`regmix/regmix-10b` was not modified by the top-up.**

Top-up tooling: `datasets/olmohq/submit_olmohq_topup.sh` → `plan_olmohq_topup.py` / `finalize_olmohq_topup_upload.py` / `trim_olmohq_topup_manifest.py`.

---

## 2b. Mixlaw 370M validation mixtures (10B each)

- **Path:** `s3://edullm-datasets/mixlaw/`
- **READY:** `2026-07-29T03:24:36Z` (`READY`, `mixlaw_upload_receipt.json`)
- **Recipe:** `validation_mixtures_10b.json` (8 mixes)
- **mix01:** server-side copy *from* `s3://edullm-datasets/regmix/regmix-10b/` into `mixlaw/mixes/mix01/` only — **regmix source read-only**
- **Other mixes:** materialized from olmohq working pool (`olmo-mix-1124`, `mix07`, `mix18`, `ML-min1pct`, `ML-near-opt-3`, `LGB-min1pct`, `LGB-near-opt-5`)
- **Submit:** `experiments/skill-dag/mixlaw/submit_mixlaw_validation_10b.sh`

---

## 3. RegMix-optimized OLMo-mix 10B

- **Path:** `s3://edullm-datasets/regmix/regmix-10b/` (consolidated from `edullm-dataset-regmix` on 2026-07-27)
- **Total token budget:** **10,000,058,051** measured content tokens
- **Creator (CloudTrail):** **nathan.zhao** — `CreateBucket` `edullm-dataset-regmix` (2026-07-25T19:51:19Z, us-east-1). FarmShare run path in README: `regmix-10b-20260725-124810`.
- **Created:** bucket + corpus 2026-07-25
- **Source:** `s3://edullm-datasets/olmo100b/olmo-mix-1124-30b`
- **Non-tokenized on S3:** **Yes** — `data/<domain>/<domain>-regmix.json.gz`
- **Tokenized on S3:** **Yes** — `tokenized/<domain>/<domain>.npy` (uint32) + `.json` metadata
- **Difficulty labels on S3:** upload via `datasets/regmix/finalize_regmix_labels_upload.py` / `submit_regmix_labels_upload.sh`
  - Heuristic (compression / Flesch / MTLD): `s3://edullm-datasets/regmix/regmix-10b/labels/` (`READY`, `SCHEMA.json`, `docs/`, `metrics/`, `metrics_index.jsonl.gz`)
  - LM learnability: `s3://edullm-datasets/regmix/regmix-10b/lm_labels/` (same portable layout; local FarmShare tree may be `lm_labels/labels/`)
  - Upload receipt: `labels_upload_manifest.json` at the corpus prefix (paths, byte counts, READY timestamps)
- **Curriculum training index on S3:** `s3://edullm-datasets/regmix/regmix-10b/curriculum/` — produced by `experiments/curriculum/scripts/build_curriculum_index.py` (doc/chunk ranks + tokenized memmaps for curriculum arms)
- **Tokenizer:** `allenai/dolma2-tokenizer` (EOS `100257`)

### Domain splits (RegMix weights → measured)

| Domain | Weight | Target | Measured |
|--------|-------:|-------:|---------:|
| dclm | 0.3750 | 3.750B | 3.750B |
| arxiv | 0.2500 | 2.500B | 2.500B |
| starcoder | 0.1406 | 1.406B | 1.406B |
| pes2o | 0.0938 | 0.938B | 0.938B |
| open-web-math | 0.0635 | 0.635B | 0.635B |
| algebraic-stack | 0.0615 | 0.615B | 0.615B |
| wiki | 0.0156 | 0.156B | 0.156B |

---

## 4. RefHQ RegMix 5.5B v1

- **Path:** `s3://edullm-datasets/refhq/refhq-regmix-5p5b-v1/` (consolidated from `edullm-dataset-refhq` on 2026-07-27)
- **Total token budget:** **5,514,030,574** measured content tokens (`manifests/final_manifest.json`)
- **Creator (CloudTrail):** **nathan.zhao** — `CreateBucket` `edullm-dataset-refhq` as `Intern-nathan.zhao-sbsandbox` / `broker-nathan.zhao-…` (2026-07-26T12:11:47Z, us-east-1). FarmShare run path in manifests: `refhq-regmix-5p5b-v1`.
- **Created:** bucket + corpus 2026-07-26
- **Source:** independent HQ-filtered domain pulls (not sampled from `edullm-datasets/olmo100b`); budget profile `regmix-5p5`
- **Non-tokenized on S3:** **Yes** — Dolma JSONL shards under `<domain>/documents-*.json.gz`
- **Tokenized on S3:** **Yes** — `tokenized/<domain>/<domain>.npy` (uint32) + `.json` metadata
- **Tokenizer:** `allenai/dolma2-tokenizer` (EOS `100257`)

### Domain splits (RegMix-5.5 weights → measured)

| Domain | Weight | Target | Measured | HQ filter |
|--------|-------:|-------:|---------:|-----------|
| dclm | 0.3750 | 2.068B | 2.068B | DataDecide `v0_rep32_ft7percentile_fw2` |
| arxiv | 0.2500 | 1.379B | 1.379B | random sample from `allenai/olmo-mix-1124` |
| starcoder | 0.1406 | 0.775B | 0.775B | `dolma-code-hq` (77.8% pass rate) |
| pes2o | 0.0938 | 0.517B | 0.517B | random |
| open-web-math | 0.0635 | 0.350B | 0.350B | `openwebmath-hq` (2.0% pass rate) |
| algebraic-stack | 0.0615 | 0.339B | 0.339B | `algebraic-stack-heuristic` |
| wiki | 0.0156 | 0.086B | 0.086B | random (`wikimedia/wikipedia` `20231101.en`) |

Manifests: `manifests/plan.json`, `manifests/final_manifest.json`, `manifests/tokenized_manifest.json`

---

## 5. Week-one Data Mix 1 — 370M 1.25×C

- **Path:** `s3://edullm-datasets/datamix1-jul22/` (consolidated from `edullm-dataset-datamix1-jul22` on 2026-07-27)
- **Total token budget:** **9,281,564,672** packed tokens (2,266,007 × 4096-token sequences)
- **Creator (CloudTrail):** **eric.wu** — `CreateBucket` `edullm-dataset-datamix1-jul22` as `Intern-eric.wu-sbsandbox` (2026-07-23T02:55:47Z, **us-east-2**). Upstream HF view: `ericrcwu/week1-general-20b-dolma2-v1` @ `5b5503d35e58c0b5e33d0d6b51a074cc8d79f172`.
- **Created:** bucket 2026-07-23 (us-east-2); README dated 2026-07-22 / 2026-07-23
- **Non-tokenized on S3:** **No** (this bucket intentionally excludes raw text / 20B-only objects)
- **Tokenized on S3:** **Yes** — headerless little-endian uint32 memmaps under `packed/` (`.npy` suffix does **not** imply a NumPy header)
- **Tokenizer:** `allenai/dolma2-tokenizer` @ revision `5292e5d6c0f40b67cc765fe41bec991cf4345b5c`

### Domain mix (within each 4096-token block)

| Domain | Share |
|--------|------:|
| DCLM | 50% |
| FineWeb-Edu | 5% |
| Academic / STEM | 10% |
| Code | 10% |
| Math | 10% |
| Wikipedia | 5% |
| StackExchange | 5% |
| FLAN | 5% |

Manifest: `views/370m-1.25xc/manifest.json`

---

## 6. Curriculum release `370m-1.25xc-static-v1`

- **Path:** `s3://edullm-datasets/curriculum-p1-jul23/releases/370m-1.25xc-static-v1/` (consolidated from `edullm-dataset-curriculum-p1-jul23` on 2026-07-27)
- **Total token budget:** **9,281,564,672** (same canonical population as datamix1)
- **Creator (CloudTrail):** **eric.wu** — `CreateBucket` `edullm-dataset-curriculum-p1-jul23` (2026-07-23T08:14:32Z, **us-east-2**). Same week-one mix packaging (release schema `edullm-curriculum/v1`).
- **Created:** bucket + `release.json` 2026-07-23
- **Non-tokenized on S3:** **No**
- **Tokenized on S3:** **Yes** — `tokens/` uint32 packed objects + curriculum views/profiles/annotations
- **Tokenizer:** `allenai/dolma2-tokenizer` @ `5292e5d6c0f40b67cc765fe41bec991cf4345b5c` (`provenance/tokenizer.lock.json`)

### Domain splits

Same block-level mix as Data Mix 1 (DCLM 50%, FineWeb-Edu 5%, academic/STEM 10%, code 10%, math 10%, Wikipedia 5%, StackExchange 5%, FLAN 5%).

---

## 7. OLMo 150B Dolma2 tokenized sample

- **Path:** `s3://edullm-datasets/olmo-150b-dolma2/`
- **Total token budget:** **155.6B** tokens available in pool (uint32 `.npy` bytes ÷ 4)
- **Creator (CloudTrail):** Shared bucket `edullm-datasets` created by **tom.liu** (`CreateBucket`, 2026-07-23T20:16:51Z, us-east-1). Related: Tom also created `olmo-data-150b-dolma3` the same day. **Prefix uploader** for `olmo-150b-dolma2/` is **not** in CloudTrail LookupEvents (S3 data events unavailable). Upstream content: AI2 OLMo-mix-0625-150Bsample / Dolma2-0625.
- **Created:** bucket 2026-07-23; local README / mix YAMLs 2026-07-25
- **Non-tokenized on S3:** **No** (tokenized shards only under this prefix)
- **Tokenized on S3:** **Yes** — `data/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer/<domain>/…/*.npy`
- **Tokenizer:** `allenai/dolma2-tokenizer`
- **Mix configs:** `configs/10b-config.yaml` (water-fill 10B), `equal-weighting-config.yaml`, `scaled-weighting-config.yaml`
- **Held-out:** `heldout-val/` (one small `.npy` per domain)

### Domain splits (available pool)

| Domain | Tokens available |
|--------|-----------------:|
| all-dressed-snazzy2 | 119.3B |
| s2pdf-redacted | 19.8B |
| stack-edu | 11.1B |
| finemath-3plus | 4.06B |
| arxiv | 1.25B |
| wikipedia | 0.064B |
| **Total** | **155.6B** |

**10B water-fill training mix** (`configs/10b-config.yaml`, `requested_tokens=10_000_000_000`):

| Source | Target ratio | ~Tokens of 10B |
|--------|-------------:|---------------:|
| all-dressed-snazzy2 | ~0.216 | ~2.17B |
| s2pdf-redacted | ~0.216 | ~2.17B |
| stack-edu | ~0.216 | ~2.17B |
| finemath-3plus | ~0.216 | ~2.17B |
| arxiv (capped) | ~0.129 | ~1.25B |
| wikipedia (capped) | ~0.006 | ~0.064B |

---

## 8. Mythos RDT — FineWeb-Edu 3B shards

- **Path:** `s3://edullm-datasets/mythos-rdt/shards/`
- **Total token budget:** **3,000,000,000** (`manifest.json` `tokens_total`)
- **Creator (CloudTrail):** Shared bucket created by **tom.liu**. Prefix uploader for `mythos-rdt/shards/` **not** in LookupEvents. Pipeline identity from path/manifest only: Mythos / RDT (`only: fineweb-edu`).
- **Created:** objects 2026-07-25; bucket 2026-07-23
- **Non-tokenized on S3:** **No**
- **Tokenized on S3:** **Yes** — `shard_XXXXX.bin` uint16
- **Tokenizer:** `HuggingFaceTB/cosmo2-tokenizer` (vocab 49,152)
- **Domain splits:** single source — **FineWeb-Edu only** (`only: fineweb-edu`, phase `base`)

---

## 9. Mythos FarmShare broad mix (`farmshare_40b`)

- **Path:** `s3://edullm-datasets/mythos-rdt/farmshare_40b/`
- **Total token budget:** **24,000,112,560** tokens across 240 shards in `manifests/broad.json` (prefix name says “40b”; measured inventory is ~24.0B)
- **Creator (CloudTrail):** Shared bucket created by **tom.liu**. Prefix uploader for `mythos-rdt/farmshare_40b/` **not** in LookupEvents. Pipeline identity from path only: Mythos / RDT.
- **Created:** objects 2026-07-25; bucket 2026-07-23
- **Non-tokenized on S3:** **No**
- **Tokenized on S3:** **Yes** — content-addressed `*.uint16` under `shards/`
- **Tokenizer:** Cosmo2-family tokenizer artifacts in `tokenizer/` (`vocab_size` 49,152; matches Cosmo2 special-token layout)

### Domain splits (manifest weights + stored tokens)

| Domain | Mix weight | Shards | Tokens |
|--------|-----------:|-------:|-------:|
| general | 0.35 | 84 | 8.400B |
| code_technical | 0.30 | 72 | 7.200B |
| math | 0.15 | 36 | 3.600B |
| science_reference | 0.10 | 24 | 2.400B |
| synthetic | 0.10 | 24 | 2.400B |
| **Total** | 1.00 | 240 | **24.000B** |

---

## 10. P1 hypothesis — Allen Zhu corpus package

- **Path:** `s3://edullm-datasets/p1hypothesis/stage1-v3-fastbatch-r5/20260724T050530Z/`
- **Total token budget:** **Not published** on S3 (multipart packed archive only)
- **Creator (CloudTrail):** Shared bucket created by **tom.liu**. Prefix uploader for `p1hypothesis/` **not** in LookupEvents. Package name only: `corpus-allenzhu-v2` (Allen Zhu content identity, not CloudTrail principal).
- **Created:** source tarball 2026-07-23; package parts / run stamp 2026-07-23–2026-07-24; bucket 2026-07-23
- **Contents:** `source/p1hypothesis-stage1-v3-fastbatch-r5-20260723.tar.gz` + `packages/corpus-allenzhu-v2.pack.tar.zst.part-*`
- **Non-tokenized on S3:** Archive only (not exploded Dolma JSONL)
- **Tokenized on S3:** Unknown without unpacking the pack
- **Tokenizer:** Not documented in the S3 listing
