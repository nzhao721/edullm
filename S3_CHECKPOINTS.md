# eduLLM S3 Checkpoints Inventory

Read-only inventory of model / training checkpoints in **sbsandbox** (`056956104102`), as of **2026-07-27** (scan **~2026-07-27 13:00 UTC**).

Access stayed read-only via sb-aws. Progress JSON / `STATUS.txt` values below are point-in-time snapshots from that scan and may advance for live runs.

For training corpora, see [`S3_DATASETS.md`](S3_DATASETS.md).

---

## Buckets

| Bucket | Region (CreateBucket) | Creator (CloudTrail) | Created (UTC) | Role |
|--------|----------------------|----------------------|---------------|------|
| `edullm-olmo-370m-ckpts` | us-east-1 | **nathan.zhao** | 2026-07-23T23:58:10Z | Primary 370M training / experiment checkpoint store |
| `edullm-checkpoints` | us-east-1 | First: **nathan.zhao** (2026-07-22); later recreations: **grant.matherne** (2026-07-25) | 2026-07-25T22:03:18Z (current) | Token-selection / MixLaw / Rel-EMA experiments |
| `edullm-olmo2-370m-cpt-checkpoints` | us-east-1 | **nathan.zhao** | 2026-07-26T11:39:38Z | Unsharded OLMo2 370M CPT ladder checkpoints |

---

## Summary

| Run / prefix | Bucket | Arch / method | Data | Steps on S3 | Status (scan) |
|--------------|--------|---------------|------|-------------|---------------|
| `edullm-370M-refhq-5p5b` | `edullm-olmo-370m-ckpts` | OLMo2 370M CE | RefHQ RegMix 5.5B | 125…1625 (+ tmp) | **Complete** |
| `edullm-370M-blade-regmix10b` | `edullm-olmo-370m-ckpts` | OLMo2 370M BLADE | RegMix 10B + RefHQ ref | 250…2384 | **Complete** |
| `edullm-370M-ce-regmix10b` | `edullm-olmo-370m-ckpts` | OLMo2 370M plain CE | RegMix 10B | 250…2384 | **Complete** |
| `edullm-370M-30B` CPT unsharded | `edullm-olmo2-370m-cpt-checkpoints` | OLMo2 370M (config.yaml) | FarmShare tokenized 30B | 5000, 10000, 15000 | Checkpoints present |
| `token-selection/rel-ema-10b-scratch-v1` | `edullm-checkpoints` | OLMo2 370M Rel-EMA | ~10B budget | 0…2386 | Weights present (final step 2386) |
| `token-selection/rho-excess-10b-scratch-v1` | `edullm-checkpoints` | OLMo2 370M ρ-excess | RegMix 10B + RefHQ ref | 0, 48, 50, 100, 150, 200 | **In progress** (early; 2384 planned) |
| `token-selection/mixlaw-pilot` | `edullm-checkpoints` | DataDecide-60M × 24 mixes | MixLaw pilot | logs + progress only | Complete (no weight checkpoints) |
| `olmo3-370m/run-10b-equal` | `edullm-olmo-370m-ckpts` | OLMo3 370M | 10B equal mix | 0, 3179, 6358, 9537, 12704, 12716 | Checkpoints present |
| `olmo3-370m/trial/{equal,scaled}` | `edullm-olmo-370m-ckpts` | OLMo3 370M trials | equal / scaled | 0, 380–382 | Early trial saves |
| `linear-attn-vs-gdn/*` | `edullm-olmo-370m-ckpts` | Linear attn vs GDN | (see runs) | up to 12716 | Multi-variant |
| `mamba3-370m/*` | `edullm-olmo-370m-ckpts` | Mamba3 370M (Vishnu) | various | see table | Multiple short / mid runs |
| `olmo400m-championship` | `edullm-olmo-370m-ckpts` | Championship tooling + ladder `model.pt` | — | `ladder/step15000/model.pt` | Code + one weight drop |
| `p1hypothesis/.../packages` | `edullm-olmo-370m-ckpts` | Packed ladder ckpt | Allen Zhu stage1 | multipart pack | Archive only |
| `code/`, `code-refhq-b200/`, `code-rho-b200/` | `edullm-olmo-370m-ckpts` | Training scripts | — | — | Code drops (not weights) |
| `lean-split/relay/` | `edullm-olmo-370m-ckpts` | Source tarball | — | — | Relay artifact |
| `token-selection/lean-split/` | `edullm-checkpoints` | Lean staging | — | — | `staging/lean-split-src.tgz` + smoke check |

---

## 1. `s3://edullm-olmo-370m-ckpts/`

Primary shared checkpoint bucket (nathan.zhao). Top-level prefixes:

```
code/  code-refhq-b200/  code-rho-b200/
edullm-370M-blade-regmix10b/  edullm-370M-ce-regmix10b/  edullm-370M-refhq-5p5b/
lean-split/  linear-attn-vs-gdn/  mamba3-370m/  olmo3-370m/  olmo400m-championship/
p1hypothesis/
```

### 1.1 `edullm-370M-refhq-5p5b/` — **complete**

- **Architecture:** `olmo_core.TransformerConfig.olmo2_370M` (`OLMo-2-370M-scratch`, reordered_norm, SiLU FFN hidden 4096, QK-norm, RoPE θ=500000)
- **Method:** plain CE (reference job matched to Rel-EMA stack; RefHQ data)
- **Dataset:** `s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/`
- **Budget:** 5,514,030,574 tokens → **1314** planned steps; saves every 125
- **Hyperparams:** lr `4e-4`, warmup 24, α_f 0.1, global batch 4,194,304 tokens, seq 2048, seed 6198, compile on, attn backend `torch`
- **Status:** `progress/status.json` → `stage=complete` (updated 2026-07-26T22:49:56Z)
- **Checkpoints on S3** (`checkpoints/`):

| Step | Notes |
|-----:|-------|
| 125, 250, 375, 500, 625, 750, 875, 1000, 1125 | Permanent save grid |
| 1250 | Present |
| 1250-tmp | Temporary / intermediate |
| 1315, 1375, 1500 | Post-plan / extended saves |
| **1625** | Latest observed (olmo-core `model_and_optim/*.distcp` + `config.json`) |

Format: distributed olmo-core checkpoint (`model_and_optim/__0_*.distcp`, `.metadata`, `config.json`, `data_paths.txt`).

### 1.2 `edullm-370M-blade-regmix10b/` — **complete**

- **Architecture:** OLMo2 370M
- **Method:** **BLADE** (warmup 509 steps + 1875 BLADE steps; τ=375, K=75, γ=0.6, λ=1.0; 5 BLADE blocks)
- **Train data:** `s3://edullm-dataset-regmix/regmix-10b/`
- **Reference data:** `s3://edullm-dataset-refhq/refhq-regmix-5p5b-v1/`
- **Budget:** 10,000,058,051 tokens → **2384** steps; world size 2; lr `4e-4`
- **Matched teammate run:** `rel-ema-5b-scratch-v1` / GPU7 RefHQ CE stack
- **Status:** `progress/status.json` → `stage=complete` (updated 2026-07-27T07:56:08Z)
- **Checkpoints on S3** (`checkpoints/`): `step250`, `step500`, `step750`, `step1000`, `step1250`, `step1500`, `step1750`, `step2000`, `step2250`, **`step2384`**
- **Also:** `logs/`, `progress/` (`run_meta.json`, `progress.json`, `STATUS.txt`, heartbeat)

### 1.3 `edullm-370M-ce-regmix10b/` — **complete**

- **Architecture:** OLMo2 370M
- **Method:** **plain CE** control for BLADE / token-selection
- **Train data:** `s3://edullm-dataset-regmix/regmix-10b/`
- **Budget:** same 10.000B / **2384** steps; world size 2; lr `4e-4`; save every 250
- **Status:** `progress/status.json` → `stage=complete` (updated 2026-07-27T09:38:15Z)
- **Checkpoints on S3** (`checkpoints/`): `step250`, `step500`, `step750`, `step1000`, `step1250`, `step1500`, `step1750`, `step2000`, `step2250`, **`step2384`**
- **Also:** `logs/`, `progress/`

### 1.4 `olmo3-370m/`

#### `run-10b-equal/`

Steps present: `step0`, `step3179`, `step6358`, `step9537`, `step12704`, `step12716`.

#### `trial/equal/` and `trial/scaled/`

Early trial saves: `step0`, `step380`, `step381`, `step382` each.

### 1.5 `linear-attn-vs-gdn/`

Architecture comparison runs (+ `_code/` scripts).

| Variant | Steps on S3 | Extra |
|---------|-------------|-------|
| `gdn/` | 3179, 6358, 9537, 12704, 12716 | `longctx_eval.json` |
| `linear/` | 3179, 6358, 9537, 12704, 12716 | `longctx_eval.json` |
| `linear-halfkv/` | 3179, 6358, 9537, 12704, 12716 | — |
| `gdn-halfkv/` | 0, 3176, 3179 | shorter / early |

### 1.6 `mamba3-370m/` (Vishnu runs)

| Run prefix | Steps on S3 |
|------------|-------------|
| `vishnu-mamba3-370m-b2-20260726` | 0, 477 |
| `vishnu-mamba3-370m-b2-48b-20260727` | 0, 1908, 2544, 3180, 4452, 5088, 5724, 6042, 6104 |
| `vishnu-mamba3-370m-b2-fix-20260727` | 0 |
| `vishnu-mamba3-370m-b2-lr2e4-20260726` | 0, 636, 1272, 1431 |
| `vishnu-mamba3-370m-b3-nc1-20260726` | 0, 159 |
| `vishnu-mamba3-370m-b3-nc1-48b-20260727` | 1908, 2544, 3180, 3291, 4452, 5088, 5724, 6042, 6104 |
| `vishnu-mamba3-370m-b3-nc1-fix-20260727` | 0, 31 |
| `vishnu-mamba3-370m-b3-nc1-lr2e4-20260726` | 0, 636, 1113, 1151 |
| `vishnu-mamba3-370m-b3-nc1-tmax-20260727` | 0 |

### 1.7 `olmo400m-championship/`

- Root: handoff docs (`HANDOFF.md`, `CAPABILITY_FINDING.md`), bench/probe scripts, notebook, `retention_general_text.jsonl`.
- `ladder/step15000/model.pt` (~1.90 GiB) — ladder weight drop.
- `box-final/` — experiment tree + logs + another copy of tooling / notebook (not a full step grid of weights).

### 1.8 `p1hypothesis/stage1-v3-fastbatch-r5/20260724T050530Z/packages/`

Multipart packed checkpoint archive:

- `checkpoint-olmo-ladder-760m-0.5xc.pack.tar.zst.part-0000` … `part-0050` (+ sha256 sidecars)
- Identity: OLMo ladder **760M @ 0.5×C** (packed), not exploded step directories.

### 1.9 Code / relay (not model weights)

| Prefix | Contents |
|--------|----------|
| `code/` | Train / tokenize / launch scripts (`train_olmo_ladder_370m.py`, `run_pipeline.sh`, …) + `PROGRESS.md` (canonical repo paths: `experiments/baseline/`) |
| `code-refhq-b200/` | B200 RefHQ scripts (`train_olmo3_370m_refhq.py`, `run_refhq_b200.sh`, …) |
| `code-rho-b200/` | RHO / ρ-excess B200 scripts (`rho_code.tgz`, `scripts/`, `token_selection/`) |
| `lean-split/relay/` | `lean-split-src.tgz` only |

---

## 2. `s3://edullm-checkpoints/`

Token-selection experiment bucket. Current top-level: `token-selection/`.

### 2.1 `token-selection/rel-ema-10b-scratch-v1/rel_ema/` — weights present

- **Run id (fingerprint):** `rel-ema-5b-scratch-v1` (prefix says 10b; fingerprint `max_tokens` ≈ 10.005B)
- **Method:** `rel_ema` (rel_k 0.6, α 0.99→0.98, t0=24)
- **Model:** `OLMo-2-370M-scratch` / `olmo2_370M`
- **Tokenizer:** `allenai/dolma2-tokenizer`
- **OLMo revision:** `99e0009ed67679c90da970ec5ba439c9459e3757`
- **Train shape:** seq 2048, global batch 4,194,304 tokens, lr `4e-4`, warmup 24, seed 42
- **Steps on S3:** `step0`, `24`, `125`, `250`, `500`, `750`, `1000`, `1250`, `1500`, `1750`, `2000`, `2250`, `2375`, **`2386`**
- **Format:** olmo-core style (`model_and_optim/`, `train/`, `.metadata.json`) + root `run_fingerprint.json`

### 2.2 `token-selection/mixlaw-pilot/` — logs only (24 mixes)

`mix01` … `mix24`. Each has `logs/` + `progress/` only (no weight checkpoint trees).

Representative `mix01/progress/run_meta.json`:

- **Model:** DataDecide-60M (~57.1M params; d_model 384, 16 layers, 12 heads)
- **Tokenizer:** `allenai/dolma2-tokenizer`
- **Budget:** 285,278,208 tokens → **1451** steps; tokens/step 196,608; lr 0.0058
- **Eval:** task-loss BPB curve (ARC / MMLU splits), eval every 120 steps
- Artifacts: `train.log`, `eval.log`, `task_loss_final.json`, `run_meta.json`

### 2.3 `token-selection/rho-excess-10b-scratch-v1/` — **in progress**

- **Run id:** `rho-excess-10b-scratch-v1`
- **Method:** `rho_excess` (rel_k 0.6, α 0.99→0.98, t0=48; milestone save at step 48)
- **Model:** `OLMo-2-370M-scratch` / `olmo2_370M`
- **Tokenizer:** `allenai/dolma2-tokenizer`
- **OLMo revision:** `99e0009ed67679c90da970ec5ba439c9459e3757`
- **Train data:** `s3://edullm-dataset-regmix/regmix-10b/tokenized`
- **Reference:** `s3://edullm-olmo-370m-ckpts/edullm-370M-refhq-5p5b/checkpoints/step1315/`
- **Train shape:** seq 2048, global batch 4,194,304 tokens, lr `4e-4`, warmup 24, **2384** total steps, 8× GPU HSDP
- **Status (scan):** `stage=train` (updated 2026-07-27T10:37:13Z); early checkpoints only
- **Steps on S3:** `step0`, `step48`, `step50`, `step100`, `step150`, `step200` under `checkpoints/`
- **Format:** olmo-core style + `checkpoints/run_fingerprint.json`

### 2.4 `token-selection/lean-split/` — staging

- `staging/lean-split-src.tgz` + `staging/stp/` (not a published checkpoint run)
- `_smoke_check/` — validation prefix

### 2.5 `token-selection/_scratch/`

Scratch prefixes: `blade/`, `mixlaw/`, `refhq370m/` (not treated as published checkpoint runs).

---

## 3. `s3://edullm-olmo2-370m-cpt-checkpoints/`

Dedicated CPT / unsharded ladder store.

### 3.1 `edullm-370M-30B/`

| Step dir | Objects (typical) |
|----------|-------------------|
| `step5000-unsharded/` | `config.yaml`, `model.pt`, `optim.pt`, `train.pt` |
| `step10000-unsharded/` | same |
| `step15000-unsharded/` | same (`model.pt` ~1.90 GiB, `optim.pt` ~3.79 GiB) |

From `step15000-unsharded/config.yaml`:

- **Run name:** `edullm-370M-30B`
- **Model:** d_model 1024, 16 layers, 16 heads, SwiGLU, RoPE θ=500000, vocab 100278 / emb 100352, amp_bf16, flash attention
- **Tokenizer:** `allenai/dolma2-tokenizer` (EOS 100257, pad 100277)
- **Data:** FarmShare path `…/olmo-ladder-370m-20260722-185217/tokenized/` (DCLM shards + holdout trainrem + extra shards); memmap uint32
- **Duration:** `max_duration: 31303986152T` (~31.3B tokens); `stop_at: 39816`; global batch 192 seqs; device microbatch 4; seq length 4096
- **Optimizer / sched:** AdamW lr ≈ 7.79e-4, cosine warmup 472 steps, α_f 0.1
- **Save:** unsharded every 500; load path referenced `step11500-unsharded` (local FarmShare; not necessarily mirrored to this bucket)
