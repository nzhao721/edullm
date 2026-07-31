# eduLLM S3 Checkpoints Inventory

Read-only inventory in **sbsandbox** (`056956104102`), scan **2026-07-31 ~12:15 UTC** via sb-aws.

All checkpoints: **`s3://edullm-checkpoints/`**. Datasets: [`S3_DATASETS.md`](S3_DATASETS.md).

Progress / status values are point-in-time from that scan.

---

## Top-level prefixes

| Prefix | Role |
|--------|------|
| `olmo-370m/` | 370M training / experiment checkpoints |
| `olmo2-370m-cpt/` | Unsharded OLMo2 370M CPT ladder checkpoints |
| `token-selection/` | Token-selection runs |

---

## Summary

| Run / prefix | Path under `edullm-checkpoints/` | Arch / method | Data | Steps on S3 | Status |
|--------------|----------------------------------|---------------|------|-------------|--------|
| `edullm-370M-refhq-5p5b` | `olmo-370m/edullm-370M-refhq-5p5b/` | OLMo2 370M CE | RefHQ / `pretrain/refhq-regmix-5p5b` v2 | 125…1125, 1315 | Complete |
| `edullm-370M-blade-regmix10b` | `olmo-370m/edullm-370M-blade-regmix10b/` | OLMo2 370M BLADE | RegMix 10B + RefHQ ref | 250…2384 | Complete |
| `edullm-370M-ce-regmix10b` | `olmo-370m/edullm-370M-ce-regmix10b/` | OLMo2 370M CE | RegMix / `pretrain/regmix-10b` v1 | 250…2384 | Complete |
| `edullm-370M-30B` | `olmo2-370m-cpt/edullm-370M-30B/` | OLMo2 370M CPT | FarmShare 30B tokenized | 5000, 10000, 15000, 20000 | Present |
| `rel-ema-10b-scratch-v1` | `token-selection/rel-ema-10b-scratch-v1/` | Rel-EMA | ~10B | 0…2386 | Weights present |
| `rho-excess-10b-scratch-v1` | `token-selection/rho-excess-10b-scratch-v1/` | ρ-excess | RegMix + RefHQ ref | 0, 48, 50, 100, 150, 200 | Early / stalled |
| `mixlaw-pilot` | `token-selection/mixlaw-pilot/` | DataDecide-60M × 24 | MixLaw pilot | logs + progress only | Complete (no weights) |
| `olmo3-370m/run-10b-equal` | `olmo-370m/olmo3-370m/run-10b-equal/` | OLMo3 370M | 10B equal | 0, 3179, 6358, 9537, 12704, 12716 | Present |
| `olmo3-370m/trial/{equal,scaled}` | `olmo-370m/olmo3-370m/trial/…` | OLMo3 trials | equal / scaled | 0, 380–382 | Early |
| `linear-attn-vs-gdn/*` | `olmo-370m/linear-attn-vs-gdn/` | Linear attn vs GDN | — | up to 12716 | Multi-variant |
| `mamba3-370m/*` | `olmo-370m/mamba3-370m/` | Mamba3 370M | various | see below | Multiple runs |
| `olmo400m-championship` | `olmo-370m/olmo400m-championship/` | Championship + ladder | — | `ladder/step15000/model.pt` | Code + weight |
| `p1hypothesis/…/packages` | `olmo-370m/p1hypothesis/…/packages/` | Packed ladder | Allen Zhu | multipart pack | Archive |
| `code/`, `code-refhq-b200/`, `code-rho-b200/` | `olmo-370m/code…` | Scripts | — | — | Code drops |
| `lean-split/relay/` | `olmo-370m/lean-split/relay/` | Tarball | — | — | Relay |
| `token-selection/lean-split/` | `token-selection/lean-split/` | Staging | — | — | Staging |
| `token-selection/_scratch/` | `token-selection/_scratch/` | Scratch | — | — | Scratch |

---

## `olmo-370m/`

Prefixes present:

```
code/  code-refhq-b200/  code-rho-b200/
edullm-370M-blade-regmix10b/  edullm-370M-ce-regmix10b/  edullm-370M-refhq-5p5b/
lean-split/  linear-attn-vs-gdn/  mamba3-370m/  olmo3-370m/  olmo400m-championship/
p1hypothesis/
```

### `edullm-370M-refhq-5p5b/` — complete

- OLMo2 370M CE; data `pretrain/refhq-regmix-5p5b` v2 (working: `edullm-datasets/refhq/refhq-regmix-5p5b-v1/`)
- Budget 5.514B → 1314 steps; saves every 125; lr `4e-4`, seq 2048, global batch 4,194,304
- Status `stage=complete` (2026-07-26T22:49:56Z)
- Steps: 125, 250, 375, 500, 625, 750, 875, 1000, 1125, **1315**
- Format: olmo-core distcp (`model_and_optim/`, `.metadata`, `config.json`)

### `edullm-370M-blade-regmix10b/` — complete

- BLADE (warmup 509 + 1875; τ=375, K=75, γ=0.6, λ=1.0)
- Train: RegMix 10B; ref: RefHQ 5.5B; 2384 steps; world size 2
- Steps: 250…2250, **2384**; plus `logs/`, `progress/`

### `edullm-370M-ce-regmix10b/` — complete

- Plain CE control; RegMix 10B; 2384 steps; save every 250
- Steps: 250…2250, **2384**; plus `logs/`, `progress/`

### `olmo3-370m/`

- `run-10b-equal/`: steps 0, 3179, 6358, 9537, 12704, 12716
- `trial/equal/`, `trial/scaled/`: steps 0, 380, 381, 382 each

### `linear-attn-vs-gdn/`

| Variant | Steps | Extra |
|---------|-------|-------|
| `gdn/` | 3179, 6358, 9537, 12704, 12716 | `longctx_eval.json` |
| `linear/` | 3179, 6358, 9537, 12704, 12716 | `longctx_eval.json` |
| `linear-halfkv/` | 3179, 6358, 9537, 12704, 12716 | — |
| `gdn-halfkv/` | 0, 3176, 3179 | — |

### `mamba3-370m/`

| Run | Steps |
|-----|-------|
| `vishnu-mamba3-370m-b2-20260726` | 0, 477 |
| `vishnu-mamba3-370m-b2-48b-20260727` | 0, 1908, 2544, 3180, 4452, 5088, 5724, 6042, 6104 |
| `vishnu-mamba3-370m-b2-fix-20260727` | 0 |
| `vishnu-mamba3-370m-b2-lr2e4-20260726` | 0, 636, 1272, 1431 |
| `vishnu-mamba3-370m-b3-nc1-20260726` | 0, 159 |
| `vishnu-mamba3-370m-b3-nc1-48b-20260727` | 1908, 2544, 3180, 3291, 4452, 5088, 5724, 6042, 6104 |
| `vishnu-mamba3-370m-b3-nc1-fix-20260727` | 0, 31 |
| `vishnu-mamba3-370m-b3-nc1-lr2e4-20260726` | 0, 636, 1113, 1151 |
| `vishnu-mamba3-370m-b3-nc1-tmax-20260727` | 0 |

### `olmo400m-championship/`

- Docs / scripts / notebook; `ladder/step15000/model.pt` (~1.90 GiB); `box-final/`

### `p1hypothesis/…/packages/`

- Multipart pack: `checkpoint-olmo-ladder-760m-0.5xc.pack.tar.zst.part-*`

### Code / relay

| Prefix | Contents |
|--------|----------|
| `code/` | Train / tokenize / launch scripts |
| `code-refhq-b200/` | B200 RefHQ scripts |
| `code-rho-b200/` | ρ-excess B200 scripts |
| `lean-split/relay/` | `lean-split-src.tgz` |

---

## `token-selection/`

### `rel-ema-10b-scratch-v1/rel_ema/`

- Method `rel_ema`; OLMo2 370M; dolma2 tokenizer; ~10.005B fingerprint budget
- Steps: 0, 24, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2375, **2386**

### `mixlaw-pilot/`

- `mix01`…`mix24`: `logs/` + `progress/` only (DataDecide-60M, ~285M tokens / 1451 steps)

### `rho-excess-10b-scratch-v1/`

- Method `rho_excess`; train RegMix 10B; ref RefHQ step1315; 2384 planned
- Status `stage=train` last updated 2026-07-27T10:37:13Z
- Steps: 0, 48, 50, 100, 150, 200

### Other

- `lean-split/staging/` — source tarball + smoke check
- `_scratch/{blade,mixlaw,refhq370m}/` — scratch

---

## `olmo2-370m-cpt/edullm-370M-30B/`

| Step dir | Typical objects |
|----------|-----------------|
| `step5000-unsharded/` | `config.yaml`, `model.pt`, `optim.pt`, `train.pt` |
| `step10000-unsharded/` | same |
| `step15000-unsharded/` | same (`model.pt` ~1.90 GiB) |
| `step20000-unsharded/` | same |

From `step15000-unsharded/config.yaml`: d_model 1024, 16 layers, 16 heads, SwiGLU, RoPE θ=500000, vocab 100278, dolma2 tokenizer, `max_duration` ~31.3B tokens, AdamW lr ≈ 7.79e-4, cosine warmup 472, α_f 0.1.
