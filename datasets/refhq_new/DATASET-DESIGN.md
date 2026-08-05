# Dataset design: `pretrain/refhq-new`

**purpose:** Filtered instruct-sourced CE corpus for OLMo-2 370M reference / rho-1 scoring, tuned toward the 20-label OLMES BPB suite (not IFEval/tools/safety).

**family:** `pretrain`  
**profile:** `pretrain-tokens/v1` + `vendored/v1` raw companion  
*(Prefer `text-corpus/v1` when Gate A ships it; until then FineWeb-style `vendor/`.)*  
**name:** `refhq-instruct` → `pretrain/refhq-instruct`  
*(Working-store prefix stays `refhq/refhq-new/`; `refhq-new` is invalid as a
dataset name because `new` is a forbidden version token in edullm-data §2.)*  
(Sibling of `pretrain/refhq-regmix-5p5b`: same reference role, instruct mix instead of HQ web. No size suffix — realized size is whatever one pass yields.)

**tokenizer:** HF `allenai/dolma2-tokenizer` → publish dep `tokenizer/dolma2-bpe` (EOS 100257). Already published; do not republish.

**budget:** One pass of filtered unique data only. No upsampling, no 5.5B cap. Count tokens during build and report realized size in plan summary + `publish()` `sources[]`.

**dedup:** None (explicit). Keep SmolTalk `openhermes-100k` even though it overlaps full OpenHermes-2.5.

---

## Irreversible decisions

| Decision | Choice |
|---|---|
| slice path | `tokens/<source>/<domain>/<split>-NNNNN.u32le.bin` (two path labels) |
| held-out | **0.15% of documents per `(source, domain)` before tokenize** (seed 42), then tokenize train/val separately |
| dtype / ext | `uint32` / `.u32le.bin` |
| target shard size | ~1 GiB; no `-of-N` in filenames |

### Path labels

- **source:** `tulu-v2` \| `openhermes-25` \| `tulu-3` \| `hermes-3` \| `smoltalk` \| `dolci`
- **domain:** `general` \| `math` \| `code` \| `science` \| `chat` — from row metadata / SmolTalk config; default `general`

Companion text (optional): `text/<source>/<domain>/…` via `stage_text_companion`.

---

## Inclusion / exclusion rules

Metadata drops run **before** Dolma English. Rules live in [`exclusion_rules.yaml`](exclusion_rules.yaml); helpers in [`exclusion.py`](exclusion.py). Domain labels: [`domain_map.py`](domain_map.py).

| Source | Keep | Drop |
|---|---|---|
| **Tulu-v2** | all | — |
| **OpenHermes-2.5** | all | optional: drop rows with `language` set and not en/eng |
| **Tulu-3** | FLAN, WildChat, personas math/GSM/code/algebra, Numina-TIR, Evol CodeAlpaca, SciRIFF, TableGPT, No Robots, OASST, hardcoded | `wildguardmix`, `wildjailbreak`, `coconot`, `tulu-3-sft-personas-instruction-following`, `aya` / Aya |
| **Hermes-3** | all (~959K; no category column) | — |
| **SmolTalk** | all configs except listed | `apigen-80k`, `smol-constraints` |
| **Dolci** | everything else | `domain == Safety`; Precise IF; CoCoNot; Aya; WildGuard/WildJailbreak; Tool Use `source_dataset`; any row with non-null `function_calls`/`functions` |

### Dolma English filter (all kept docs)

Configs under [`configs/`](configs/); tag+mix wrapper in [`process.py`](process.py) (mirrors RefHQ).

- **Taggers:** `ft_lang_id_en_paragraph_with_doc_score_v2` (optional companion tagger `ft_lang_id_en_doc_v2`)
- **Mix include:** document English score (`…__doc_en`) ≥ **0.5**
- No Gopher/C4/toxicity/NSFW — English only
- Conversations → Dolma docs: flatten `messages`/`conversations` to plain text (role labels optional but consistent); one example = one document

Drop Tulu-3/Dolci Aya via metadata **before** Dolma so those rows never hit lang-id.

---

## Layout

```
tokens/<source>/<domain>/<split>-NNNNN.u32le.bin
text/<source>/<domain>/…          # companion via stage_text_companion
```

Working / landing: `s3://edullm-datasets/refhq/refhq-new/` → Gate A → `s3://edullm-data/pretrain/refhq-instruct/v1/`.

---

## Dependencies

- **tokenizer:** `tokenizer/dolma2-bpe` — must already be published
- **parent:** none

---

## Deferred (backfillable, don't block)

`about` / rich mix table / measured `sources[]` token counts after the build.

---

## `publish()` call (target)

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
from edullm_text_companion import PUBLISH_PROFILE, TEXT_GROUP_META
import datetime

publish(
    stage_dir,
    dataset_id="pretrain/refhq-instruct",
    purpose=(
        "One-pass filtered instruct mix (Tulu-v2/OH-2.5, Tulu-3/Hermes-3, SmolTalk/Dolci) "
        "for OLMo-2 370M CE reference / rho-1; tool/safety/IF/Aya removed; Dolma English; "
        "tuned for 20-label OLMES BPB"
    ),
    profile=PUBLISH_PROFILE,  # tokens + text
    tokenizer="tokenizer/dolma2-bpe",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    group_meta=TEXT_GROUP_META,
    about=...,  # TODO after counts
    sources=[...],  # measured token counts per HF source after filters
    license={"id": "ODC-By-1.0", "basis": "declared"},
    notes="No dedup. No upsampling. Realized size is one filtered pass.",
    limitations=[{"kind": "license", "detail": "Tulu ODC-BY with some NC subsets; research use"}],
)
```
