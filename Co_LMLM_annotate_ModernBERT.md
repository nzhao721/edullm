# Reviewer's guide — `Co_LMLM_annotate_ModernBERT.ipynb`

This document explains what the companion notebook is for, how it is supposed to work, and how to
diagnose it if it misbehaves. It is written for an engineer or agent who has **no prior context** on
this project. Read this before editing the notebook.

---

## 1. What we are trying to do

We are reproducing the **annotation pipeline** of **Co-LMLM** (*Continuous-Query Limited Memory
Language Models*, arXiv:2607.07707). Co-LMLM trains a language model that **externalizes facts** to
an external knowledge base instead of memorizing them: during pre-training, factual spans in the
corpus are tagged, and the LM loss is masked over those spans so the model learns to *retrieve*
them rather than store them in its weights.

Producing those fact tags at scale is done with **two distilled annotators**:

1. **A fact-span annotator** — a ModernBERT encoder fine-tuned for BIO token classification that
   marks *where* the facts are. (This is the model this notebook uses.)
2. A question generator — a Qwen decoder that writes a question/answer per span. **Not used here.**

**This notebook runs only annotator #1** over a corpus, to produce fact-span annotations. It is the
cheap, GPU-light half of the pipeline (a single encoder forward pass per document). The question
generator is intentionally out of scope — see "Scope" below.

The end goal of *this* notebook: take a corpus of raw documents sitting in a Google Drive folder,
run the tuned ModernBERT tagger over every document, and write the resulting fact spans back to
Google Drive as compressed JSONL.

---

## 2. What the notebook does (input → output)

| | |
|---|---|
| **Input** | A Google Drive folder (by ID) **shared with the Colab account** — the `edullm` root — walked recursively. If it contains no raw text, the notebook streams the upstream HF dataset named in its `meta.json` instead. |
| **Model** | A tuned `ModernBertForTokenClassification` checkpoint in the user's My Drive. |
| **Output** | One `*.annotations.jsonl.zst` shard per input file, written to My Drive, plus a `_manifest.json`. |
| **Hardware** | Any CUDA GPU (A100 ideal; T4/L4 fine — the model is 395M params). CPU works but is slow. |

### Output schema (one JSON object per line, zstd-compressed)

```json
{
  "id": "somefile-42",
  "source": "fineweb-edu",
  "text": "full document text (present only if INCLUDE_TEXT=True)",
  "annotations": [
    {"span": "Albert Spalding", "char_start": 37, "char_end": 52, "faithful": true},
    {"span": "1911", "char_start": 67, "char_end": 71, "faithful": true}
  ]
}
```

`char_start`/`char_end` index into `text`; `span == text[char_start:char_end]` always holds
(`faithful` is `true` by construction). There are no `question`/`answer` fields — this is the
tagger-only output.

---

## 3. Environment and dependencies

- Runs on **Google Colab** with a GPU runtime. It is Colab-specific: it imports `google.colab`
  (`auth`, `drive`) and `googleapiclient` (the Google Drive API), which exist in Colab by default.
- `%pip install -q -U "transformers>=4.48" zstandard pyarrow` installs the rest. **`transformers>=4.48`
  is mandatory** — ModernBERT support landed in 4.48. As of this writing Colab installs
  transformers **5.14.x**, which is fine; the notebook was validated against 5.14.1.
- `torch` is preinstalled by Colab.

---

## 4. Prerequisites the user must satisfy (most failures start here)

1. **The model must be in Drive.** `MODEL_DIR` (default
   `/content/drive/MyDrive/co-lmlm-span-tagger/final`) must be a folder containing `config.json`,
   `model.safetensors`, and tokenizer files (`tokenizer.json`, `tokenizer_config.json`). If the
   default path is wrong, cell 8 auto-searches My Drive for a ModernBERT token-classifier.
2. **The input folder must be shared with the Colab Google account** (viewer is enough). It is read
   by ID via the Drive API, so it does **not** need to be in My Drive and does **not** need to be
   public.
3. **GPU runtime** selected (Runtime → Change runtime type → GPU).
4. **`TEXT_FIELD` must match the data schema** (see cell 6). This is the single most common
   misconfiguration — read the troubleshooting section.

---

## 5. Cell-by-cell walkthrough and design rationale

- **Cell 2 — runtime check.** Prints torch version / GPU / bf16 support. Warns (does not fail) if
  no GPU.
- **Cell 3 — install deps.**
- **Cell 5 — auth + mount.** Two separate auth flows, both required and intentional:
  - `auth.authenticate_user()` authorizes the **Drive API**, used to read the *shared* input folder.
  - `drive.mount("/content/drive")` mounts **My Drive** for writing output.
  - **Why two?** A folder *shared with you* is **not** exposed under a mounted Drive — Colab's mount
    only surfaces *My Drive* and Shared Drives, not "Shared with me" items. The Drive API can reach
    any file you have access to **by ID**, so it is used for input. This split is the central design
    decision of the notebook; do not "simplify" it by trying to read the input from a mounted path.
- **Cell 6 — configuration.** All knobs live here (see reference below).
- **Cell 8 — load model.** Notable choices:
  - Device/dtype auto-selected: `bfloat16` on bf16-capable GPUs (A100), `float16` on other GPUs
    (T4), `float32` on CPU.
  - **The model is loaded without a dtype kwarg, then cast with `model.to(dtype=...)`.** This is
    deliberate: transformers renamed the `from_pretrained(torch_dtype=...)` argument to `dtype=` at
    v4.56, so passing either name is version-fragile. Casting after load avoids the issue. **Do not
    add `torch_dtype=`/`dtype=` to `from_pretrained` here.**
  - `attn_implementation` is tried as `sdpa` then `eager`. ModernBERT can use `flash_attention_2`,
    but flash-attn is not installed on Colab by default; `sdpa` works on CUDA and `eager` everywhere.
  - `model.config.reference_compile = False` disables ModernBERT's optional `torch.compile` path,
    which can error or stall on Colab.
- **Cell 10 — `tag_batch` (the core).** Tokenizes a batch with `return_offsets_mapping=True` and
  `return_special_tokens_mask=True`, runs the model, argmaxes per token, and decodes the BIO tag
  sequence into character spans:
  - `B-FACT` opens a span, `I-FACT` extends it, anything else (or a special/pad token) closes it.
    An `I-FACT` with no open span is treated as a start (matches the training-time decoder).
  - **Whitespace trimming** (`while ... text[s].isspace(): s += 1`) is **essential and must not be
    removed.** ModernBERT uses a byte-level BPE tokenizer whose offset mapping folds the *preceding
    space* into a token's span; without trimming, every span's `char_start` is shifted and `span !=
    text[char_start:char_end]`, breaking faithfulness.
  - Requires a **fast tokenizer** (offset mapping is a fast-tokenizer feature). ModernBERT ships one
    (`tokenizer.json`).
  - Ends with a demo `print(tag_batch([...]))` so you can eyeball a known example.
- **Cell 12 — list input files (Drive API).** Recursively walks the folder (handling subfolders and
  pagination). Uses `supportsAllDrives=True, includeItemsFromAllDrives=True` so it also works for
  Shared Drives. Prints the root folder's name and the file list **grouped by subfolder**, marking
  unreadable extensions with `x` — **use this to confirm the folder is reachable and to see where
  the raw text shards actually are.** It also downloads and prints any small `.json` sidecars and
  harvests an upstream HF repo id from them for the fallback. Sidecar keys are consulted in
  priority order (`hf_path` first); internal ids like `parent_corpus: "pretrain/fineweb-edu-10b"`
  are rejected because they look like an HF repo id but are not one.
- **Cell 14 — annotate loop (resumable).**
  - `read_records(path, name)` dispatches by extension: `.jsonl`/`.ndjson`, `.json` (array or
    `{"data": [...]}`), `.parquet` (pyarrow), `.csv`/`.tsv`, `.txt` (whole file = one doc), and
    `.zst` / `.gz` variants of the JSON forms (including `train-*.jsonl.gz`). Each record must
    expose the text under `TEXT_FIELD`.
  - For each file: download to `/content/_annotate_tmp`, stream records, accumulate `BATCH` docs,
    `tag_batch`, write results to `OUTPUT_DIR/<stem>.annotations.jsonl.zst`, delete the temp file.
  - **Resumability:** finished input file names are recorded in `OUTPUT_DIR/_manifest.json`. On
    re-run, files already in the manifest are skipped. This is what makes it safe against Colab
    session timeouts — just re-run the whole notebook.
  - `MAX_FILES` / `MAX_DOCS_PER_FILE` bound a trial run.
  - **HF fallback.** Files whose first few records carry no `TEXT_FIELD` are reported as
    `skipped: no 'text' in records; keys seen: [...]` and do not count against `MAX_FILES`. If
    *nothing* got annotated from Drive and `INPUT_MODE` is `"auto"` or `"hf"`, the cell streams the
    upstream dataset (`HF_PATH` or the `hf_path` from `meta.json`) instead, sharding every
    `SHARD_DOCS` docs and recording shard progress in the manifest.
  - Prints `docs/s` periodically — **use this to measure real throughput** and extrapolate ETA.
- **Cell 16 — verify.** Reads back the newest shard and asserts every span is a verbatim substring
  at its offset. `0 offset mismatches` is the success signal.

---

## 6. Configuration reference (cell 6)

| Name | Default | Meaning / when to change |
|---|---|---|
| `INPUT_MODE` | `"drive"` | `"drive"` reads only the Drive folder; `"hf"` only streams the upstream HF dataset; `"auto"` tries Drive and falls back to HF if the folder holds no raw text. |
| `INPUT_FOLDER_ID` | `14l8nIqn...` | Drive folder ID of the corpus to annotate — the shared `edullm` root. Walked **recursively**, so subfolders are covered. |
| `INPUT_PATH_PREFIX` | `"fineweb-edu-1b-smollm2-raw/"` | Only annotate files under this relative path. Skips the tokenized sibling. Set `None` for the whole tree. |
| `HF_PATH` / `HF_NAME` / `HF_SPLIT` | `None` / `None` / `"train"` | Upstream dataset for the HF fallback. Left `None`, they are taken from the `hf_path` / `hf_name` in the folder's `meta.json`. |
| `MODEL_DIR` | `/content/drive/MyDrive/co-lmlm-span-tagger/final` | Path to the tuned model folder. Change if stored elsewhere. |
| `OUTPUT_DIR` | `/content/drive/MyDrive/co-lmlm-annotations` | Where shards + manifest are written (My Drive). |
| `TEXT_FIELD` | `"text"` | **Field/column holding document text. Must match the data.** |
| `ID_FIELD` | `"id"` | Field holding a stable id; auto-generated `<stem>-<n>` if absent. |
| `SOURCE_FIELD` / `SOURCE_DEFAULT` | `"source"` / `"fineweb-edu"` | Provenance tag; default used if field absent. |
| `INCLUDE_TEXT` | `True` | Write the document text alongside its spans (self-contained output). Set `False` to store spans only (smaller; requires re-joining to source later). |
| `MAX_LENGTH` | `4096` | Max tokens per doc. Tokens beyond this are truncated, so **spans are only found in the first `MAX_LENGTH` tokens.** ModernBERT supports up to 8192 (higher memory). |
| `BATCH` | `32` | Docs per forward pass. Lower to 8–16 on a T4 if OOM; raise on an A100 for speed. |
| `ZSTD_LEVEL` | `10` | Output compression level. |
| `MAX_FILES` | `1` | **Trial limit** (Drive mode). Set to `None` for the whole corpus after validating. |
| `MAX_DOCS_PER_FILE` | `None` | Cap docs per file for testing (Drive mode). |
| `MAX_DOCS` / `SHARD_DOCS` | `2000` / `50000` | HF-mode docs per run and docs per output shard. |

---

## 7. How to tell it is working (correctness invariants)

- **Cell 10 demo** prints spans for the Eiffel Tower sentence — expect facts like `1889`, `330
  metres`, `Paris`.
- **Cell 16** prints `... 0 offset mismatches`. Any non-zero value means span offsets are wrong
  (almost always the whitespace-trim bug — see below).
- **Sane span counts.** On FineWeb-Edu-like prose the tagger averages ~14 spans per document and
  masks/tags roughly **8–9% of tokens** (measured: 8.7% vs the frontier annotator's 9.2%). If it
  tags ~0% or ~50% of tokens, something is wrong (wrong `TEXT_FIELD`, wrong model, or label
  mismatch).
- **Spans look like facts** — names, dates, numbers, places, citations — not random words.

### Known-good validation (performed by the author, off-Colab)

The notebook's actual `read_records` and `tag_batch` code was executed against real seed data and
the real checkpoint on CPU: **40 documents → 563 spans, 0 offset/faithfulness errors**, output shard
written and re-verified. So the tagging, decoding, output, and verify logic are confirmed correct;
only the Colab-specific Drive calls (auth/list/download/mount) are unavoidably untested outside
Colab and are standard API usage.

Reference quality of the model itself (held-out 486-doc eval, not on this corpus): span
**precision 0.63 / recall 0.60 / F1 0.61**, token accuracy 0.95 (exact-match; overlap/soft F1 ≈
0.76). These are the numbers to expect; a drastically different masking rate on new data suggests a
config or model problem rather than genuine domain shift.

---

## 8. Troubleshooting (most likely → least likely)

### A. "0 docs / 0 spans" or empty output shards

Two different causes; the `keys seen:` list printed next to each skipped file tells them apart.

**A1 — wrong `TEXT_FIELD`.** The keys look like a document record (`content`, `raw_text`, `body`,
…) but not `text`, so `rec.get(TEXT_FIELD)` is `None` and every record is silently skipped.
**Fix:** set `TEXT_FIELD` to the real name. For parquet/csv it is the column name.

**A2 — the folder holds a *pre-tokenized* corpus, so there is no raw text in it at all.** The
symptom is that the only readable files are sidecars and their keys are corpus metadata rather
than document fields:

```
subsets.json  keys seen: ['parent_corpus', 'subsets']
meta.json     keys seen: ['dataset', 'hf_name', 'hf_path', 'integrity', 'num_docs', ...]
manifest.json keys seen: ['num_loss_tokens', 'num_sequences', 'seq_len']
```

The documents live in `.u32le.bin` token shards (listed with an `x` by cell 12), which this
notebook cannot read and which cannot be tagged anyway — the tagger needs characters, not token
ids. **No value of `TEXT_FIELD` fixes this.** Two ways out:

1. Point `INPUT_FOLDER_ID` at a folder that actually holds raw text shards (e.g. the `edullm`
   root with `INPUT_PATH_PREFIX = "fineweb-edu-1b-smollm2-raw/"`). Cell 12 walks subfolders and
   prints the tree grouped by folder, so run it and look for `.jsonl(.gz|.zst)` /
   `.parquet` files.
2. Let the **HF fallback** run: `meta.json` names the upstream corpus (`hf_path` / `hf_name`),
   and with `INPUT_MODE = "auto"` the notebook streams that dataset when Drive yields nothing.
   Force it with `INPUT_MODE = "hf"`, or set `HF_PATH` explicitly.

### B. Drive API permission error when listing/downloading (cell 12/14)
**Cause:** the folder is not actually shared with the account you authenticated, or you authorized a
different Google account. **Fix:** confirm the folder is shared (viewer) with the exact account used
in `auth.authenticate_user()`; re-run cell 5 and pick the right account. Accessing by ID requires
*some* access; it is not a public-link download.

### C. Model fails to load (cell 8)
- `KeyError`/unknown model type `modernbert` → transformers too old. Ensure
  `transformers>=4.48` actually installed (restart runtime after the pip cell if needed).
- Errors mentioning `flash_attention_2` / attention → the notebook already falls back to `sdpa`/
  `eager`; if a custom edit forced flash-attn, remove it.
- `torch_dtype`/`dtype` `TypeError` from `from_pretrained` → do not pass a dtype kwarg to
  `from_pretrained`; keep the `model.to(dtype=...)` pattern.
- `config.json` not found → `MODEL_DIR` is wrong; let the auto-search run or set the path.

### D. Wrong / shifted spans, or cell 16 reports offset mismatches
**Cause:** the whitespace-trim loop in `tag_batch` was removed or altered, or a **slow tokenizer**
loaded (no offset mapping). **Fix:** restore the trim loop; ensure `tokenizer.json` exists so a fast
tokenizer is used (`AutoTokenizer.from_pretrained(..., use_fast=True)` is the default).

### E. CUDA out of memory (cell 14)
**Cause:** `BATCH` too high for the GPU, especially on long documents near `MAX_LENGTH`. **Fix:**
lower `BATCH` (T4: 8–16), or lower `MAX_LENGTH` if the corpus doesn't need 4096-token context.

### F. Session disconnects mid-run
**Expected on long runs / T4.** Just re-run the whole notebook — the manifest skips finished files.
Partial progress within the currently-processing file is lost (files are the resume unit), so
smaller input files give finer-grained resumability.

### G. Very slow
Confirm a GPU is actually attached (cell 2). CPU tagging is 1–2 orders of magnitude slower.
Throughput guide: ~100k+ tokens/s on A100 (~2–4 h per 1B tokens), ~30k tokens/s on T4 (~9–14 h per
1B tokens). Read the `docs/s` print to get the real rate for your hardware.

### H. Documents look truncated (spans stop partway through long docs)
`MAX_LENGTH=4096` truncates longer documents; spans are only found in the first 4096 tokens. Raise
`MAX_LENGTH` (≤8192 for ModernBERT) at higher memory cost, or pre-chunk long documents.

---

## 9. How to validate a fix

1. Set `MAX_FILES = 1` (and optionally `MAX_DOCS_PER_FILE = 200`).
2. Run all cells. Check: model loads on GPU (cell 8); the cell-10 demo prints fact-like spans;
   cell 12 lists the expected files; cell 14 reports non-zero docs and a plausible spans/doc (~10–20
   on educational/encyclopedic text); cell 16 reports **0 offset mismatches**.
3. Only then set `MAX_FILES = None` for the full corpus.

---

## 10. Scope / non-goals

- **Tagger-only.** Produces fact *spans*, not questions or answers. It is the "what to mask" half of
  Co-LMLM annotation.
- **Does not** produce the Co-LMLM training stream (inserting `<FACT> ... </FACT>` markers) — that is
  a downstream conversion step.
- **Does not** train anything, build a retrieval index, or run the Co-LMLM model.

---

## 11. Provenance of the model (context for judging output quality)

The checkpoint is `ModernBertForTokenClassification` (ModernBERT-large, 395M, 28 layers, hidden
1024), 3-class BIO (`O`, `B-FACT`, `I-FACT`), trained per the Co-LMLM paper's Appendix A.2 / Table 7:
AdamW (β=0.9/0.95, ε=1e-8), grad-clip 1.0, peak LR 2e-4 → final 2e-5 (cosine-with-min-lr), warmup
0.05, weight decay 1e-3, 4 epochs, effective batch 32, `max_length` 4096, bf16.

**Important data caveat:** this model was trained on the **FineWeb-Edu** half of the seed set only
(45.5k docs), not the Wikipedia half. The paper trained on both (15k Wikipedia + 45k FineWeb-Edu).
Recall on non-FineWeb-Edu-style text may therefore be lower than the paper's tagger. If output
quality is disappointing on this corpus, the most likely remedy is retraining with Wikipedia added,
not a change to this notebook.

---

## 12. References

- Co-LMLM paper: https://arxiv.org/abs/2607.07707
- Seed annotations (schema, faithfulness definition): https://github.com/prestonloats/co-lmlm-seed-annotations
- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-large
