# Plan: Reconstruct a Representative 1B‑Token Span‑Marked Corpus from Co‑LMLM

**Goal:** Produce a ~1‑billion‑token corpus of FineWeb‑Edu documents with the **Co‑LMLM‑released fact spans marked inline** (`<FACT>span</FACT>`), as **Parquet**, using **AWS CPU instances only**. The 1B slice must be a **fair, unbiased representative** of the full 100B corpus (no skew toward fact‑dense or long documents).

**Status:** Plan / not yet executed. Figures marked *(measured)* come from a local verification run; the rest are engineering estimates.

---

## 1. Executive summary

- Co‑LMLM's FineWeb corpus is **confirmed to be the public `HuggingFaceFW/fineweb-edu` `sample-100BT`** *(measured: doc‑id overlap 32,175 vs. 30,830 expected if same set, ~14× an independent draw)*. So the document text is already public; the release supplies the **fact spans** and their **document + order**.
- The reconstruction is a **CPU‑only data‑engineering job**: two filtered passes (one over the released `entries.db`, one over `sample-100BT`), a join on document id, and in‑document string placement of the spans. **No GPUs are required** (see §7).
- **Selection is uniform‑random by a hash of the document id (~1% of docs)** — representative and unbiased by construction; it does *not* prefer high‑fact‑density docs.
- **Hardware:** 1 × `i4i.4xlarge` (16 vCPU Intel Ice Lake, 3.75 TB NVMe) — or `i4i.8xlarge` for speed. **0 GPUs.**
- **Estimated cost:** ~**$3–8 on‑demand** (~$2–4 spot). **Estimated wall‑clock:** ~**1.5–2.5 h**, dominated by the one‑time download of the 712 GB `entries.db`.
- **Output:** ~5 GB Parquet (~1B tokens, ~1M docs) with inline `<FACT>…</FACT>` markers.

| Quick fact | Value |
|---|---|
| Source corpus | FineWeb‑Edu `sample-100BT` (confirmed) |
| Released spans store (`entries.db`) | **712.7 GB**, 2,174,081,398 entries |
| — FineWeb facts / Wikipedia facts | ~1.94B / ~236M |
| Fact density (FineWeb) | ~1 fact / ~52 tokens |
| `sample-100BT` | 140 Parquet files, ~450 GB, ~101.6M docs |
| Target slice | ~1B tokens ≈ **~1M docs / ~19.4M facts** |
| Output | ~5 GB Parquet, inline `<FACT>span</FACT>` |
| GPUs | **0** |

---

## 2. Objective & scope

### In scope
- A **span‑marked** corpus: original document text with each released fact span wrapped as `<FACT>span</FACT>` at its position, in `fact_idx` order.
- **~1B tokens**, a **fair representative** ~1% subsample of the 100B corpus.
- **Parquet** output, inline markers.
- **AWS, CPU‑only.**

### Out of scope (and why)
- **Questions / answers** (`<FACT q="…" a="…">`): the released retrieval index is *context‑keyed* and stores only the verbatim span — **`q`/`a` are in no released artifact**. Reproducing them requires re‑running the (unreleased) annotator models = a GPU stage (see Appendix C).
- **Model training / embedding / index building**: not requested (Appendix C sketches it if scope later grows).
- **Zero‑fact documents**: kept as plain text if selected, but they carry no spans.

---

## 3. Confirmed facts & assumptions

All items below were verified locally except where noted "assumption."

1. **Corpus identity** *(measured)*: release FineWeb docs = `sample-100BT`. So we obtain document text directly from `sample-100BT` and do **not** need to scan the full 1.3T FineWeb‑Edu.
2. **`entries.db`** (bucket `co-lmlm-360m-fw-fineweb-wiki-index`, file `fineweb_with_fullwiki_entries.db`, **712.7 GB**): SQLite, table `entries(entry_id TEXT PRIMARY KEY, data TEXT)`, where `data` is JSON `{"id": …, "fact_span": …, "metadata": {"sample_id": <doc id>, "fact_idx": <int>}}` *(from the release code; the `.db` is the authoritative build output)*.
3. **`entry_id` format** *(measured)*: `"<urn:uuid:…>_factN"` for FineWeb; `"<num>_<num>_factN"` for Wikipedia. Document id = substring **before the first `_`** = the FineWeb‑Edu native `id` (`<urn:uuid:…>`), which joins directly to `sample-100BT.id`.
4. **Index composition** *(measured)*: 2,174,081,398 entries; FineWeb occupies the low `faiss_id` range (~1.94B facts), Wikipedia the top (~236M).
5. **`sample-100BT`** *(measured)*: 140 Parquet files (~2.15 GB each), ~726K docs/file (~101.6M docs). Columns: `text, id, dump, url, file_path, language, language_score, token_count, score, int_score`. The `token_count` column lets us calibrate to exactly ~1B tokens and QA representativeness cheaply.
6. **Fact spans are verbatim substrings** of the source document (guaranteed by the annotation design) — so placement is exact substring matching.
7. **Access constraints** *(measured)*: HF Storage Buckets serve **whole files only** (no HTTP byte‑range); anonymous access is **rate‑limited (HTTP 429)**. Plan around this (whole‑file, sequential, backoff, cache in S3).
8. **Assumption:** `metadata.sample_id` equals the `entry_id` prefix (native `id`); we verify on the first 100 rows before the full run. Fallback: derive doc id from `entry_id` via `split('_',1)[0]` (already validated).

---

## 4. Data inventory

| Artifact | Source | Size | Needed? | Why |
|---|---|---|---|---|
| `fineweb_with_fullwiki_entries.db` | HF bucket (xet) | **712.7 GB** | **Yes** | only source of fact‑span **text** + `fact_idx` + doc id |
| `sample-100BT/*.parquet` (140) | HF dataset | **~450 GB** | **Yes** | document **text** + `token_count` |
| `faiss_id_to_entry_id.db` | HF bucket | 133.7 GB | No | id‑map only (no span text); used earlier only for corpus‑identity test |
| `faiss.index` | HF bucket | 228 GB | No | vector index; irrelevant to text reconstruction |
| **Output corpus** | produced | **~5 GB** | — | deliverable |

Total ingress ≈ **~1.16 TB** (one‑time, free into AWS). Working disk needed ≈ **~800 GB–1 TB** (holds `entries.db` + output + temp; `sample-100BT` is streamed/filtered).

---

## 5. Representative sampling methodology (unbiased ~1B)

**Requirement:** the 1B slice must mirror the full 100B corpus — same distribution of document lengths, fact density, crawls/sources — with **no bias** toward fact‑dense or long documents.

**Method — uniform random by document‑id hash (~1%):**

- Select a document **iff** `md5(doc_id) mod 100 == 0`. This picks each document with probability 1/100, **independent of its length or fact count** → the subsample is an unbiased 1% of the corpus; aggregate statistics match the full set in expectation.
- 1% of ~100B tokens ≈ **~1B tokens**; 1% of ~101.6M docs ≈ **~1.0M docs**; 1% of ~1.94B FineWeb facts ≈ **~19.4M facts**.
- **Cross‑tool‑consistent hash:** use MD5 on the raw `doc_id` string on *both* sides — DuckDB `md5(id)` and Python `hashlib.md5(doc_id.encode())`. Because both passes apply the identical predicate, they independently yield the **same** document set — no scattered‑id lookups needed.
- **Deterministic** (bonus reproducibility): anyone re‑running the hash gets the same slice.

**Why not the cheap alternatives (rejected to avoid bias):**
- *Contiguous `faiss_id`/shard range* — cheapest (reads ~1%), but `faiss_id` order ≈ crawl/shard order, so a contiguous block over‑represents particular crawls/topics. **Rejected** (biased).
- *Selecting by fact count / density* — explicitly what the user wants to avoid. **Rejected.**

**Calibration to exactly ~1B:** sum `token_count` over selected docs (cheap, from `sample-100BT`). If the 1% sum deviates, adjust the modulus (e.g., `mod 101`) or trim/extend by whole documents. Target 1.00B ± 2%.

**QA (see §12):** compare sampled vs. full distributions of `token_count`, facts‑per‑doc, and `dump` (crawl) to confirm no skew.

---

## 6. Architecture overview

```
                 (free ingress, one-time, whole-file + 429 backoff)
   HF bucket/dataset ────────────────────────────────────────────► S3 (region-local cache)  [optional]
        │                                                                    │
        │  entries.db 712.7 GB                                               │  entries.db, sample-100BT
        │  sample-100BT ~450 GB                                              ▼
        └───────────────────────────────►  EC2  i4i.4xlarge (CPU, NVMe)  ◄──┘
                                             │
             Pass A: scan entries.db  ──►  filter md5(doc_id)%100==0 & urn:uuid  ──►  spans(doc_id, fact_idx, span)  ~19.4M
             Pass B: scan sample-100BT ─►  filter md5(id)%100==0                 ──►  docs(id, text, token_count)   ~1.0M
                                             │
                              Join on doc_id + in-doc span placement (verbatim, fact_idx order)
                                             │
                                             ▼
                                  Output: ~5 GB Parquet (inline <FACT>…</FACT>)  ──►  S3
```

**S3 caching is optional but recommended** if you might iterate: HF→S3 once (free), then all compute reads in‑region (fast, free, no 429). For a strict one‑shot, you can pull HF→NVMe directly and skip S3.

---

## 7. Hardware (compute, count, and the GPU question)

### GPUs: **0 (none).**
The entire job is **string matching + hash filtering + a join** — I/O‑ and CPU‑bound, embarrassingly parallel across CPU cores. There is **no neural/model step**, so no GPU is used or needed. (This is the direct consequence of the "CPU‑only, span‑marked, no q/a" scope. If scope later expands to regenerate questions or train a model, that adds GPUs — see Appendix C.)

### CPU instances

| Role | Instance | CPU | vCPU / RAM | Local NVMe | Network | Count | ~$/hr (on‑dem, us‑east‑1) |
|---|---|---|---|---|---|---|---|
| **Primary (recommended)** | `i4i.4xlarge` | Intel Xeon 8375C "Ice Lake" (≤3.5 GHz) | 16 / 128 GiB | 3,750 GB | up to 25 Gbps | **1** | ~$1.37 |
| Faster option | `i4i.8xlarge` | same | 32 / 256 GiB | 7,500 GB | up to 37.5 Gbps | 1 | ~$2.75 |
| Budget (tight disk) | `i4i.2xlarge` | same | 8 / 64 GiB | 1,875 GB | up to 12.5 Gbps | 1 | ~$0.69 |
| Cheaper ARM alt | `im4gn.4xlarge` | AWS Graviton2 | 16 / 64 GiB | 3,750 GB | up to 25 Gbps | 1 | ~$1.16 |

**Rationale:** this is a *storage‑optimized* job — you're really buying **fast local NVMe** to hold the 712 GB SQLite and read it quickly, not raw compute. `i4i` (Nitro SSD) gives GB/s + high IOPS at no extra storage cost. 16 vCPU parallelizes the `entries.db` scan and the `sample-100BT` DuckDB pass. A single instance suffices; spot pricing (~65–70% off) makes it cheap. Use `im4gn` (Graviton) to shave ~15%.

**Distributed alternative (only if you want it faster or will run repeatedly):** convert `entries.db` → partitioned Parquet in S3 once, then run the passes on a small **EMR/Ray** cluster (e.g., 4× `c7i.4xlarge`). Similar cost, ~2–3× faster wall‑clock, and future slices become cheap S3/Athena queries. Not necessary for a single 1B slice.

---

## 8. Implementation plan (step by step)

### Phase 0 — Prerequisites
- AWS account, an S3 bucket in the compute region, an IAM role for the EC2 instance with S3 read/write.
- Launch 1 × `i4i.4xlarge` (Amazon Linux 2023), mount the NVMe instance store at `/data`.
- Install tooling:

```bash
sudo dnf install -y python3.12 sqlite
python3.12 -m pip install duckdb pyarrow "huggingface_hub[hf_xet]" boto3 xxhash
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # anonymous; avoids stale-token 401s
```

### Phase 1 — Ingest to fast storage (one‑time, ~1–1.5 h)
Pull whole files with sequential downloads and 429 backoff. Land on NVMe (`/data`) for the one‑shot; optionally also mirror to S3 for reuse.

```bash
# entries.db (712.7 GB) — the only span-text source
hf buckets cp -q \
  "hf://buckets/lil-lab/co-lmlm-360m-fw-fineweb-wiki-index/fineweb_with_fullwiki_entries.db" \
  /data/entries.db

# sample-100BT (140 parquet files) — download whole files, backoff on 429
python3.12 fetch_sample100bt.py --out /data/s100 --retries 10 --backoff 20
```

- `fetch_sample100bt.py`: list files via the HF tree API, download each whole file sequentially (`urllib`/`hf`), exponential backoff on 429. (Whole‑file GETs avoid the slow, rate‑limited range reads.)
- **Optional S3 cache:** `aws s3 cp /data/entries.db s3://$BUCKET/colmlm/ ...` so re‑runs skip HF.

> Sanity check before the full run: `sqlite3 /data/entries.db "SELECT entry_id,data FROM entries LIMIT 3"` and confirm `data.metadata.sample_id` == `entry_id` prefix.

### Phase 2 — Pass B: select & extract documents (~5–15 min)
Filter `sample-100BT` to the 1% hash sample; keep text + token_count. DuckDB, column‑pruned, multi‑threaded.

```python
import duckdb
c = duckdb.connect("/data/work.duckdb"); c.execute("PRAGMA threads=16")
c.execute("""
CREATE TABLE docs AS
SELECT id AS doc_id, text, token_count, dump, url
FROM read_parquet('/data/s100/*.parquet')
WHERE (hash(id) % 100) = 0          -- or: (abs(hash(md5(id))) % 100)=0 to match Python md5
""")
print(c.execute("SELECT count(*), sum(token_count) FROM docs").fetchone())  # ~1.0M docs, ~1B tokens
```

> Use one hash definition consistently across Phases 2 & 3. Recommended: MD5 → first 8 hex → int → `% 100 == 0`, implemented identically in DuckDB (`md5(id)`) and Python.

### Phase 3 — Pass A: extract fact spans for selected docs (~15–30 min)
One parallel streaming pass over `entries.db`, keeping only FineWeb entries whose doc id hashes into the sample. Parallelize with N worker processes over `rowid` ranges (the table has an implicit rowid).

```python
# span_extract_worker.py  (run N in parallel over rowid ranges; merge outputs)
import sqlite3, json, hashlib, sys, pyarrow as pa, pyarrow.parquet as pq
lo, hi, shard = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
con = sqlite3.connect("/data/entries.db"); cur = con.cursor()
rows = []
for entry_id, data in cur.execute(
        "SELECT entry_id, data FROM entries WHERE rowid>=? AND rowid<?", (lo, hi)):
    if not entry_id.startswith("<urn:uuid:"):        # FineWeb only
        continue
    doc_id = entry_id.split("_", 1)[0]
    if int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16) % 100 != 0:
        continue
    d = json.loads(data)
    rows.append((doc_id, d["metadata"]["fact_idx"], d["fact_span"]))
pq.write_table(pa.table({"doc_id":[r[0] for r in rows],
                         "fact_idx":[r[1] for r in rows],
                         "span":[r[2] for r in rows]}), f"/data/spans_{shard}.parquet")
```

```bash
# fan out ~16 workers across the rowid space, then load all spans_*.parquet
python3.12 launch_span_extract.py --db /data/entries.db --workers 16
```

Result: ~19.4M rows `(doc_id, fact_idx, span)` for the selected ~1M docs.

### Phase 4 — Join + place markers (~5–10 min)
Group spans by `doc_id` (ordered by `fact_idx`), join to `docs`, and insert `<FACT>…</FACT>` by an in‑order greedy verbatim scan.

```python
def mark(text, spans):                 # spans: list ordered by fact_idx
    out, cur, placed = [], 0, 0
    for s in spans:
        i = text.find(s, cur)
        if i < 0:                      # unplaceable (rare: dup/edge) -> skip marker, log
            continue
        out.append(text[cur:i]); out.append("<FACT>"); out.append(s); out.append("</FACT>")
        cur = i + len(s); placed += 1
    out.append(text[cur:])
    return "".join(out), placed
```

Driver: DuckDB join `docs ⋈ spans` → per‑doc `list(span ORDER BY fact_idx)` → apply `mark` (parallel via `map`/Ray/multiprocessing) → emit rows.

### Phase 5 — Write output + QA (~2–5 min)
Write Parquet with schema:

| column | type | notes |
|---|---|---|
| `doc_id` | string | `<urn:uuid:…>` |
| `text_marked` | string | text with inline `<FACT>span</FACT>` |
| `token_count` | int64 | from `sample-100BT` |
| `n_facts` | int32 | spans for this doc |
| `n_facts_placed` | int32 | markers actually inserted |
| `dump`, `url` | string | provenance |

```python
c.execute("COPY marked TO 's3://$BUCKET/colmlm/corpus_1b/' (FORMAT PARQUET, PARTITION_BY (dump))")
```

Run QA (§12).

### Phase 6 — Teardown
Copy output + QA report to S3; delete NVMe scratch; **terminate the instance**; optionally delete S3 caches (lifecycle rule) if one‑shot.

---

## 9. Cost estimate

Region us‑east‑1, one‑shot, primary design (1 × `i4i.4xlarge`).

| Item | Basis | On‑demand | Spot |
|---|---|---|---|
| HF → AWS ingress (~1.16 TB) | inbound free | $0 | $0 |
| Compute (i4i.4xlarge, ~2.5 h) | $1.373/h | ~$3.4 | ~$1.2 |
| S3 storage (~1.2 TB, ~2 days, optional) | $0.023/GB‑mo | ~$1.8 | ~$1.8 |
| S3 ↔ EC2 (same region) | free | $0 | $0 |
| Output storage (5 GB) | negligible | ~$0 | ~$0 |
| Output egress (only if pulled out of AWS) | $0.09/GB × 5 | ~$0.45 | ~$0.45 |
| **Total** | | **~$3–8** | **~$2–4** |

**GPU cost: $0.** If you skip S3 (pull HF→NVMe directly) the storage line is $0 and total drops toward ~$2–4.

---

## 10. Time estimate

| Phase | Work | Est. wall‑clock |
|---|---|---|
| 1. Ingest | 712.7 GB + ~450 GB whole‑file (429 backoff) | **~1–1.5 h** (dominant) |
| 2. Docs pass | filter `sample-100BT` (450 GB), DuckDB | ~5–15 min |
| 3. Spans pass | parallel scan of `entries.db` (712 GB / 2.17B rows) | ~15–30 min |
| 4. Join + mark | ~1M docs / ~19.4M spans | ~5–10 min |
| 5. Write + QA | ~5 GB Parquet + checks | ~2–5 min |
| **Total** | | **~1.5–2.5 h** |

Ingest dominates. With an S3 cache already warm, a re‑run is ~30–60 min. Distributed variant: ~45–75 min end‑to‑end.

---

## 11. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| HF **429 rate‑limit** (anonymous) | ingest stalls | whole‑file sequential GETs + exponential backoff; cache in S3 so it's one‑time; a valid HF token raises limits |
| **No byte‑range** on buckets | must land full 712 GB | accept the floor; it's a one‑time download; read only what's needed afterward |
| `entries.db` scan speed | Phase 3 slow | NVMe instance + **parallel** native‑sqlite over `rowid` ranges; or pre‑reshape to Parquet + DuckDB |
| **Sampling bias** | non‑representative slice | uniform `md5(doc_id)%100` (length/density‑independent) + QA distribution checks (§12) |
| **Unplaceable spans** (dup/identical substrings) | a few markers wrong/missing | in‑order greedy `find` from a moving cursor; log `n_facts − n_facts_placed`; expect a small % |
| **Disk capacity** | run fails | size NVMe ≥ ~1 TB (i4i.4xlarge = 3.75 TB); stream `sample-100BT`, don't store all |
| Spot interruption | job killed | checkpoint per phase to S3; or use on‑demand for the ~2.5 h |
| `sample_id` format surprise | join breaks | verify on first 100 rows (Phase 1 sanity check); fallback `split('_',1)[0]` already validated |

---

## 12. QA & validation

1. **Token target:** `sum(token_count)` of selected docs is 1.00B ± 2% (adjust modulus/trim if not).
2. **Representativeness (no bias):** compare **sampled vs. full** distributions for
   - `token_count` (histogram / mean / p50 / p95),
   - facts‑per‑doc (`n_facts` distribution),
   - `dump` (crawl) share.
   The full‑corpus baselines are cheap to compute from `sample-100BT` metadata + a coarse `entries.db` aggregate. Sampled means should match full means within sampling error.
3. **Span fidelity:** report overall `sum(n_facts_placed)/sum(n_facts)`; manually eyeball 20 marked docs; confirm every `<FACT>…</FACT>` content is a verbatim substring.
4. **Join coverage:** every selected doc with ≥1 fact appears; note count of zero‑fact docs.

Emit a `qa_report.json` alongside the corpus.

---

## 13. Deliverables

- `s3://$BUCKET/colmlm/corpus_1b/` — ~5 GB Parquet (partitioned by `dump`), schema in Phase 5.
- `qa_report.json` — token total, representativeness stats, span‑placement rate.
- Scripts: `fetch_sample100bt.py`, `launch_span_extract.py`, `span_extract_worker.py`, `mark_and_write.py`.
- This document.

---

## Appendix A — Schemas

**`entries.db`** (`fineweb_with_fullwiki_entries.db`)
```sql
CREATE TABLE entries (entry_id TEXT PRIMARY KEY, data TEXT NOT NULL);
-- data JSON: {"id": "...", "fact_span": "...", "metadata": {"sample_id": "<urn:uuid:...>", "fact_idx": N}}
-- entry_id: "<urn:uuid:...>_factN"  (FineWeb)   |   "<n>_<m>_factN"  (Wikipedia)
```

**`sample-100BT`** parquet columns: `text, id, dump, url, file_path, language, language_score, token_count, score, int_score` (`id` = `<urn:uuid:…>`).

---

## Appendix B — Rejected alternative: brute‑force scan/match

Original framing ("scan the corpus and string‑match every released span"): build an Aho‑Corasick/Hyperscan automaton over the released spans and scan the corpus text. **Rejected** because (a) the released id‑map already gives each fact's document, so a **join** is far cheaper than multi‑pattern search; (b) short spans are non‑unique → billions of ambiguous matches; (c) at scale the automaton doesn't fit in memory. The join + in‑doc placement (Phases 3–4) is strictly better. Keep this only as a fallback if the `sample_id` linkage ever failed.

---

## Appendix C — Optional GPU stages (only if scope expands)

Not part of this plan, but for completeness (this is where GPUs *would* enter):

- **Regenerate questions/answers** (`<FACT q="…" a="…">`): the `q`/`a` are unreleased. You'd retrain the two annotators (ModernBERT‑large span tagger — spans already known so optional; **Qwen2.5‑1.5B question generator**) from a fresh Gemini‑seeded set, then run the question generator over the 1M selected docs. Hardware: ~1–4× `g5`/`g6` (NVIDIA A10G/L4) or 1× `p4d` (A100) for the seed + generation; ~a few GPU‑hours for 1M docs given KV‑cached document prefixes. Adds ~$50–300 and stochastic (non‑identical) questions.
- **Train a model on the 1B corpus**: e.g., SmolLM2‑135M/360M. Hardware: 1–8× A100/H100/B200 depending on token budget; cost/time set by the run, not this pipeline.

Both are additive; the CPU pipeline here produces their input.

---

## Appendix D — Decisions taken (from clarifying questions)

- Scope: **CPU‑only**, span‑marked corpus (no q/a). → 0 GPUs.
- Output: **Parquet**, inline `<FACT>span</FACT>`.
- Approach: **ID‑join + in‑document span placement** (not brute‑force scan).
- Slice: **fair representative ~1B**, uniform `md5(doc_id)%100` (no fact‑density/length bias).
- Deliverable: this Markdown document.
