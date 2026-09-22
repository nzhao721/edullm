# Benchmark contamination audit (matched-span, length-robust)

This directory holds the code and results behind the domain-weighting
paper's contamination audit of the shared 127B-token data reservoir that
every mixture in this experiment (and in `../../skillit/`) draws from.
Contamination is a property of the reservoir's domains, not of any one
mixture, so this directory's scan and its results are shared by both the
static-mixture (MixLaw / LightGBM) arms here and the dynamic-reweighting
(Skill-It) arms in `../../skillit/contamination/`, which reads this directory's
`results_olmo127b-reservoir.json` directly rather than duplicating the scan.

## Methodology

- **Unit of overlap: a length-floored n-gram.** An eval item's `stem` or
  `gold` field is indexed at `n = 13` only if it has >= 13 words; an 8-12
  word field is indexed as one exact whole-string match; a field under 8
  words is **NOT ASSESSABLE** and excluded from every rate (counted, never
  scored as clean).
- **Index text comes from the evaluator's own scored strings.**
  `dump_eval_items.py` reads `request["request"]["context"]` /
  `["continuation"]` off ai2-olmo's own `OEEvalTask` objects -- the exact
  strings the model is conditioned on and scored on -- rather than
  re-deriving them from HuggingFace dataset rows.
- **MMLU's per-subject few-shot exemplars are stripped at the index.**
  `strip_exemplars` (in `item_identity.py`) removes the label-wide prompt
  header, then strips each item's own per-subject exemplar block via a
  sorted-neighbor sliding window, since MMLU draws its five exemplars per
  *subject* rather than per label.
- **Matched-span word rate as the primary length-robust metric**, alongside
  the length-biased matched-*document* rate and a macro item-fraction rate
  averaged across the 10 distinct benchmarks.
- **Provenance-tagged, loose index and silent-failure gates.** Every scan
  asserts a normalizer fingerprint match, a canary document's exact
  post-normalization form, and full recovery of synthetic "spike" documents
  before it emits a single real hit. See `scan_corpus.py`'s module docstring
  for the failure modes each gate closes.

Full per-file detail is in each file's own docstring.

## Corpus scanned

Denominator: **40,582 evaluation items** (`eval_items_summary.json` ->
`total_rows`; `item_index_summary.json` -> `n_items`), the same 20
(task, split) labels of the OLMo-ladder RC suite used across every P1
experiment (`TASK_LOSS_RAW_LABELS` in `dump_eval_items.py`).

The reservoir: `pretrain/olmo-127b` (`data_source` in
`validation_mixtures_10b.json` below), the corpus every mixture in this
experiment is a domain-stratified re-weighting of. Scanned at
`olmo127b-edullm-publish-20260730T233445Z/publish-stage/text/<domain>/` on
FarmShare scratch (`/scratch/users/nzhao2/`), 360 `.json.gz` shards across
the 7 domains (dclm 124, arxiv 68, starcoder 20, pes2o 52, open-web-math 41,
algebraic-stack 39, wiki 16), 115 GB on disk.

## Results

### Per-domain, full reservoir

From `results_olmo127b-reservoir.json` -> `domains`. 45,058,504 documents,
60,136,262,363 words scanned in total; 30,044 matched documents, 946,670
matched words, 5,244 distinct items matched (any field) out of 40,582.

| Domain | Docs scanned | Words scanned | Matched-doc rate | Matched-span word rate |
| --- | --- | --- | --- | --- |
| `dclm` | 22,675,551 | 22,300,405,425 | 7.367e-04 | 1.965e-05 |
| `arxiv` | 3,950,480 | 11,432,655,539 | 8.455e-04 | 1.034e-05 |
| `starcoder` | 4,559,744 | 2,261,421,283 | 2.741e-05 | 2.043e-06 |
| `pes2o` | 1,977,195 | 8,869,717,747 | 2.332e-04 | 1.268e-06 |
| `open-web-math` | 2,892,814 | 6,577,346,314 | 7.021e-04 | 1.037e-05 |
| `algebraic-stack` | 2,831,500 | 6,014,118,124 | 4.870e-04 | 8.548e-06 |
| `wiki` | 6,171,220 | 2,680,597,931 | 9.727e-04 | 9.504e-05 |

`wiki` carries the highest matched-span word rate and `pes2o` the lowest --
a **74.95x spread** (`results_olmo127b-reservoir.json` ->
`span_rate_spread`) across domains this mixture experiment directly
reweights. This is the reason a mixture's contaminated exposure is not a
constant across arms: whichever mixture puts more weight on `wiki` and less
on `pes2o`/`starcoder` trains on more contaminated text by construction,
independent of anything the mixture-selection method itself is doing.

### Per-arm exposure

`../validation_mixtures_10b.json` lists the four arms validated at 370M,
the same four named in `../README.md`'s "Arms actually run (370M)" table,
matched here to their `run_name` by their published weight vectors in
"Validated mixture weights (370M)": `olmo-mix-1124` = OLMo Mix 1124
(control), `mix01` = Data Mixing Laws paper, `ML-pilot_caps` = MixLaw,
`LGB-min1pct` = LightGBM. `exposure_by_arm.json` covers exactly these four
(see Reproducing below).

From `exposure_by_arm.json`, baseline `olmo-mix-1124` (tag `natural`, the
published Dolma/OLMo-mix domain proportions with no re-weighting).

| Arm | Paper name | Matched-span word rate | vs. baseline | Matched-doc rate | vs. baseline |
| --- | --- | --- | --- | --- | --- |
| `mix01` | Data Mixing Laws paper | 1.303e-05 | 0.687x | 6.031e-04 | 0.845x |
| `LGB-min1pct` | LightGBM | 1.494e-05 | 0.788x | 6.545e-04 | 0.917x |
| `olmo-mix-1124` | OLMo Mix 1124 (control) | 1.895e-05 | 1.000x | 7.139e-04 | 1.000x |
| `ML-pilot_caps` | MixLaw | 4.016e-05 | 2.119x | 7.576e-04 | 1.061x |

Span-rate spread across these four arms: **3.08x** (lowest `mix01`,
highest `ML-pilot_caps`; `exposure_by_arm.json` ->
`span_rate_spread_across_arms`). MixLaw's exposure is more than double the
control's (2.12x) while LightGBM's is close to or below it (0.79x), yet the
paper reports both beating the control by a comparable margin (1.6057 and
1.6080 fitted-final bpb respectively) -- so a contamination-driven
explanation for the win would have to apply very differently to the two
search methods, which is weak evidence for one. The Data Mixing Laws paper
mixture has the lowest exposure of any arm (0.69x control) and is the only
one of the three non-control arms that does not beat the control, which
argues against lower exposure conferring an advantage either.

## Code map

Index construction

- `item_identity.py` -- shared normalizer, exemplar-stripping, hashing.
  Stdlib only.
- `dump_eval_items.py` -- reads eval item text off ai2-olmo's own task
  objects; writes `eval_items.jsonl.gz`. **Requires ai2-olmo + torch**; see
  Dependencies below.
- `build_item_index.py` -- builds the length-floored, provenance-tagged
  n-gram index from `eval_items.jsonl.gz`. Stdlib only.
- `verify_item_identity.py` -- asserts a training run scored the same item
  set this index was built from. Stdlib only; not needed to reproduce the
  numbers above.

Scanning, aggregation, exposure

- `scan_corpus.py` -- scans one corpus domain against the index. Stdlib
  only.
- `aggregate_by_domain.py` -- reduces one corpus's per-domain hit files and
  `DONE` sentinels into `results_olmo127b-reservoir.json`. Stdlib only.
- `mixture_exposure.py` -- projects the per-domain rates onto each arm's
  mixture weights to produce `exposure_by_arm.json`. Stdlib only; reads
  `validation_mixtures_10b.json`'s schema directly.
- `scan_array.sbatch` -- Slurm array template, one task per line of a
  task-list TSV (`domain\tshard\tpath`).

## Dependencies and what's not vendored

Everything above is **stdlib-only** except `dump_eval_items.py`, which needs
`torch` and `ai2-olmo` (`olmo.config`, `olmo.eval`, `olmo.tokenizer`)
importable. Its output, `eval_items.jsonl.gz`, is committed here so
everything downstream is runnable with the stdlib alone, without ai2-olmo,
from that file.

Not vendored: the raw 127B reservoir itself (115 GB; FarmShare path
above); `item_index.pkl` (172 MB, larger than GitHub's per-file limit --
regenerates from the committed `eval_items.jsonl.gz` via
`build_item_index.py`); and raw per-shard hit files and `DONE` sentinels
(regenerable from `scan_corpus.py` against the committed index and the
reservoir).

## Reproducing

```bash
CONTAM=path/to/this/directory   # experiments/skill-dag/mixlaw/contamination

# Dump eval item text (needs ai2-olmo; produces the committed eval_items.jsonl.gz)
python "$CONTAM/dump_eval_items.py" \
  --out "$CONTAM/eval_items.jsonl.gz" --summary "$CONTAM/eval_items_summary.json"

# Build the index (stdlib only; not committed, see Dependencies above)
python "$CONTAM/build_item_index.py" \
  --items "$CONTAM/eval_items.jsonl.gz" \
  --out /path/to/item_index.pkl --summary "$CONTAM/item_index_summary.json"

# One task-list TSV: one line per domain shard file.
RESERVOIR=/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/publish-stage/text
: > reservoir.tsv
for d in dclm arxiv starcoder pes2o open-web-math algebraic-stack wiki; do
  i=0
  for f in "$RESERVOIR/$d"/*.json*.gz; do
    printf '%s\t%04d\t%s\n' "$d" "$i" "$f" >> reservoir.tsv
    i=$((i + 1))
  done
done

# Scan (Slurm array, one task per task-list line; 360 lines here).
N=$(wc -l < reservoir.tsv)
CN_TASKLIST="$PWD/reservoir.tsv" CN_INDEX=/path/to/item_index.pkl \
  CN_OUT=/scratch/users/nzhao2/agent-runs/<run>/hits \
  sbatch --array="0-$((N - 1))" "$CONTAM/scan_array.sbatch"

# Aggregate once the array completes.
python "$CONTAM/aggregate_by_domain.py" \
  --hits /scratch/.../hits --items "$CONTAM/eval_items.jsonl.gz" \
  --domains dclm,arxiv,starcoder,pes2o,open-web-math,algebraic-stack,wiki \
  --corpus-name "olmo127b-edullm-publish full reservoir" \
  --out "$CONTAM/results_olmo127b-reservoir.json"

# Project onto the four arms actually trained at 370M (see Per-arm exposure
# above for why this excludes the other candidates validation_mixtures_10b.json
# catalogs).
python "$CONTAM/mixture_exposure.py" \
  --per-domain "$CONTAM/results_olmo127b-reservoir.json" \
  --mixtures "$CONTAM/../validation_mixtures_10b.json" \
  --baseline natural \
  --include olmo-mix-1124,mix01,ML-pilot_caps,LGB-min1pct \
  --out "$CONTAM/exposure_by_arm.json"
```
