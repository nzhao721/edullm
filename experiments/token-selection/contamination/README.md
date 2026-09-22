# Benchmark contamination audit (matched-span, length-robust)

This directory holds the code and results behind the paper's contamination
audit of the three corpora used in the token-selection experiment. It
**replaces** an earlier verbatim-13-gram audit that lived at this path: that
audit's own denominator (40,087 items) and per-corpus match counts (358 /
438 / 1,472) did not match the numbers this paper's text now reports (40,582
items; 957 / 1,010 / 2,432), and neither the scanner nor the eval-item index
that produced them had ever been committed here, so the discrepancy was not
reproducible from this repo. This directory now vendors the code that
produced the *current* numbers, so the join can be checked directly rather
than taken on faith. See [Superseded methodology](#superseded-methodology-not-vendored-here)
below for what changed and why the two audits' numbers are not comparable.

## Methodology

Full write-up: the "methodology v4" sections referenced in every file's
docstring below. Summary, in contrast with the superseded audit:

- **Unit of overlap: a length-floored n-gram, not a fixed 13-gram.** An item's
  `stem` or `gold` field is indexed at `n = 13` only if it has >= 13 words;
  an 8-12 word field is indexed as one exact whole-string match; a field
  under 8 words is **NOT ASSESSABLE** and excluded from every rate (counted,
  never scored as clean). A fixed 13-gram with no floor systematically misses
  short items and silently drops them into the "clean" bucket.
- **Index text comes from the evaluator's own scored strings, not
  reconstructed HuggingFace fields.** `dump_eval_items.py` reads
  `request["request"]["context"]` / `["continuation"]` off ai2-olmo's own
  `OEEvalTask` objects -- the exact strings the model is conditioned on and
  scored on -- rather than re-deriving them from HF dataset rows. This is not
  a cosmetic difference: ai2-olmo *assembles* what it scores (HellaSwag's
  context is activity label + `ctx_a` + capitalized `ctx_b`; WinoGrande's
  scored continuation turned out to be the sentence *suffix*, not the option
  word), so an index built from reconstructed fields is an index over text
  the evaluator never actually sees.
- **MMLU's per-subject few-shot exemplars are stripped at the index, not
  worked around after the fact.** `strip_exemplars` (in `item_identity.py`)
  removes the label-wide prompt header first, then strips each item's own
  per-subject exemplar block via a sorted-neighbor sliding window, since MMLU
  draws its five exemplars per *subject* rather than per label -- a
  label-wide common-prefix strip alone finds almost nothing there. Left
  unstripped, a single mirrored copy of a subject's dev examples matches
  every item in that subject and inflates the subject's rate roughly 5x (see
  [MMLU exemplar fix](#mmlu-exemplar-fix) below). Because this is fixed at
  the index rather than by excluding MMLU from a "self-contained" subset
  after scanning, MMLU's own rate is reported directly rather than omitted.
- **Matched-span word rate as the primary length-robust metric**, alongside
  the length-biased matched-*document* rate (a document is "matched" if any
  span in it hits) and a macro item-fraction rate averaged across benchmarks
  (so a 10,000-item benchmark and a 300-item benchmark get equal say, matching
  how the endpoint itself is weighted). `scan_corpus.py` unions all matched
  word spans per document so the span rate is computable directly rather than
  only the document-level upper bound.
- **Provenance-tagged, loose index.** Every indexed n-gram records which
  field (`stem` or `gold`) it came from, so nested nothing needs a second
  scan -- the stem-only, gold-only and "any field" rates are all read off one
  pass.
- **Silent-failure gates.** Every scan asserts a normalizer fingerprint
  match, a canary document's exact post-normalization form, and full recovery
  of synthetic "spike" documents carrying known item n-grams before it emits
  a single real hit. A scan that would otherwise report a suspiciously clean
  result because its normalizer diverged from the index's, or because a
  decompressor died partway through a shard, raises instead of finishing
  quietly. See `scan_corpus.py`'s module docstring for the failure modes each
  gate closes.

## Corpora scanned

Denominator: **40,582 evaluation items** (`eval_items_summary.json` ->
`total_rows`; `item_index_summary.json` -> `n_items`), the same 20
(task, split) labels of the OLMo-ladder RC suite the training runs in this
experiment evaluate `task_loss_bpb` on (`TASK_LOSS_RAW_LABELS` in
`dump_eval_items.py`). This is a corrected count from the same 20 labels the
superseded audit's 40,087 used -- see
[Superseded methodology](#superseded-methodology-not-vendored-here).

| Corpus | Role | Path scanned (FarmShare scratch) | Docs scanned | Words scanned |
| --- | --- | --- | --- | --- |
| `regmix-10b` | 10B training corpus (`pretrain/regmix-10b` v1) | `regmix-10b-20260725-124810/trim/<domain>/<domain>-trimmed.json.gz`, one file per domain | 4,748,990 | 5,879,719,994 |
| `hq-reference-v1` | HQ reference corpus (Reference A) | `hq-reference-v1/out/<domain>/`, one directory per domain | 3,367,856 | 2,298,753,521 |
| `refhq-new-v1` | Instruct reference corpus (Reference B) | `refhq-new-v1/out/<source>/<category>/documents/`, `category` is the aggregation domain | 6,193,748 | 2,738,073,602 |

All three share the domain set `{algebraic-stack, arxiv, dclm, open-web-math,
pes2o, starcoder, wiki}` except `refhq-new-v1`, which is organized by
`{chat, code, general, math, science}` instead (its 23 source/category
shards are listed in the reproduction commands below). Every path above is
relative to `/scratch/users/nzhao2/` on FarmShare -- the actual location the
runs read from, recorded here for the same reason the superseded audit's
README recorded its own working directory: so the exact input is on the
record rather than merely asserted. See
[Dependencies and what's not vendored](#dependencies-and-whats-not-vendored)
for why the raw corpora themselves are not (and cannot practically be)
committed to this repo.

## Results

From `results_regmix-10b.json`, `results_hq-reference-v1.json`,
`results_refhq-new-v1.json` -> `totals`. Percentages rounded to 2 decimal
places except where the source table needs more precision to show a
difference.

| Corpus | Matched items (any field) | Item rate | Self-contained-stem rate | Matched-span word rate |
| --- | --- | --- | --- | --- |
| `regmix-10b` (training) | 957 / 40,582 | **2.36%** | **2.00%** | 1.095e-05 |
| `hq-reference-v1` (HQ reference) | 1,010 / 40,582 | **2.49%** | **2.22%** | 2.643e-05 |
| `refhq-new-v1` (Instruct reference) | 2,432 / 40,582 | **5.99%** | **7.50%** | 2.195e-04 |

"Item rate" is `distinct_items_any / n_items` (union of the stem and gold
fields). "Self-contained-stem rate" is `macro_stem_rate`: the stem-field hit
rate per benchmark, averaged across the 10 distinct benchmarks (not the 20
task/split labels) so no single large benchmark dominates the average --
this is the rate to read as an estimate of standalone-question leakage, and
it is now computed the same way for every benchmark including MMLU (see
[Methodology](#methodology)).

Contamination in the training corpus is shared by every arm trained on it,
so it cannot explain a gap between arms. The Instruct reference corpus --
used as the frozen reference model for the RHO-1 and BLADE arms -- is
matched by roughly **2.5x** more items than the training corpus (2,432 vs
957). A reference model that has memorized more evaluation items would raise
excess loss on those items and bias its arm toward *outperforming* the
control; the paper's finding that RHO-1 and BLADE still underperform the
full-loss control holds despite that bias, not because contamination is
absent.

### `refhq-new-v1` (Instruct) per-benchmark, selected rows

From `results_refhq-new-v1.json` -> `totals.per_benchmark`.

| Benchmark | Stem found / assessable | Stem rate | Gold found / assessable | Gold rate |
| --- | --- | --- | --- | --- |
| `csqa` (val only, this benchmark has no test split here) | 396 / 1,161 | **34.11%** | -- (no assessable gold; CSQA's gold is a single letter, always under the 8-word floor) | -- |
| `mmlu` (pooled: 4 subject groups x val+test) | 728 / 13,827 | **5.27%** | 240 / 5,148 | **4.66%** |

### MMLU exemplar fix

`mmlu` above pools all four subject groups and both val/test splits (13,827
assessable stems). The superseded verbatim-13-gram audit reported its MMLU
figure differently -- `mmlu_val` alone (1,519 items, val split only) at
**27.3%** -- so the two numbers are not scoped identically and a single
"5x" is not a clean per-item ratio. The comparison is still informative in
direction and rough magnitude: this pipeline's pooled rate of **5.27%** is
far below 27.3%, and the gap is explained by *why* the old number was high,
not by the denominator difference. That audit's index was built from a
fixed-width 13-gram over MMLU's raw indexed text, which (per this pipeline's
`item_identity.strip_exemplars`, applied here and not there) still carried
each item's per-subject few-shot exemplars: a single mirrored dev block
matches every item in the subject, so the reported rate was dominated by
exemplar overlap rather than standalone-question leakage. `csqa`, which has
no per-subject exemplar structure to strip, moves much less between the two
audits (32.3% -> 34.1%) even though its own denominator also shifted
slightly (1,220 items in the superseded audit vs 1,161 *assessable* stems
here, out of 1,221 -- 60 CSQA stems fall under this pipeline's 8-word floor
and are excluded rather than scored). The residual movement there is
consistent with that item-count and floor correction rather than an
exemplar effect, since CSQA has no per-subject exemplar block to strip.

### Regmix and HQ-reference `csqa` / `mmlu`, for comparison

| Corpus | `csqa` stem rate | `mmlu` stem rate | `mmlu` gold rate |
| --- | --- | --- | --- |
| `regmix-10b` | 0.52% | 1.85% | 1.03% |
| `hq-reference-v1` | 0.09% | 1.89% | 1.22% |
| `refhq-new-v1` | 34.11% | 5.27% | 4.66% |

The Instruct corpus's `csqa` rate is nearly two orders of magnitude above the
training and HQ-reference corpora's (34.11% vs 0.52% / 0.09%); its `mmlu`
rate is smaller in relative terms but still roughly 3x theirs (5.27% vs
1.85% / 1.89%). CSQA and MMLU's dev/validation splits are both widely
redistributed inside public instruction-tuning mixtures, which is exactly
what this measures.

## Code map

Index construction (section 4a/4b of the methodology)

- `item_identity.py` -- the shared normalizer, exemplar-stripping and hashing
  logic. Stdlib only. Imported by every other file here.
- `dump_eval_items.py` -- reads the eval item text ai2-olmo actually scores
  off its own task objects; writes the full-text `eval_items.jsonl.gz` and a
  per-label summary, both committed here only in hash-only form (see
  [Committed eval-item dump is hash-only](#committed-eval-item-dump-is-hash-only)).
  **Requires ai2-olmo + torch**; see
  [Dependencies](#dependencies-and-whats-not-vendored).
- `build_item_index.py` -- builds the length-floored, provenance-tagged n-gram
  index from a full-text `eval_items.jsonl.gz`. Stdlib only. Refuses the
  committed hash-only file with an explicit error rather than a `KeyError`.
- `verify_item_identity.py` -- asserts a training run scored the same item
  set this index was built from (doc_id digests + sampled content hashes).
  Stdlib only; not needed to reproduce the numbers in this README, included
  for the fidelity claim that the scan indexes what training actually scores.

Scanning and aggregation (sections 3, 5, 6)

- `scan_corpus.py` -- scans one corpus domain (a file or a directory of
  shard files) against the index; one process per domain or per-domain
  shard. Stdlib only (reads `.json.gz` / `.jsonl.gz` directly; shells out to
  the `zstd` CLI for `.zst`/`.zstd` shards, since no corpus scanned for this
  paper needed that path but the reader supports it for corpora that do).
- `aggregate_by_domain.py` -- reduces one corpus's per-domain hit files and
  `DONE` sentinels into the `results_*.json` tables above. Stdlib only.
- `scan_array.sbatch` -- Slurm array template, one task per line of a
  task-list TSV (`domain\tshard\tpath`). See "Reproducing" below for how the
  task list was built for each corpus.

## Dependencies and what's not vendored

Everything above is **stdlib-only** except `dump_eval_items.py`, which needs
`torch` and `ai2-olmo` (`olmo.config`, `olmo.eval`, `olmo.tokenizer`)
importable -- the same training environment the runs in this experiment
used. That dependency is real and disclosed, not a hidden path: reading eval
item text from ai2-olmo's own task objects rather than reconstructing it from
HuggingFace fields is the fix this methodology makes (see Methodology
above). Its output, `eval_items.jsonl.gz`, **is committed in this directory
in hash-only form** (see the next section): `verify_item_identity.py` and
`aggregate_by_domain.py` run from the committed file with the stdlib alone,
but rebuilding the n-gram index needs the item text, and so needs a full-text
dump regenerated with ai2-olmo.

Not vendored, and why:

- **The raw corpora** (`regmix-10b`, `hq-reference-v1`, `refhq-new-v1`) --
  tens of billions of words each, and not something a paper's code repo
  should carry. Their FarmShare paths are recorded above for provenance.
- **`item_index.pkl`** (172 MB; see `item_index_summary.json` ->
  `index_bytes_on_disk`) -- larger than GitHub's per-file limit, and it
  regenerates deterministically via `build_item_index.py`, in well under a
  minute, from a regenerated full-text dump.
- **Raw per-shard hit files and `DONE` sentinels** -- regenerable from
  `scan_corpus.py` against that index and the corpora above; the reduced
  `results_*.json` tables are what the paper's numbers cite.

## Committed eval-item dump is hash-only

`eval_items.jsonl.gz` in this directory is deliberately **hash-only**. The
full-text dump `dump_eval_items.py` writes carries every evaluation item's
normalized `stem` and `gold` -- the benchmark questions and their answers,
including 7 test-split labels of the 20 -- and committing that to a public
repository republishes the benchmark into the crawl path of future training
corpora, which is the very contamination this directory measures. So each of
the committed file's 40,582 rows keeps only `label`, `doc_id`,
`stem_sha256`, `gold_sha256`, `full_context_sha256`, `stem_words`,
`gold_words` and `n_candidates`, in the original row order, and the `stem`
and `gold` text fields are dropped. `eval_items_summary.json` drops its
per-label `example_stem` / `example_gold` for the same reason; every count,
every statistic, and the `normalizer_fingerprint` are kept.

What the committed file still supports:

- **`verify_item_identity.py`** reads only `label`, `doc_id`,
  `full_context_sha256` and `gold_sha256`, so the section 4b identity check
  -- that a training run scored the same item set this scan indexed -- runs
  unchanged against it.
- **`aggregate_by_domain.py`** reads only `label`, `stem_words` and
  `gold_words`, so re-reducing hit files into the `results_*.json` tables
  also runs from it.
- **The per-item hash record itself**: exact-match identity of any candidate
  text against an item this audit indexed. Normalize and hash the candidate
  the way `item_identity.py` does (`normalize`, then `sha256`) and compare
  against `stem_sha256` / `gold_sha256`.

What it does not support: **rebuilding the 13-gram span index offline.**
`build_item_index.py` indexes the literal strings, so it cannot run from a
file that no longer has them; it refuses the committed file with an error
that says so rather than failing on a `KeyError`. Rebuilding the index
therefore starts by regenerating the full-text dump with `dump_eval_items.py`
(step 4a under [Reproducing](#reproducing)), written outside this checkout
and never committed.

That regeneration needs the evaluator environment. The project's evaluator
environment pins
`ai2-olmo @ https://github.com/allenai/OLMo/archive/090253dac6688f2532509daa7aa2eb5fae50e956.tar.gz`
([`../olmo_core_token_selection/requirements-token-selection-eval.txt`](../olmo_core_token_selection/requirements-token-selection-eval.txt)),
and `dump_eval_items.py` also needs `torch`. That pin describes the pinned
evaluator environment; this repository does not record which ai2-olmo build
produced the committed dump, so the pin alone does not establish that a
regenerated dump reproduces it. The per-item hashes do: every hash is taken
after the committed normalizer, whose behavior is pinned by
`normalizer_fingerprint`, so a regenerated dump can be checked item by item
against the committed one, and any row that differs names an item whose
evaluator-side text changed. The script also forces HuggingFace offline mode
unless `HF_DATASETS_OFFLINE` / `HF_HUB_OFFLINE` are already set, so run it
where the eval datasets and `allenai/dolma2-tokenizer` are cached, or set both
to `0`. With `$FULLTEXT` and `$CONTAM` as under Reproducing:

```bash
python - "$FULLTEXT" "$CONTAM" <<'EOF'
import gzip, json, sys
from pathlib import Path

regen, committed = map(Path, sys.argv[1:])
KEEP = ("label", "doc_id", "stem_sha256", "gold_sha256", "full_context_sha256",
        "stem_words", "gold_words", "n_candidates")

def rows(d):
    with gzip.open(d / "eval_items.jsonl.gz", "rt", encoding="utf-8") as fh:
        return [tuple(json.loads(line)[k] for k in KEEP) for line in fh]

def fingerprint(d):
    summary = json.loads((d / "eval_items_summary.json").read_text(encoding="utf-8"))
    return summary["normalizer_fingerprint"]

assert fingerprint(regen) == fingerprint(committed), "normalizer changed"
new, old = rows(regen), rows(committed)
bad = [i for i, (a, b) in enumerate(zip(new, old)) if a != b]
print(f"{len(new)} vs {len(old)} rows; {len(bad)} differ (first rows: {bad[:5]})")
sys.exit(1 if bad or len(new) != len(old) else 0)
EOF
```

## Reproducing

Exact commands run for this paper, paths as on FarmShare. `dump_eval_items.py`
needs the private training monorepo's environment (ai2-olmo + torch; see
[Committed eval-item dump is hash-only](#committed-eval-item-dump-is-hash-only)
for the pinned evaluator); every step after it needs only this directory,
step 4a's full-text output, and Python's stdlib. The one departure from the
commands as run is that step 4a now writes to `$FULLTEXT`, outside this
checkout, rather than into `$CONTAM`, so a rerun does not overwrite the
committed hash-only files.

```bash
CONTAM=path/to/this/directory   # experiments/token-selection/contamination
FULLTEXT=path/outside/this/checkout   # full-text dump; never commit it

# 4a. Dump eval item text (needs ai2-olmo). The committed eval_items.jsonl.gz
# is this output with the stem/gold text dropped.
python "$CONTAM/dump_eval_items.py" \
  --out "$FULLTEXT/eval_items.jsonl.gz" \
  --summary "$FULLTEXT/eval_items_summary.json"

# 4a/6.5. Build the index from the full-text dump (stdlib only; not
# committed, see Dependencies above)
python "$CONTAM/build_item_index.py" \
  --items "$FULLTEXT/eval_items.jsonl.gz" \
  --out /path/to/item_index.pkl \
  --summary "$CONTAM/item_index_summary.json"

# One task-list TSV per corpus: <domain>\t<shard-id>\t<path>.
# regmix-10b: one line per domain, the corpus's single per-domain file.
for d in algebraic-stack arxiv dclm open-web-math pes2o starcoder wiki; do
  printf '%s\t0000\t%s\n' "$d" \
    "/scratch/users/nzhao2/agent-runs/regmix-10b-20260725-124810/trim/$d/$d-trimmed.json.gz"
done > regmix-10b.tsv

# hq-reference-v1: one line per domain, the corpus's per-domain directory
# (scan_corpus.py reads every *.json.gz/*.jsonl.zstd shard inside, sorted).
for d in algebraic-stack arxiv dclm open-web-math pes2o starcoder wiki; do
  printf '%s\t0000\t%s\n' "$d" "/scratch/users/nzhao2/hq-reference-v1/out/$d"
done > hq-reference-v1.tsv

# refhq-new-v1: one line per (source, category) documents directory; category
# is the aggregation domain, source becomes the shard id so several sources
# can contribute to one category.
find /scratch/users/nzhao2/refhq-new-v1/out -maxdepth 3 -type d -name documents \
  | while read -r dir; do
      cat=$(basename "$(dirname "$dir")")
      src=$(basename "$(dirname "$(dirname "$dir")")")
      printf '%s\t%s\t%s\n' "$cat" "$src" "$dir"
    done > refhq-new-v1.tsv

# Scan each corpus (Slurm array, one task per task-list line).
for corpus in regmix-10b hq-reference-v1 refhq-new-v1; do
  N=$(wc -l < "$corpus.tsv")
  CN_TASKLIST="$PWD/$corpus.tsv" CN_INDEX=/path/to/item_index.pkl \
    CN_OUT="/scratch/users/nzhao2/agent-runs/<run>/$corpus/hits" \
    sbatch --array="0-$((N - 1))" "$CONTAM/scan_array.sbatch"
done

# Aggregate each corpus once its array completes.
python "$CONTAM/aggregate_by_domain.py" \
  --hits /scratch/.../regmix-10b/hits --items "$CONTAM/eval_items.jsonl.gz" \
  --domains algebraic-stack,arxiv,dclm,open-web-math,pes2o,starcoder,wiki \
  --corpus-name regmix-10b --out "$CONTAM/results_regmix-10b.json"

python "$CONTAM/aggregate_by_domain.py" \
  --hits /scratch/.../hq-reference-v1/hits --items "$CONTAM/eval_items.jsonl.gz" \
  --domains algebraic-stack,arxiv,dclm,open-web-math,pes2o,starcoder,wiki \
  --corpus-name hq-reference-v1 --out "$CONTAM/results_hq-reference-v1.json"

python "$CONTAM/aggregate_by_domain.py" \
  --hits /scratch/.../refhq-new-v1/hits --items "$CONTAM/eval_items.jsonl.gz" \
  --domains chat,code,general,math,science \
  --corpus-name refhq-new-v1 --out "$CONTAM/results_refhq-new-v1.json"
```

## Superseded methodology (not vendored here)

This directory previously held a verbatim-13-gram audit (`n = 13`, hashed
with `PYTHONHASHSEED=0`, indexed over reconstructed HuggingFace fields,
denominator 40,087 items). That audit's numbers (0.89% / 1.09% / 3.67%
overall; 358 / 438 / 1,472 matched items) do not match this paper's current
text, and its scanner, its eval-side index builder, and the FarmShare run
directory it was copied from were never committed to this repo -- so when a
reviewer tried to check the paper's numbers against this directory, there
was nothing here that could reproduce either the old or the new figures.

Rather than reconcile the two audits' numbers after the fact, this directory
now vendors the pipeline that produced the paper's *current* numbers in
full, per [Dependencies and what's not vendored](#dependencies-and-whats-not-vendored)
above, so the join is checkable directly. The two audits are not expected to
agree term-for-term: a fixed-width 13-gram with no length floor, scanned
against reconstructed HuggingFace fields, is a different measurement from a
length-floored, provenance-tagged, exemplar-stripped one, and their
"self-contained" and "overall" columns are not even defined over the same
benchmark subsets (the old audit's self-contained rate flatly counts items
across 8 of the 10 benchmarks; this pipeline's `macro_stem_rate` averages
per-benchmark rates across all 10, MMLU included), so a single conversion
factor between them does not exist. [MMLU exemplar fix](#mmlu-exemplar-fix)
below works through the closest thing to a before/after comparison this
directory can make -- the same benchmark, on the same corpus, indexed with
and without exemplar-stripping -- and is explicit about where even that
comparison's denominators still differ.
