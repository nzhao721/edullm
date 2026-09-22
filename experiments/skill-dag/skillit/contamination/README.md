# Benchmark contamination audit -- Skill-It's time-varying exposure

Skill-It reweights domains mid-training (see `../README.md`'s "Skill-It
update"), so unlike the fixed-mixture arms in `../../mixlaw/contamination/`, a
single static weight vector does not describe either of its two arms'
exposure to contaminated eval text over a full run. This directory computes
that exposure honestly: per-segment, then as a token-weighted average over
the run.

**This directory does not re-scan the reservoir.** Both Skill-It arms train
on domain-stratified samples of the same `pretrain/olmo-127b` reservoir the
MixLaw arms do, against the same 40,582-item eval suite, so the per-domain
contamination rates are identical; only the mixture weights differ, and only
because they change over time here. `../../mixlaw/contamination/` holds the
scan, the eval-item index pipeline, and `results_olmo127b-reservoir.json`;
this directory reads that file directly rather than duplicating a ~10 MB
eval-item dump and a 172 MB index for a scan that would produce the same
numbers.

## The two arms

| Arm | Starting mixture | Adjacency |
| --- | --- | --- |
| Offline probe | LightGBM optimum (`LGB-min1pct`) | fixed offline `A` |
| Online derivative | LightGBM optimum (`LGB-min1pct`) | recomputed `A(r)` from MixLaw derivatives at each update |

Both arms start from the same mixture -- the LightGBM-optimized mixture --
and differ only in how the adjacency that drives each update is built.
Comparisons here are reported against two reference points:
`olmo-mix-1124` (the shared baseline `../../mixlaw/contamination/exposure_by_arm.json`
already uses) and `LGB-min1pct` itself, since that is the mixture both arms
actually start from and is also one of `exposure_by_arm.json`'s four rows.

## Method

Each arm's realized mixture is piecewise-constant: it holds fixed between
Skill-It's five update steps (500, 875, 1250, 1625, 2000), jumping to a new
mixture at each one, then holds again until the run ends at step 2384. Six
segments in total for a full run. For each segment,

    exposure(segment) = sum over domains of  weight(segment, d) * rate(d)

exactly as for a fixed mixture, using the domain-weighted rates from
`../../mixlaw/contamination/results_olmo127b-reservoir.json`. Segments are then
combined into a single run-level number by weighting each segment's exposure
by its length in steps -- token-weighted, since every step draws the same
number of tokens (a run's contaminated-word exposure in aggregate, not just
at any one point in it):

    exposure(run) = sum over segments of  length(segment) * exposure(segment)  /  total_steps

`trajectory_exposure.py` computes this from each arm's own
`skillit_updates_<arm>.jsonl` -- one row per Skill-It update step, `step`
and the realized post-update mixture `p_after`.

**Provenance of these six-row files.** Each is `step` and `p_after` read
directly off the raw per-step Skill-It update log (`skillit_updates.jsonl`)
written by that arm's own FarmShare training run --
`skillit-370m-probe-rerun-20260918-011120` for offline probe,
`skillit-370m-deriv-20260916-124719` for online derivative -- not
transcribed from a table. Both logs' step-0 `p_after` matches
`LGB-min1pct`'s published weight vector in
`../../mixlaw/validation_mixtures_10b.json` to full float precision,
confirming both arms actually start from the LightGBM-optimized mixture.

## Results

All figures below are from `exposure_offline-probe.json` and
`exposure_online-derivative.json`. Both arms start at span rate 1.494e-05,
doc rate 6.545e-04 -- `LGB-min1pct`'s own exposure
(`../../mixlaw/contamination/README.md`'s per-arm table).

### Offline probe

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.553 | 0.011 | 1.494e-05 | 6.545e-04 |
| 500 | 375 | 0.646 | 0.011 | 1.610e-05 | 6.748e-04 |
| 875 | 375 | 0.720 | 0.011 | 1.698e-05 | 6.889e-04 |
| 1250 | 375 | 0.779 | 0.010 | 1.765e-05 | 6.991e-04 |
| 1625 | 375 | 0.827 | 0.009 | 1.817e-05 | 7.067e-04 |
| 2000 | 384 | 0.864 | 0.008 | 1.856e-05 | 7.123e-04 |

**Time-weighted average: span rate 1.696e-05 (0.89x the `olmo-mix-1124`
baseline; 1.14x the LightGBM-optimized mixture it actually starts from),
doc rate 6.877e-04 (0.96x baseline; 1.05x its own start).**

### Online derivative

| Step | Length (steps) | `weight[dclm]` | `weight[wiki]` | Span rate | Doc rate |
| --- | --- | --- | --- | --- | --- |
| 0 | 500 | 0.553 | 0.011 | 1.494e-05 | 6.545e-04 |
| 500 | 375 | 0.556 | 0.016 | 1.536e-05 | 6.554e-04 |
| 875 | 375 | 0.557 | 0.021 | 1.583e-05 | 6.564e-04 |
| 1250 | 375 | 0.556 | 0.028 | 1.642e-05 | 6.579e-04 |
| 1625 | 375 | 0.553 | 0.037 | 1.712e-05 | 6.597e-04 |
| 2000 | 384 | 0.549 | 0.047 | 1.795e-05 | 6.621e-04 |

**Time-weighted average: span rate 1.621e-05 (0.86x baseline; 1.08x its own
start), doc rate 6.575e-04 (0.92x baseline; 1.00x its own start).**

### Reading these together

Both arms start at the identical LightGBM-optimized mixture but diverge
sharply in how they move. Offline probe's fixed adjacency drives `dclm`
weight up monotonically -- 0.553 to 0.864 by step 2000 -- squeezing every
other domain down, `wiki` included (0.011 to 0.008). Online derivative's
recomputed adjacency does the opposite on `wiki`: it more than quadruples
that share (0.011 to 0.047) while leaving `dclm` close to where it started
(0.553 to 0.549).

Despite moving in opposite directions on the highest-rate domain, both
arms end up *more* exposed than their own starting mixture. `dclm`'s own
span rate (1.965e-05) already sits above the LightGBM blend's average
(1.494e-05), so offline probe's concentration into `dclm` raises exposure
by itself, with no help from `wiki` -- exposure rises even as the arm
moves away from the single highest-rate domain. Online derivative's rise
comes from the opposite mechanism: `wiki`'s span rate (9.504e-05) is 4.8x
`dclm`'s, so even a modest reallocation onto it (1.1% to 4.7% of the
mixture) is enough to lift the time-weighted average while `dclm` barely
moves.

Span rate and document rate agree in direction for both arms here (both
rise relative to each arm's own start), unlike the mixlaw arms in
`../../mixlaw/contamination/README.md`, where a large realized shift onto
`wiki` specifically is what pulls the two metrics apart. Neither arm here
moves enough onto (or away from) `wiki` on its own to produce that split at
the level of the time-weighted average.

This directory reports exposure only -- it does not compare either arm's
task-loss outcome, since a same-scale contamination comparison depends on
knowing what each arm's own results were, which belongs in `../README.md`
alongside this run's other metrics.

## Code map

- `trajectory_exposure.py` -- computes the per-segment and time-weighted
  average exposure described above. Stdlib only. Reads
  `../../mixlaw/contamination/results_olmo127b-reservoir.json` for per-domain
  rates; needs no other file from `../../mixlaw/`.
- `skillit_updates_offline-probe.jsonl`, `skillit_updates_online-derivative.jsonl`
  -- each arm's realized per-update-step mixture, `step` and `p_after` read
  directly off that arm's own FarmShare `skillit_updates.jsonl` run log (see
  Provenance above). Six rows each (steps 0, 500, 875, 1250, 1625, 2000).
- `exposure_offline-probe.json`, `exposure_online-derivative.json` --
  `trajectory_exposure.py`'s output for each arm; the source of every number
  in Results above.

For the underlying per-domain scan (the eval-item dump, index, scanner and
aggregator, and their own dependencies), see
[`../../mixlaw/contamination/README.md`](../../mixlaw/contamination/README.md).

## Reproducing

```bash
CONTAM=path/to/this/directory       # experiments/skill-dag/skillit/contamination
MIXLAW=path/to/mixlaw/contamination # experiments/skill-dag/mixlaw/contamination

# The per-domain scan itself is reproduced from $MIXLAW; see that
# directory's own README. Given results_olmo127b-reservoir.json there:

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_offline-probe.jsonl" \
  --arm offline-probe --final-step 2384 \
  --out "$CONTAM/exposure_offline-probe.json"

python "$CONTAM/trajectory_exposure.py" \
  --per-domain "$MIXLAW/results_olmo127b-reservoir.json" \
  --updates "$CONTAM/skillit_updates_online-derivative.jsonl" \
  --arm online-derivative --final-step 2384 \
  --out "$CONTAM/exposure_online-derivative.json"
```

To reproduce the `skillit_updates_<arm>.jsonl` files themselves from source,
pull the raw `skillit_updates.jsonl` progress log written by each arm's own
FarmShare run (`skillit-370m-probe-rerun-20260918-011120` for offline probe,
`skillit-370m-deriv-20260916-124719` for online derivative) and extract each
row's `step` and `p_after` fields; the rest of that log (`A`, `losses`,
`p_before`, `r`, ...) is not needed here.
