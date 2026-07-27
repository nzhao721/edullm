# Mixing-law pilot: 24 × DataDecide-60M on the p6 B200 cluster

Fit Ye et al. (arXiv:2403.16952) data mixing laws to **OLMo-ladder task loss**
(bits-per-byte on the OLMES 5-shot RC suite), using the exact DataDecide 60M
geometry and a designed grid of 24 mixtures over the 7 OLMoHQ domains.

**Do not start anything until you have reviewed this plan.** All scripts are
written; nothing has been launched.

---

## What you get

| Piece | Path |
|---|---|
| Mixture grid | `mixtures.json` |
| Shared constants | `mixlaw_common.py` |
| Tokenize working pool from olmohq | `tokenize_working_pool.py` |
| Select + fetch only needed shards | `select_and_fetch_shards.py` |
| Plan + materialize slices | `build_mixture_data.py` |
| One-shot data prep | `prepare_data.sh` |
| Train one mixture (1 GPU) | `train_datadecide_60m.py` |
| Full 20-label task-loss eval | `eval_task_loss.py` |
| Per-mixture worker | `run_mixture.sh` |
| Fit mixing law + optimize | `fit_mixing_law.py` |
| Budget table | `budget_calculator.py` |

---

## Compute envelope: ≈12 B200 GPU-hours

| | |
|---|---|
| Mixtures | 24 |
| Model | DataDecide 60M body (`d_model=384`, 16L, 12H, batch 96, LR 5.8e-3) |
| Params (this run) | **114.8M total** / **76.3M non-embedding** (dolma2, untied) |
| Default budget | **tokens/param = 5** → **285M tokens / 1451 steps** per mix |
| Per-mix wall | ~30 min (train + final task-loss eval) |
| Total GPU-hours | **≈12** |
| Data source | `s3://edullm-dataset-olmohq/olmo-mix-1124-30b` |

The olmohq pool is large enough that **wiki alone supports ~30B tokens/mix**.
At tokens/param=5 the binding constraint is **GPU time**, not data. Calibrated
from the GPU7 smoke (~206k tok/s, mbs=32).

Calibrate after a smoke run: measure tok/s, then raise/lower `TOKENS_PER_PARAM`.

```bash
python budget_calculator.py
```

---

## Parallelization

One mixture = one GPU. Wall clock scales as `12 / N` hours:

| GPUs | Waves | Est. wall |
|---:|---:|---:|
| 1 | 24 | ~12 h |
| 4 | 6 | ~3 h |
| 8 | 3 | ~1.5 h |
| 24 | 1 | ~0.5 h |

Run mixtures with `run_mixture.sh` (one mix per GPU). Use your own Slurm array
or multi-GPU scheduler to assign mix IDs 1–24 across available GPUs:

```bash
# One mixture on GPU 0
bash run_mixture.sh 1 0

# Example Slurm array (one task per mixture, one GPU each)
# sbatch --array=1-24%8 --gres=gpu:1 --wrap 'bash run_mixture.sh $SLURM_ARRAY_TASK_ID 0'
```

---

## Evaluation: task loss (not LM val loss)

No held-out corpus split. Targets are OLMo-ladder **bits-per-byte** on the gold
continuation of the OLMES 5-shot RC suite (20 labels → 13 task families).

- **In-run curve** (cheap, every 120 steps): ARC + MMLU val bpb — for the step law
- **Final eval** (once per checkpoint): full 20-label suite — for the mixing law

---

## Data path (olmohq → slices)

olmohq has **raw** `data/<domain>/*.json.gz` only (no `tokenized/`). Prep does
**not** sync the full ~130 GiB tree:

1. Pull the tiny `plan/manifest.jsonl` inventory
2. Randomly select only enough shards to cover peak demand at the chosen tpp
   (`select_and_fetch_shards.py`, ~a few GiB at the default budget)
3. Tokenize a working pool from those shards
4. Random sequence-aligned block subsample per mixture → exact weight realization

```bash
# On shared storage every compute node can read (set up venv with torch + olmo_core first)
bash prepare_data.sh          # TOKENS_PER_PARAM=5 by default
# Dry-run the shard plan without downloading:
#   python select_and_fetch_shards.py --manifest … --dry-run …
```

---

## Recommended run order (when you are ready)

1. Set up a venv with `torch`, `ai2-olmo-core`, and dependencies
2. `bash prepare_data.sh` on shared FS
3. Full grid: run `run_mixture.sh` for mix IDs 1–24 across your GPUs
4. `python fit_mixing_law.py collect --runs-dir $WORK/runs`
5. `python fit_mixing_law.py fit --data mixlaw_data.json` (reports LOO CV)

---

## Model parameters

Exact DataDecide 60M **body** (`d_model=384`, 16 layers, 12 heads, `mlp_ratio=8`,
seq 2048, untied LM head). Vocabulary is **dolma2** (`embedding_size=100,352`)
because the olmohq shards are tokenized that way.

| | Body (blocks + final norm) | Non-embedding (excl. `wte`) | Total (untied) |
|---|---:|---:|---:|
| **This run (dolma2)** | 37,761,408 | **76,296,576** | **114,831,744** |
| DataDecide published (50,304-row embed) | 37,761,408 | 57,078,144 | 76,394,880 |

OLMo’s “model size” for tokens/param is non-embedding (everything except `wte`,
so the untied LM head counts). Token budgets still use the published
**57,078,144** DataDecide size so tokens/param stays comparable to the paper;
the body geometry that defines the 60M scale is unchanged.
