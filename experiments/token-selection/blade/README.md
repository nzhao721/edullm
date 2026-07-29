# BLADE arm (RegMix 10B)

Bi-level adaptive token selection with a **dynamic reference** synchronized to the
proxy on a locked schedule. Rebuild of `experiments/token-selection/blade/` under
the shared RefHQ-matched OLMo-2 370M contract.

| Knob | Value |
|------|-------|
| Architecture | `TransformerConfig.olmo2_370M` (full attn) |
| GBS / seq / LR | `4_194_304` / 2048 / `4e-4` (warmup 24, `alpha_f=0.1`) |
| `total_steps` | ≈ **2384** (`10B // 4194304`) |
| `blade_start` | **500** (steps `0..499` proxy-only full CE) |
| `tau` / `K` / `γ` / `λ` | **375** / **75** / **0.6** / **1.0** |
| Sync steps | **500, 875, 1250, 1625, 2000** then hold ref → 2384 |
| Selection score | **`L_ref − L_proxy`**, keep top-γ |
| Train | RegMix 10B |
| Val / HQ (K updates) | RefHQ 5.5B |
| Run id (example) | `blade-regmix10b-v2` |
| S3 export | `s3://edullm-checkpoints/token-sel/blade/` (`S3_EXPORT=0` to disable) |

## Files

| File | Role |
|------|------|
| [`prepare_blade_data.py`](prepare_blade_data.py) | Local memmap path lists (RegMix + RefHQ) |
| [`train_blade_olmo_370m.py`](train_blade_olmo_370m.py) | Trainer (1..N GPU via `torchrun`) |
| [`launch_train.sh`](launch_train.sh) | Thin `torchrun` wrapper (`NPROC_PER_NODE`) |

## Sync / checkpoint / resume design

**Phase timeline**

```text
steps 0 .. 499      proxy warmup (full CE, no selection); reference absent
step 500            sync #1: ref ← proxy; K=75; then γ=0.6 selection + proxy step
step 875            sync #2
step 1250           sync #3
step 1625           sync #4
step 2000           sync #5
steps 2001 .. 2384  hold last reference; continue γ=0.6 selection
```

**Permanent ladder** (shared helper): `{0, 125, …, 2250, 2384}` — **omit 2375**.
Every sync step is on this ladder.

**Each permanent `state.pt` stores**

1. **Proxy** `train_module` = full unsharded model + optimizer (`full_state_dict` gather)
2. **Dynamic reference** = dense weights after the latest sync+K (or `null` before first sync)

Format id: `blade_proxy_ref_v1`.

At a sync-step checkpoint, the reference is saved **after** K updates (and after that
step’s proxy update), so a resume at that step continues selection with the same
reference the original run would have used next.

**Resume (methodology-safe)**

- Load proxy + reference from the checkpoint.
- **Do not** re-sync `ref ← proxy` unless the next step is itself a scheduled sync.
- Between syncs the loaded reference is exactly the frozen post-K snapshot.

**Task-loss eval**

On every permanent save, rank 0 spawns async
`scripts/farmshare/task_loss/eval_task_loss_olmo_core.py` against **proxy** weights
→ `task_loss_results/blade/step{N}_task_loss.json`. Disable with `--no-task-loss-eval`
or `TASK_LOSS_EVAL=0` (env is honored when CLI eval is left enabled).

## Prepare data (local corpora only)

```bash
python experiments/token-selection/blade/prepare_blade_data.py \
  --work /data/blade \
  --train-tokenized-root /data/regmix-10b/tokenized \
  --ref-tokenized-root /data/refhq-5p5b/tokenized
```

## Launch

```bash
# 1 GPU
bash experiments/token-selection/blade/launch_train.sh \
  --name blade-regmix10b-v2 \
  --train-paths-file /data/blade/train_tokenized/paths_train.txt \
  --ref-paths-file /data/blade/ref_tokenized/paths_refhq.txt \
  --save-folder /data/ckpts/blade-regmix10b-v2 \
  --progress-dir /data/runs/blade-regmix10b-v2 \
  --length-tokens 10000058051 \
  --fresh

# N GPUs (world size from env; GBS must divide evenly)
NPROC_PER_NODE=4 bash experiments/token-selection/blade/launch_train.sh \
  --name blade-regmix10b-v2 \
  ...same args... \
  --fresh
```

Or call `torchrun` directly with `PYTHONPATH=experiments/token-selection`.

Resume: omit `--fresh` (loads latest `step*` under `--save-folder`) or pass
`--load-path /data/ckpts/blade-regmix10b-v2/step1250`.

Does **not** submit AWS jobs.
