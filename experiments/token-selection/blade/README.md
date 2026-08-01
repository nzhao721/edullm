# BLADE arm (RegMix 10B)

Bi-level adaptive token selection with a **dynamic reference** synchronized to the
proxy on a locked schedule. Rebuild of `experiments/token-selection/blade/` under
the shared RefHQ-matched OLMo-2 370M contract.

| Knob | Value |
|------|-------|
| Architecture | `TransformerConfig.olmo2_370M` (full attn) |
| GBS / seq / LR | `4_194_304` / 2048 / `4e-4` (warmup 24, `alpha_f=0.1`) |
| `total_steps` | **2360** (`9900000000 // 4194304`; shared one-epoch matrix budget under ~9.989B published train) |
| `blade_start` | **500** (steps `0..499` proxy-only full CE) |
| `tau` / `K` / `γ` / `λ` | **375** / **75** / **0.6** / **1.0** |
| Sync steps | **500, 875, 1250, 1625, 2000** then hold ref → end |
| Selection score | **`L_ref − L_proxy`**, keep top-γ |
| Train | `pretrain/regmix-10b` (`s3://edullm-data/`, `resolve_latest`) |
| Val / HQ (K updates) | `pretrain/refhq-regmix-5p5b` (`s3://edullm-data/`, `resolve_latest`) |
| Run id (example) | `blade-regmix10b-v2` |
| Artifact durability | Runtime scratch + W&B (**fail-closed for production online checkpoint uploads**) |

## Ephemeral-machine contract

Scratch starts empty and is wiped after the job:

1. Stage corpora each run from `s3://edullm-data` via `prepare_blade_data.py` (`BLADE_WORK`).
2. Write checkpoints / progress / task_loss JSON under job-scoped scratch paths.
3. Permanent saves upload to W&B; production online checkpoint uploads fail closed.
4. Never use legacy dataset buckets or pre-staged FarmShare/laptop corpora.
5. Cross-job resume: `--wandb-resume-artifact entity/project/name:alias` restores into scratch before load.

## Files

| File | Role |
|------|------|
| [`prepare_blade_data.py`](prepare_blade_data.py) | Resolve + stage edullm-data shards → local path lists |
| [`train_blade_olmo_370m.py`](train_blade_olmo_370m.py) | Trainer (1..N GPU via `torchrun`); permanent steps complete checkpoint → synchronous 20-label eval → strict export → durable marker |
| [`launch_train.sh`](launch_train.sh) | Thin `torchrun` wrapper (`NPROC_PER_NODE`, requires `BLADE_WORK` or path lists) |

## Sync / checkpoint / resume design

**Phase timeline**

```text
steps 0 .. 499      proxy warmup (full CE, no selection); reference absent
step 500            sync #1: ref ← proxy; K=75; then γ=0.6 selection + proxy step
step 875            sync #2
step 1250           sync #3
step 1625           sync #4
step 2000           sync #5
steps 2001 .. end   hold last reference; continue γ=0.6 selection
```

**Permanent ladder** (shared helper): `{0, 125, …}` plus final step — omit the
penultimate ladder tick when it would collide with `total_steps - interval`.
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

On every permanent save, rank 0 may spawn async `task_loss_bpb` eval against **proxy**
weights via `token_selection.olmo_ext.task_loss_hook`. Results land under
`<progress-dir>/task_loss_results/` (job-scoped) and upload to W&B with progress.
Set `TASK_LOSS_EVAL_SCRIPT` if the eval entrypoint is not on the machine; disable with
`--no-task-loss-eval` or `TASK_LOSS_EVAL=0` (missing script skips eval without failing train).

## Prepare data (from edullm-data on a clean machine)

Requires the `edullm-data` package and AWS read access to `s3://edullm-data`.

```bash
python experiments/token-selection/blade/prepare_blade_data.py \
  --work /scratch/$USER/blade-work
# stages pretrain/regmix-10b + pretrain/refhq-regmix-5p5b (resolve_latest)
# → paths_train.txt, paths_refhq.txt, length_tokens.txt
```

## Launch

```bash
# Clean / ephemeral machine: stage + train
BLADE_WORK=/scratch/$USER/blade-work \
bash experiments/token-selection/blade/launch_train.sh \
  --name blade-regmix10b-v2 \
  --save-folder /scratch/$USER/ckpts/blade-regmix10b-v2 \
  --progress-dir /scratch/$USER/runs/blade-regmix10b-v2 \
  --fresh

# Or after an explicit in-job prepare:
bash experiments/token-selection/blade/launch_train.sh \
  --name blade-regmix10b-v2 \
  --train-paths-file /scratch/$USER/blade-work/train_tokenized/paths_train.txt \
  --ref-paths-file /scratch/$USER/blade-work/ref_tokenized/paths_refhq.txt \
  --save-folder /scratch/$USER/ckpts/blade-regmix10b-v2 \
  --progress-dir /scratch/$USER/runs/blade-regmix10b-v2 \
  --length-tokens "$(cat /scratch/$USER/blade-work/length_tokens.txt)" \
  --fresh

# N GPUs (world size from env; GBS must divide evenly)
NPROC_PER_NODE=4 BLADE_WORK=/scratch/$USER/blade-work \
bash experiments/token-selection/blade/launch_train.sh \
  --name blade-regmix10b-v2 \
  --save-folder /scratch/$USER/ckpts/blade-regmix10b-v2 \
  --progress-dir /scratch/$USER/runs/blade-regmix10b-v2 \
  --fresh
```

Or call `torchrun` directly with `PYTHONPATH=experiments/token-selection`.

Same-job resume: omit `--fresh` (loads latest `step*` under `--save-folder`) or pass
`--load-path …/step1250`.

Cross-job resume on wiped scratch:

```bash
BLADE_WORK=/scratch/$USER/blade-work \
WANDB_RESUME_ARTIFACT=<entity/project/blade-regmix10b-v2-checkpoint:step-0001250> \
LOAD_PATH=/scratch/$USER/ckpts/blade-regmix10b-v2/step1250 \
bash experiments/token-selection/blade/launch_train.sh \
  --name blade-regmix10b-v2 \
  --save-folder /scratch/$USER/ckpts/blade-regmix10b-v2 \
  --progress-dir /scratch/$USER/runs/blade-regmix10b-v2
# omit --fresh and set WANDB_RESUME_ARTIFACT to restore a checkpoint
```

Does **not** submit AWS jobs. S3 credentials are needed only for input staging;
run artifacts stay on scratch and are uploaded to W&B.
