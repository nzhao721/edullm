# Middle-PPL token arm

Keep the **middle 60%** of valid tokens per sequence by **frozen RefHQ late-avg**
token CE (`L_ref` ≈ log-PPL). Drop the easiest and hardest `(1−k)/2` each.
Per Marion et al. ([2309.04564](https://arxiv.org/pdf/2309.04564)), perplexity
ranking uses a **separate reference model**, not the training model.

Scorer lives in the shared package (`middle_ppl`); this directory owns the run
config, launch scripts, and artifacts.

| Knob | Value |
|------|-------|
| Method | `middle_ppl` |
| Reference | RefHQ avg(steps 1000, 1125, 1315) — same late ref as `middle-ppl-doc` |
| Keep rate `k` | 0.6 |
| Masking warmup | `t0_steps=0` (selection from step 0) |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Corpus | RegMix one-epoch (`pretrain/regmix-10b` on `s3://edullm-data/`) → **2360** steps (`9900000000 // GBS`) |
| Permanent ckpts | `{0, 125, …, 2125, 2360}` (skip 2250) |
| `run_id` | `middle-ppl-token-10b-v2` |
| Task loss | full 20-label RC 5-shot `task_loss_bpb` on every permanent ckpt |
| Artifact durability | Runtime scratch + W&B |

**Ephemeral scratch:** set `WORK`/`RUN_DIR` to an empty job scratch dir; stage from
edullm-data each run; durable artifacts upload to W&B. For `RESUME=1` on an empty
save folder, set `WANDB_RESUME_ARTIFACT`. Do not use the repo tree as scratch.

Config: [`configs/run_middle_ppl_token_10b.yaml`](configs/run_middle_ppl_token_10b.yaml).

## Launch (1…N GPUs)

Hardware-agnostic: discover `nproc` from `NUM_GPUS` / `CUDA_VISIBLE_DEVICES` /
`nvidia-smi` (else 1). Outside Slurm, set `CUDA_VISIBLE_DEVICES`. Optional:
`RANK_MICROBATCH_SIZE`. Global batch stays `4_194_304`.

```bash
export EDULLM_ROOT=/path/to/edullm
export OLMO_CORE_DIR=/path/to/OLMo-core   # pinned revision in YAML
export CUDA_VISIBLE_DEVICES=0             # required outside Slurm
export NUM_GPUS=1                         # or 2, 4, …
export WORK=/path/to/empty/scratch        # required (job-local tokens/ckpts)
# Optional: RANK_MICROBATCH_SIZE=16384    # smaller GPUs

bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/launch_train.sh"
```

Resume (durable):

```bash
RESUME=1 WORK=/path/to/scratch \
  bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/launch_train.sh"
```

Prep only (sync tokens, manifest, freeze order, validate — no train):

```bash
MODE=prepare WORK=/path/to/scratch \
  bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/launch_train.sh"
```

## Manual task-loss eval (one checkpoint)

Usually fired automatically on each permanent save. To re-run:

```bash
bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/eval_checkpoint.sh" \
  /path/to/checkpoints/middle_ppl/step125
```

Outputs: `experiments/token-selection/task_loss_results/middle-ppl-token/step{N}_task_loss.json`
(or under `$WORK` when results_dir is rewritten).

## Disable async eval during train

```bash
TASK_LOSS_EVAL=0 bash …/launch_train.sh
```

Checkpoints remain on runtime scratch and are uploaded to W&B artifacts.
Production online runs fail closed if a required checkpoint upload fails.
