# Middle-PPL token arm

Keep the **middle 60%** of valid tokens per sequence by current-model CE
(`L_curr` ≈ log-PPL). Drop the easiest and hardest `(1−k)/2` each.
Scorer lives in the shared package (`middle_ppl`); this directory owns the run
config, launch scripts, and artifacts.

| Knob | Value |
|------|-------|
| Method | `middle_ppl` |
| Keep rate `k` | 0.6 |
| Masking warmup | `t0_steps=0` (selection from step 0) |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Corpus | RegMix 10B → **2384** steps |
| Permanent ckpts | `{0, 125, …, 2250, 2384}` (skip 2375) |
| `run_id` | `middle-ppl-token-10b-v1` |
| Task loss | full 20-label RC 5-shot `task_loss_bpb` on every permanent ckpt |
| S3 export | `s3://edullm-checkpoints/token-sel/middle-ppl-token/` |

Config: [`configs/run_middle_ppl_token_10b.yaml`](configs/run_middle_ppl_token_10b.yaml).

## Launch (1…N GPUs)

Hardware-agnostic: discover `nproc` from `NUM_GPUS` / `CUDA_VISIBLE_DEVICES` /
`nvidia-smi` (else 1). Outside Slurm, set `CUDA_VISIBLE_DEVICES`. Optional:
`RANK_MICROBATCH_SIZE`, `WORK` (local data/ckpt root). Global batch stays
`4_194_304`.

```bash
export EDULLM_ROOT=/path/to/edullm
export OLMO_CORE_DIR=/path/to/OLMo-core   # pinned revision in YAML
export CUDA_VISIBLE_DEVICES=0             # required outside Slurm
export NUM_GPUS=1                         # or 2, 4, …
# Optional: RANK_MICROBATCH_SIZE=16384    # smaller GPUs
# Optional: WORK=/path/to/scratch

bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/launch_train.sh"
```

Resume:

```bash
RESUME=1 bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/launch_train.sh"
```

Prep only (sync tokens, manifest, freeze order, validate — no train):

```bash
MODE=prepare bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/launch_train.sh"
```

## Manual task-loss eval (one checkpoint)

Usually fired automatically on each permanent save. To re-run:

```bash
bash "$EDULLM_ROOT/experiments/token-selection/middle-ppl-token/eval_checkpoint.sh" \
  /path/to/checkpoints/middle_ppl/step125
```

Outputs: `experiments/token-selection/task_loss_results/middle-ppl-token/step{N}_task_loss.json`.

## Disable async eval during train

```bash
TASK_LOSS_EVAL=0 bash …/launch_train.sh
```
