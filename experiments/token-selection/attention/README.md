# Attention top-60% (`attention_topk`)

Shared package: [`../token_selection/`](../token_selection/).  
Config: [`configs/run_attention_10b.yaml`](configs/run_attention_10b.yaml).

## Method

ssToken-adapted **causal attention received** on the **last transformer layer**:

```text
score[i] = mean_h( sum_{j >= i} A^h[j, i] )
```

Keep the top **60%** of valid target tokens per sequence (`k=0.6`). Selection is
active from **step 0** (`t0_steps=0` / `t0_frac=0`). No frozen reference / EMA.

FlashAttention path: during the train forward, a pre-hook captures the last
block’s attention input; Q/K are recomputed for that layer only to form causal
weights (no full-matrix materialization during FlashAttention).

## Contract

| Knob | Value |
|------|--------|
| Arch | `olmo2_370M` (RefHQ-matched) |
| Data | `pretrain/regmix-10b` via `data.dataset_id` → `s3://edullm-data/` (staged per job) |
| Steps | **2360** (`9900000000 // 4_194_304`) |
| Keep rate `k` | 0.6 |
| Masking warmup | `t0_steps=0` (selection from step 0) |
| Permanent ckpts | `{0, 125, …, 2125, 2360}` (omit 2250) |
| Eval | full 20-label `task_loss_bpb` on each permanent save |
| `run_id` | `attention-topk-10b-scratch-v1` |
| S3 export | `s3://edullm-checkpoints/token-sel/attention/` |

## Launch

From `experiments/token-selection/` (set `PYTHONPATH` to that dir, or use the
script which exports it):

```bash
export OLMO_ROOT=/path/to/OLMo-core   # pinned revision in the YAML

# Single GPU
CUDA_VISIBLE_DEVICES=0 NPROC=1 bash attention/launch.sh

# Multi-GPU (example: 4)
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash attention/launch.sh
```

Or directly:

```bash
export PYTHONPATH=experiments/token-selection
CUDA_VISIBLE_DEVICES=<gpus> python -m torch.distributed.run --standalone \
  --nproc_per_node=<N> -m token_selection.scripts.train_olmo_template \
  --config attention/configs/run_attention_10b.yaml \
  --method attention_topk \
  --olmo-root "$OLMO_ROOT" \
  --launch
```

No hardcoded GPU count or host paths. Under Slurm, leave
`cuda_visible_devices` empty so the allocator’s `CUDA_VISIBLE_DEVICES` wins.

Dry-run (print plan only): omit `--launch`.

Resume: `RESUME=1 bash attention/launch.sh` (fingerprint-gated). On ephemeral
scratch, `--resume` fetches `run_fingerprint.json` + step dirs from
`s3://edullm-checkpoints/token-sel/attention/` when the local save folder is
empty — keep `S3_EXPORT=1` during training.

Task-loss results: `task_loss_results/attention/step{N}_task_loss.json`
(disable with `TASK_LOSS_EVAL=0`).
