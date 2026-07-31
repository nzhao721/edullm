# Baseline OLMo-ladder 370M CPT

Train **OLMo-ladder 370M** from scratch on the ~30B FarmShare tokenized corpus
(`edullm-370M-30B`), using the exact scaling laws and LR schedule from
[allenai/OLMo-ladder](https://github.com/allenai/OLMo-ladder) (`src/ladder/ladder.py`).

Checkpoints are published under `s3://edullm-checkpoints/olmo2-370m-cpt/edullm-370M-30B/`
(steps 5000, 10000, 15000). See [`S3_CHECKPOINTS.md`](../../S3_CHECKPOINTS.md) for details.

## Layout

| Path | Purpose |
|------|---------|
| [`farmshare/train_olmo_ladder_370m.py`](farmshare/train_olmo_ladder_370m.py) | Main training entrypoint (FarmShare Slurm / multi-GPU DDP) |
| [`farmshare/upload_ckpts_to_s3.sbatch`](farmshare/upload_ckpts_to_s3.sbatch) | Sync unsharded checkpoints (steps 5000/10000/15000) to S3 |
| [`aws/train_olmo_ladder_370m.py`](aws/train_olmo_ladder_370m.py) | AWS variant (single B200, no W&B, no evals; also used for RefHQ probes) |
| [`aws/setup_venv.sh`](aws/setup_venv.sh) | EC2 venv bootstrap (ai2-olmo 0.6, torch, flash-attn, OLMo-ladder clone) |

Shared FarmShare platform helpers (`bootstrap.sh`, `prepare_aws_session*.sh`, …) stay in
[`scripts/farmshare/`](../../scripts/farmshare/). Cross-corpus dataset utilities
(`olmo_shard_utils.py`, `download_s3_shard.py`, …) live in [`datasets/`](../../datasets/).

## FarmShare run (typical)

1. Stage a run directory under `/scratch/users/$USER/agent-runs/` with `env.sh`, venv,
   tokenized memmap path lists, and `OLMO_LADDER_ROOT` pointing at a clone of OLMo-ladder.
   FlashAttention does not work on FarmShare — leave `OLMO_FLASH_ATTENTION=0` and use SDPA.
2. Launch training with `torchrun`, e.g.:

```bash
source "${RUN_DIR}/env.sh"
source "${RUN_DIR}/venv/bin/activate"
export OLMO_LADDER_ROOT="${RUN_DIR}/OLMo-ladder"   # or path from env.sh

torchrun --nproc_per_node=4 \
  "${EDULLM_ROOT}/experiments/baseline/farmshare/train_olmo_ladder_370m.py" \
  --name edullm-370M-30B \
  --paths-file "${RUN_DIR}/tokenized/train_paths.txt" \
  --val-paths-file "${RUN_DIR}/tokenized/val_paths.txt" \
  --save-folder "${RUN_DIR}/checkpoints/edullm-370M-30B" \
  --progress-dir "${RUN_DIR}/progress" \
  --length-tokens 31303986152 \
  --device-batch-size 8 \
  --save-interval 500
```

3. After training, upload selected unsharded checkpoints:

```bash
export RUN_DIR=... AWS_SESSION_ENV=... BUCKET=edullm-checkpoints
sbatch "${EDULLM_ROOT}/experiments/baseline/farmshare/upload_ckpts_to_s3.sbatch"
```

## AWS run (EC2)

```bash
WORK=/opt/edullm/run bash experiments/baseline/aws/setup_venv.sh
source "${WORK}/env.sh"
source "${WORK}/venv/bin/activate"

torchrun --nproc_per_node=1 \
  experiments/baseline/aws/train_olmo_ladder_370m.py \
  --name my-run \
  --paths-file /path/to/train_paths.txt \
  --save-folder /path/to/checkpoints \
  --progress-dir /path/to/progress \
  --length-tokens 5500000000 \
  --device-batch-size 24
```

W&B is hard-disabled in the AWS script. Use the FarmShare script when you need
optional W&B logging or held-out validation loss.
