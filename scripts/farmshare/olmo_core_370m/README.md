# OLMo-core 370M experiments on FarmShare

FarmShare bootstrap lives in each OLMo-core branch under `.edullm/farmshare/`
(local adapters, same role as `.edullm/runpod/`).

| Experiment | Worktree | Submit |
|------------|----------|--------|
| Skill-It | `OLMo-core-skillit-370m` | `ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh` |
| MixLaw | `OLMo-core` | `ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh` |
| Curriculum | `OLMo-core-curriculum-370m` | `ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh` |
| Token selection | `OLMo-core-token-selection-370m` | `ARM=attention bash .edullm/farmshare/submit_from_laptop.sh` |

## Shared prerequisites

1. FarmShare SSH control socket (`/tmp/farmshare-nzhao2.sock`)
2. `edullm` repo for session push helpers:
   - `scripts/farmshare/push_aws_session_to_farmshare.sh`
   - `scripts/farmshare/push_wandb_session_to_farmshare.sh`
3. W&B key file at `/mnt/c/Users/natha/.wandb_api_key` (or `WANDB_API_KEY`)

Submit scripts mint AWS on the laptop, sync branch code to scratch, stage sealed
`edullm-data` inputs via Slurm, delete credentials, then launch 8-GPU training
with PyTorch SDPA (no FlashAttention).

Override queue sizing before submit:

```bash
export TRAIN_GPUS=8 TRAIN_CPUS=64 TRAIN_MEM=384G TRAIN_TIME=72:00:00
export STAGE_CPUS=8 STAGE_MEM=32G STAGE_TIME=06:00:00
```

See each branch's `.edullm/farmshare/README.md` for arm indices and recovery.
