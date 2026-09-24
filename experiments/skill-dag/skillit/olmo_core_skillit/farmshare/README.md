# Skill-It on FarmShare (4 × L40S)

Runs 4 data-parallel ranks, not RunPod's production 8 — FarmShare's per-user
GPU quota caps at 4 concurrent GPUs. Both launch paths below always pass
`--allow-local-only`, so these are non-production runs by this repo's own
contract (skips the exactly-8-rank assertion and the forced
`--wandb-mode=online` requirement). Training, the Skill-It controller,
checkpoints, and the full task-loss eval suite are unaffected — see
`SKILLIT.md` for the exact distinction. `TRAIN_TIME=20:00:00` below is a
~16h expected wall-clock run with headroom already applied; treat it as
provisional until a real run confirms it, the same way RunPod's
`maximum_runtime_hours` is set from a measured benchmark rather than assumed.

## Current path: no AWS (`launch_no_aws.sh` / `train_no_aws.sbatch`)

The `edullm-data`/AWS staging path below is stale — sealed S3 access is no
longer used. `launch_no_aws.sh` reads training data straight from an
already-existing local `ready.json` manifest (`EDULLM_RUNPOD_INPUT_MANIFEST`);
`setup_venv_no_aws.sh` skips the `boto3`/`edullm-data` installs entirely,
since `.edullm/runpod/entrypoint.py`'s `resolve_local_datasets` never imports
them. No AWS session, no staging job, no S3 access anywhere in this path. See
`SKILLIT.md`'s "FarmShare deployment" section for where the real local copy
of `pretrain/olmo-127b/v1` currently lives and how its permissions are scoped.

To hand this off to a second student on their own FarmShare account, use
`HANDOFF.md` plus the packaged bundle (code + this manifest + these scripts,
no dataset — the manifest points at absolute paths on the first student's
scratch, readable cluster-wide without a transfer). Submission on the second
account is direct `sbatch` against `train_no_aws.sbatch` from their own
socket — see `HANDOFF.md` for exact commands; it does not go through
`submit_from_laptop.sh` below.

## Legacy AWS-staging path (stale, kept for reference)

Mirrors the RunPod adapter: stage sealed `s3://edullm-data/` inputs with a
temporary AWS session, delete credentials, then train with PyTorch SDPA (no
FlashAttention). The stager selects deterministic per-domain shard prefixes
from the initial weights with 25% headroom; later adaptive weight shifts
sample from those bounded local pools. Do not use this path — AWS access to
this dataset is no longer available.

### Quick start (engineer laptop + WSL)

```bash
cd /mnt/c/alpha_ai/OLMo-core-skillit-370m
ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh
```

Probe is `ARM_INDEX=0`, derivative is `ARM_INDEX=1`.

Prerequisites:

- FarmShare control socket (`/tmp/farmshare-nzhao2.sock`)
- `edullm` repo at `/mnt/c/alpha_ai/edullm` for `push_aws_session_to_farmshare.sh`
  and `push_wandb_session_to_farmshare.sh`
- W&B key file at `/mnt/c/Users/natha/.wandb_api_key` (or `WANDB_API_KEY` in env)

### Running both arms under separate FarmShare accounts (legacy path only)

To run `probe` and `deriv` in parallel across two independent per-user GPU
quotas, override `FARMSHARE_SUNET` per submission — `RUN_DIR`, `HOST`, and the
default socket path all derive from it:

```bash
ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh                    # your own account
FARMSHARE_SUNET=studentb ARM_INDEX=1 bash .edullm/farmshare/submit_from_laptop.sh
```

This only works if that account's control socket is reachable from the
machine running `submit_from_laptop.sh` — see the FarmShare operating skill's
"Establish access" convention if the second student is on a different
machine. For the current no-AWS path, use `HANDOFF.md` instead: the second
account submits directly on FarmShare via its own socket, no laptop relay
needed.

## Resource defaults

Override before submit:

```bash
export TRAIN_GPUS=4 TRAIN_CPUS=32 TRAIN_MEM=192G TRAIN_TIME=20:00:00
export STAGE_CPUS=8 STAGE_MEM=32G STAGE_TIME=06:00:00
```

Jobs always exclude `wheat-01`.

## Recovery

Re-submit with the same `RUN_DIR` and `RECOVERY_MODE=resume` (or `retry-start`
before the first checkpoint). The legacy AWS path additionally needs a fresh
`aws-session.env` pushed before restaging; the no-AWS path never needs this.

## W&B policy

Every permanent checkpoint uploads full 20-task eval metrics to W&B. Only the
final checkpoint is uploaded as a model artifact (branch code).
