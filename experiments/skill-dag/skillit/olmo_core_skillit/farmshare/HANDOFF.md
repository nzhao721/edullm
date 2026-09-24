# Skill-It OLMo2-370M — FarmShare handoff

Run one Skill-It arm (`probe` or `deriv`) on FarmShare at 4×L40S. This bundle
is self-contained: no AWS/S3 access, no GitHub access, and no dataset
download are required. The training corpus is read directly from a shared
collaborator's FarmShare scratch over the cluster's shared filesystem — you
only need read access to one already-open path, not a copy of the data.

## What you need before starting

- A FarmShare account with an open SSH control socket (`/tmp/farmshare-<your-sunet>.sock`).
  If it's not open, run this yourself and complete Duo/password interactively —
  never hand your password or Duo code to an agent:
  ```bash
  ssh -M -S /tmp/farmshare-<your-sunet>.sock -o ControlPersist=yes <your-sunet>@login.farmshare.stanford.edu
  ```
- Your own `WANDB_API_KEY` with write access to the `skillit` W&B project
  under the `eduLLM` entity (ask nzhao2 to add you), or point at your own
  W&B project if you'd rather log separately.
- Which arm index to run: **coordinate with nzhao2 first** so you don't both
  submit the same arm. `0` = probe, `1` = deriv.

Nothing else. Do not attempt `sb-aws-creds login` or any AWS setup — this
path never touches AWS.

## 0. If your agent uses Claude Code: install the FarmShare operating skill

This bundle ships a copy of the `operate-farmshare` Claude Code skill under
`skills/operate-farmshare/` — the same conventions (control-socket access,
never handling passwords/Duo, safety boundaries, minimal job sizing) used to
prepare this handoff. If your agent runs in Claude Code, copy it into your
own skills directory so it picks up the same conventions automatically:

```bash
mkdir -p ~/.claude/skills
cp -r /path/to/unzipped/skillit-handoff/skills/operate-farmshare ~/.claude/skills/operate-farmshare
```

This is optional — every command below is a plain `ssh`/`sbatch` call that
works regardless of whether the skill is installed.

## 1. Verify FarmShare access

```bash
ssh -S /tmp/farmshare-<your-sunet>.sock -o BatchMode=yes <your-sunet>@login.farmshare.stanford.edu 'hostname'
```

## 2. Verify you can actually read the shared dataset

Before unpacking anything, confirm the shared path is readable from your
account (it was opened for exactly this purpose — traversal-only, not
listable, so `ls` on the parent won't show anything, but a direct read of a
known file will work):

```bash
ssh -S /tmp/farmshare-<your-sunet>.sock -o BatchMode=yes <your-sunet>@login.farmshare.stanford.edu \
  'head -c 16 /scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/publish-stage/tokens/dclm/train-00000.u32le.bin | xxd'
```

If this fails with permission denied, stop and ask nzhao2 to check the
permissions on that path before continuing — do not attempt to work around
it by copying the dataset yourself (it's ~507 GB and this path is meant to
avoid exactly that transfer).

## 3. Unpack this bundle into a scratch run directory

```bash
SUNET=<your-sunet>
RUN_DIR="/scratch/users/${SUNET}/agent-runs/skillit-370m-$(date -u +%Y%m%d-%H%M%S)"
ssh -S /tmp/farmshare-${SUNET}.sock -o BatchMode=yes ${SUNET}@login.farmshare.stanford.edu \
  "mkdir -p '${RUN_DIR}' && chmod 700 '${RUN_DIR}'"

# From wherever this bundle was unzipped locally, copy it up:
tar -C /path/to/unzipped/skillit-handoff -czf - . | \
  ssh -S /tmp/farmshare-${SUNET}.sock -o BatchMode=yes ${SUNET}@login.farmshare.stanford.edu \
  "tar -xzf - -C '${RUN_DIR}'"
```

This gives you `${RUN_DIR}/OLMo-core/` (code), `${RUN_DIR}/scripts/`
(FarmShare launch scripts), and `${RUN_DIR}/manifest/ready.json` (the
dataset manifest, already pointing at absolute paths on nzhao2's scratch —
do not edit it).

## 4. Submit the training job

Run this directly on FarmShare (via your socket), not from a laptop —
everything here is self-contained on the cluster:

```bash
ssh -S /tmp/farmshare-${SUNET}.sock -o BatchMode=yes ${SUNET}@login.farmshare.stanford.edu bash -s <<EOF
set -Eeuo pipefail
RUN_DIR="${RUN_DIR}"
chmod +x "\${RUN_DIR}/scripts"/*.sh
mkdir -p "\${RUN_DIR}/logs"
JOB=\$(sbatch --parsable \
  --export=ALL,RUN_DIR="\${RUN_DIR}",ARM_INDEX=<0 or 1>,RECOVERY_MODE=fresh,EDULLM_RUNPOD_INPUT_MANIFEST="\${RUN_DIR}/manifest/ready.json",WANDB_API_KEY="<your key>" \
  --chdir="\${RUN_DIR}" \
  "\${RUN_DIR}/scripts/train_no_aws.sbatch")
echo "submitted job \${JOB}"
EOF
```

Replace `<0 or 1>` and `<your key>` before running. `train_no_aws.sbatch`
already requests 4 GPUs / 32 CPUs / 192G / 20h — matched to FarmShare's
per-user GPU quota and this arm's measured runtime. Don't increase these
without a reason; over-requesting mainly means a longer wait in the queue.

## 5. Monitor

```bash
ssh -S /tmp/farmshare-${SUNET}.sock -o BatchMode=yes ${SUNET}@login.farmshare.stanford.edu \
  "squeue --me; tail -n 50 ${RUN_DIR}/logs/train-*.out"
```

Success looks like: the job stays `RUNNING` past the first few minutes (past
venv setup + dataset resolution), a W&B run appears under the `skillit`
project (or your own, if you pointed elsewhere) named
`skillit-<arm>-farmshare-<timestamp>`, and checkpoints start appearing under
`${RUN_DIR}/runs/<arm>/checkpoints/` (via the `RUN_ROOT`/`arm_root` path — the
job's own logs will show the exact path).

If `sacct` shows the job failed immediately with a manifest or permission
error, re-check step 2 before re-submitting.

## 6. Resume after an interruption

Re-submit with the same `RUN_DIR` and `RECOVERY_MODE=resume` in the
`--export` list instead of `fresh`.

## Known fragility, on purpose

This bundle intentionally does not include the dataset (~507 GB) — that
would defeat the point of a same-cluster handoff. Training reads directly
from `/scratch/users/nzhao2/agent-runs/olmo127b-edullm-publish-20260730T233445Z/publish-stage/tokens/`
for the lifetime of your job. If nzhao2 ever deletes, moves, or re-restricts
that path, your job will fail on its next read. This is a soft dependency on
another student's scratch quota and retention policy, not a permanent
data source — if you need this to be durable long-term, copy the referenced
files to your own scratch and regenerate `manifest/ready.json` with the new
paths (same schema: `{"schema_version":1,"family":"skillit","dataset_id":"pretrain/olmo-127b","dataset_version":"v1","domains":[{"name":..., "objects":[{"path":..., "size":...}, ...]}, ...]}`,
domains in this exact order: dclm, arxiv, starcoder, pes2o, open-web-math,
algebraic-stack, wiki).

## Reference

See `OLMo-core/.edullm/SKILLIT.md` for the full experiment methodology, and
`OLMo-core/.edullm/farmshare/README.md` for the general FarmShare adapter
this bundle's scripts are drawn from.
