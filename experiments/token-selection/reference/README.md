# Reference arm — RefHQ RegMix 5.5B CE

Plain CE on published RefHQ (`pretrain/refhq-regmix-5p5b`) with the
RefHQ-matched OLMo-2 370M stack. Produces the reference ladder consumed by
downstream token-selection arms (via DistCP → `export_refhq_reference.py`).

| Knob | Value |
|------|-------|
| Architecture | `TransformerConfig.olmo2_370M` (full attn) |
| GBS / seq / LR | `4_194_304` / 2048 / `4e-4` (warmup 24, `alpha_f=0.1`) |
| Token budget | published train rows (~5.509B → ≈1313 steps) |
| Dataset | `pretrain/refhq-regmix-5p5b` (`s3://edullm-data/`, latest validated) |
| S3 export | `s3://edullm-checkpoints/token-sel/reference/` (`S3_EXPORT=0` to disable) |

## Files

| File | Role |
|------|------|
| [`prepare_refhq_data.py`](prepare_refhq_data.py) | Resolve + stage edullm-data shards → path list |
| [`train_olmo3_370m_refhq.py`](train_olmo3_370m_refhq.py) | Trainer (1..N GPU via `torchrun`) |
| [`launch_train.sh`](launch_train.sh) | Stage/train + upload-before-end sync |
| [`export_refhq_reference.py`](export_refhq_reference.py) | DistCP → flat `model.pt` for other arms |

## Export flat reference weights

Default DistCP URI (trainer `NAME=refhq-regmix-5p5b-v1`, final step ≈1313):

`s3://edullm-checkpoints/token-sel/reference/checkpoints/refhq-regmix-5p5b-v1/step1313/`

```bash
python experiments/token-selection/reference/export_refhq_reference.py \
  --work-dir /scratch/refhq-export \
  --output /scratch/refhq_step1313.pt
# override: --s3-uri s3://edullm-checkpoints/token-sel/reference/checkpoints/<NAME>/step<N>/
```

## Ephemeral runtime

Scratch starts empty and may be wiped when the job ends.

- **Allowed:** stage shards from `s3://edullm-data` into scratch for the job.
- **Forbidden:** assuming FarmShare/laptop corpora, legacy `s3://edullm-datasets/`,
  or persistent local checkpoints already on disk.
- **Durable:** permanent DistCP steps + progress upload to
  `s3://edullm-checkpoints/token-sel/reference/` (live after each ladder save,
  plus a final sync in `launch_train.sh`). Upload failure aborts the job
  (fail-closed). `S3_EXPORT=0` is local-smoke opt-out only.

Resume: sync the needed `step*` prefix from S3 into a fresh `SAVE_FOLDER` (or
pass `--load-path` to a locally materialized DistCP dir). Do not rely on an
old scratch tree surviving between jobs.

## Prepare (optional; trainer can stage itself)

```bash
python experiments/token-selection/reference/prepare_refhq_data.py \
  --work /scratch/refhq
# → tokenized/paths_train.txt, length_tokens.txt, refhq_data_summary.json
```

## Launch

```bash
# Clean ephemeral node: stage + train + S3 export
STAGE_DIR=/scratch/staged \
SAVE_FOLDER=/scratch/ckpts/refhq-regmix-5p5b-v1 \
PROGRESS_DIR=/scratch/progress/refhq-regmix-5p5b-v1 \
NPROC=4 \
bash experiments/token-selection/reference/launch_train.sh

# Or via prepare:
REFHQ_WORK=/scratch/refhq \
SAVE_FOLDER=/scratch/ckpts/refhq-regmix-5p5b-v1 \
PROGRESS_DIR=/scratch/progress/refhq-regmix-5p5b-v1 \
bash experiments/token-selection/reference/launch_train.sh

# Train from s3:// URIs (no local stage):
STAGE_DIR= \
SAVE_FOLDER=/scratch/ckpts/refhq-regmix-5p5b-v1 \
PROGRESS_DIR=/scratch/progress/refhq-regmix-5p5b-v1 \
bash experiments/token-selection/reference/launch_train.sh
```

Requires the `edullm-data` package and AWS read access to `s3://edullm-data`
(plus write to `edullm-checkpoints` when `S3_EXPORT` is on). Does **not** submit
AWS jobs from this wrapper.
