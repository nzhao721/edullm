# RHO-1 (rho_excess) arm

Top-60% token selection by RHO-1 excess loss `L_curr − L_ref` on RegMix 10B.

| Knob | Value |
|------|-------|
| Method | `rho_excess` (shared `token_selection` package) |
| Keep rate `k` / γ | 0.6 (top 60%) |
| Selection warmup | `t0_steps=0` / `t0_frac=0` (active from step 0) |
| Arch | `TransformerConfig.olmo2_370M` (RefHQ-matched) |
| GBS / microbatch | `4_194_304` / `65_536` (tune microbatch for GPU memory) |
| LR / warmup / `alpha_f` | `4e-4` / 24 / 0.1 |
| Compile | `compile_model=true` (YAML + FarmShare default) |
| Steps | **2360** (`9900000000 // 4_194_304`; one epoch under ~9.989B published train) |
| Permanent ckpts | `{0, 125, …, 2125, 2360}` (omit 2250); `max_checkpoints=None`; no ephemeral |
| Frozen reference | RefHQ 370M **step1315** |
| `run_id` | `rho-1-regmix10b-v1` |
| Train corpus | `pretrain/regmix-10b` via `s3://edullm-data/` (stage each job) |
| Artifact durability | Runtime scratch + W&B |

**Discarded:** `rho-excess-10b-scratch-v1` (~step200). Do not resume it. GPU retrain of the new `run_id` is still required.

**Ephemeral scratch:** start empty; stage tokens from edullm-data; keep run
artifacts on scratch and upload them to W&B. Do not rely on FarmShare/laptop
corpora, local venvs, or wiped save folders. For `RESUME=1` on empty scratch,
set `WANDB_RESUME_ARTIFACT`.

Shared package: [`../token_selection/`](../token_selection/).  
Config: [`configs/run_rho_10b.yaml`](configs/run_rho_10b.yaml).

## Reference load path contract

`FrozenReference` refuses remote `s3://` URIs. Export a local flat `model.pt` first, or leave `reference.load_path` null and let `--launch` auto-materialize from `reference.s3_uri`:

```bash
python experiments/token-selection/reference/export_refhq_reference.py \
  --s3-uri s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/ \
  --work-dir /path/to/ref_work \
  --output /path/to/refhq_step1315_model.pt
```

Then set `reference.load_path` in the runtime YAML (FarmShare / `launch.sh` do this automatically) to that `.pt` path (or a directory containing `model.pt`).

Provenance (not loaded by the trainer):

- S3: `s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/`
- Planned end of RefHQ 5.5B CE (`planned_total_steps=1314`)

## Launch (hardware-agnostic)

World size comes from `torchrun --nproc_per_node` / Slurm / `CUDA_VISIBLE_DEVICES`.  
GBS must divide evenly by `world_size * rank_microbatch_size`. No hardcoded device IDs.

### Local / any host

```bash
export PYTHONPATH=experiments/token-selection
export CUDA_VISIBLE_DEVICES=0   # or 0,1,... ; required unless Slurm sets it
export OLMO_ROOT=/path/to/OLMo-core
export REFERENCE_LOAD_PATH=/path/to/refhq_step1315_model.pt
export TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT=experiments/token-selection/rho-1/farmshare/enqueue_task_loss.sh

# Convenience launcher (writes a runtime YAML with reference.load_path filled)
NPROC=1 bash experiments/token-selection/rho-1/launch.sh

# Or: prepare manifest + order + validate, then torchrun directly
# (fill reference.load_path in a runtime copy of configs/run_rho_10b.yaml first)
python -m token_selection.scripts.build_token_manifest --config $RUNTIME_CFG
python -m token_selection.scripts.freeze_order --config $RUNTIME_CFG
python -m token_selection.scripts.validate_experiment --config $RUNTIME_CFG --olmo-root $OLMO_ROOT
torchrun --standalone --nproc_per_node=$NUM_GPUS \
  -m token_selection.scripts.train_olmo_template \
  --config $RUNTIME_CFG \
  --method rho_excess \
  --olmo-root $OLMO_ROOT \
  --launch
```

Crash resume of **this** `run_id` only: `RESUME=1 bash rho-1/launch.sh` (fingerprint-gated; fetches from S3 if local empty). Never point at `rho-excess-10b-scratch-v1`.

### FarmShare

```bash
export RUN_DIR=/scratch/users/$USER/agent-runs/rho-1-regmix10b-v1
export EDULLM_ROOT=/path/to/edullm
export NUM_GPUS=4                    # or 1; discovered from Slurm if unset
export RANK_MICROBATCH_SIZE=16384    # tune for GPU memory
export FROM_SCRATCH=1                # default for the rebuild
# Push aws-session.env into RUN_DIR for S3 stage/export (FarmShare cannot sb-aws-creds login)

bash "$EDULLM_ROOT/experiments/token-selection/rho-1/farmshare/run_rho_train.sh" prepare
cd "$RUN_DIR"
sbatch --exclude=wheat-01 --gres=gpu:${NUM_GPUS} \
  "$EDULLM_ROOT/experiments/token-selection/rho-1/farmshare/train_rho.sbatch"
```

Or: `bash .../farmshare/submit_scratch.sh`. Crash resume: `RESUME=1 bash .../submit_scratch.sh`.

Helpers live under [`farmshare/`](farmshare/).

## Task-loss eval

On every permanent checkpoint save, `TaskLossEvalCallback` enqueues the full 20-label OLMo-ladder `task_loss_bpb` suite and uploads the step dir to S3. Outputs:

`experiments/token-selection/task_loss_results/rho-1/step{N}_task_loss.json`

(or under `$RUN_DIR` when `output_dir` / results_dir are rewritten).

Set `TOKEN_SELECTION_TASK_LOSS_EVAL_SCRIPT` or `eval.task_loss.command_template` so the enqueue has a launcher (see `farmshare/enqueue_task_loss.sh`).

## Controlled stack (shared spine)

YAML + `train_olmo_template` / `TokenSelectTrainModule` match RefHQ on
`z_loss_multiplier=1e-5`, `max_grad_norm=1.0`, and
`torch.set_float32_matmul_precision("high")`. CE+z are folded over the same
token-selection mask.
