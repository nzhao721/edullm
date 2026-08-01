# Learnability token arm

Top-60% token selection by frozen RefHQ learnability score
`score = L_early − L_late` (larger = larger early→late improvement).

Near-clone of [`../learnability-doc/`](../learnability-doc/): same polarity
(early−late, top 60%) and keep fraction, but this arm scores **tokens online**
with dual frozen RefHQ refs; the doc arm filters offline then trains plain CE.

| Knob | Value |
|------|--------|
| Method | `learnability` |
| Keep fraction `k` | 0.6 |
| Selection warmup | `t0_steps=0` / `t0_frac=0` (active from step 0) |
| Early ref | RefHQ step250 |
| Late ref | mean of RefHQ step1000 / 1125 / 1315 |
| Corpus | `pretrain/regmix-10b` (`data.dataset_id` → `s3://edullm-data/`; stage per job) |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Steps | **2360** (`9900000000 // 4_194_304`) |
| Permanent ckpts | `{0, 125, …, 2125, 2360}` — **omit 2250** |
| Eval | full 20-label `task_loss_bpb` on every permanent save |
| run_id | `learnability-token-10b-scratch-v1` |
| Artifact durability | Runtime scratch + W&B |

Config: [`configs/run_learnability_10b.yaml`](configs/run_learnability_10b.yaml).

## Score polarity

- `L_early` / `L_late` = per-token CE under the corresponding **frozen** RefHQ weights.
- `score = L_early − L_late`. Tokens the late model finds much easier than the early
  model get large positive scores and are kept (top 60% per sequence).
- Do **not** flip the sign; that would select tokens that got *worse*.

## Reference export

From `experiments/token-selection` (needs `olmo_core` + local/S3 DistCP access).
Committed YAML keeps `reference.early/late.load_path: null` until export (fail-closed).

```bash
python learnability-token/export_learnability_refs.py \
  --work-dir /tmp/learnability-refs-work \
  --out-dir /path/to/refs
# → refhq_step250_early.pt
# → refhq_late_avg_1000_1125_1315.pt
```

Then set in a runtime YAML (or patch the arm config):

```yaml
reference:
  early:
    load_path: /path/to/refs/refhq_step250_early.pt
  late:
    load_path: /path/to/refs/refhq_late_avg_1000_1125_1315.pt
```

Single-step export (reuse RHO helper):

```bash
python reference/export_refhq_reference.py \
  --s3-uri s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step250/ \
  --work-dir /tmp/refhq-step250 --output /path/to/refs/refhq_step250_early.pt
```

## Data prep + launch

```bash
export PYTHONPATH=experiments/token-selection
export OLMO_ROOT=/path/to/OLMo-core
CFG=experiments/token-selection/learnability-token/configs/run_learnability_10b.yaml

# 1) Optional preflight: stage tokens+order from edullm-data (also done by --launch).
python -m token_selection.scripts.validate_experiment --config "$CFG" --stage

# 2) Train — single GPU (--launch stages corpus + materializes S3 refs)
export CUDA_VISIBLE_DEVICES=0
NPROC=1 bash experiments/token-selection/learnability-token/launch.sh

# 2b) Train — multi-GPU (world size from NPROC / CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2 bash experiments/token-selection/learnability-token/launch.sh

# Resume from W&B when local save is empty
RESUME=1 WANDB_RESUME_ARTIFACT=entity/project/run-checkpoint:latest \
  NPROC=2 bash experiments/token-selection/learnability-token/launch.sh
```

Equivalent direct torchrun:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m token_selection.scripts.train_olmo_template \
  --config "$CFG" --method learnability \
  --olmo-root "$OLMO_ROOT" --launch
```

Task-loss outputs: `task_loss_results/learnability-token/step{N}_task_loss.json`
(or disable with `TASK_LOSS_EVAL=0`). Durable checkpoints live under
W&B checkpoint artifacts — do not rely on scratch surviving across jobs.

Do **not** submit AWS training from this arm without an explicit authorized workload.
