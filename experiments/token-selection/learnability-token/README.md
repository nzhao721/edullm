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
| Corpus | RegMix 10B (`s3://edullm-datasets/regmix/regmix-10b/`) |
| Arch | `olmo2_370M` (RefHQ-matched) |
| Steps | 2384 (`10e9 // 4_194_304`) |
| Permanent ckpts | `{0, 125, …, 2250, 2384}` — **omit 2375** |
| Eval | full 20-label `task_loss_bpb` on every permanent save |
| run_id | `learnability-token-10b-scratch-v1` |
| S3 export | `s3://edullm-checkpoints/token-sel/learnability-token/` |

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

# 1) Sync tokens into <output_dir>/tokens and freeze order (same as RHO/middle_ppl).
python -m token_selection.scripts.build_token_manifest --config "$CFG"
python -m token_selection.scripts.freeze_order --config "$CFG"

# 2) Validate (requires early/late load_path set to real local .pt files).
python -m token_selection.scripts.validate_experiment --config "$CFG" --olmo-root "$OLMO_ROOT"

# 3) Train — single GPU
export CUDA_VISIBLE_DEVICES=0
NPROC=1 bash experiments/token-selection/learnability-token/launch.sh

# 3b) Train — multi-GPU (world size from NPROC / CUDA_VISIBLE_DEVICES)
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2 bash experiments/token-selection/learnability-token/launch.sh

# Resume
RESUME=1 NPROC=2 bash experiments/token-selection/learnability-token/launch.sh
```

Equivalent direct torchrun:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m token_selection.scripts.train_olmo_template \
  --config "$CFG" --method learnability \
  --olmo-root "$OLMO_ROOT" --launch
```

Task-loss outputs: `task_loss_results/learnability-token/step{N}_task_loss.json`
(or disable with `TASK_LOSS_EVAL=0`).

Do **not** submit AWS training from this arm without an explicit authorized workload.
