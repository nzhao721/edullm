# Token selection



Train **OLMo-2 370M from scratch** on a shared ~10B RegMix corpus with online

token selection. This pass writes **checkpoints + train metrics** only;

held-out eval comes later.



## Methods (what the code actually supports)



| Method | After warmup | Scorer | Status |

|--------|--------------|--------|--------|

| `full` | loss on all valid tokens | none | implemented; no dedicated 10B config yet |

| `rho_excess` | keep top `k` by `excess = L_curr − L_ref` | frozen reference checkpoint | code + config present; **not launch-ready** until `reference.load_path` and a GPU are set |

| `middle_ppl` | keep middle `k` by `L_curr` (CE ≈ log-PPL) | none (reuses train-forward CE) | code + config present; not launched |



Defaults for selecting methods: `k=0.6`, `t0_frac=0.02` (~48 of ~2384 steps).



Not implemented yet (P1 leftovers): random-k, raw-loss (top by CE),

attention, learnability, dynamic reference. Add them as new method plugs +

separate configs — do not retarget an existing arm’s YAML.



## Arm isolation



One shared spine (`TokenSelectTrainModule` + scorers). Arms stay separate by

**config identity**, not git branches:



| Arm | Config | `run_id` | `output_dir` | S3 `prefix` |

|-----|--------|----------|--------------|-------------|

| RHO | `configs/run_rho_10b.yaml` | `rho-excess-10b-scratch-v1` | `token_selection/data/rho_10b` | `token-selection/rho-excess-10b-scratch-v1` |

| middle_ppl | `configs/run_middle_ppl_10b.yaml` | `middle-ppl-10b-scratch-v1` | `token_selection/data/middle_ppl_10b` | `token-selection/middle-ppl-10b-scratch-v1` |



**GPU pins are host-specific**, not part of scientific identity. Before

`--launch`, set `train.cuda_visible_devices` to an idle index on the machine

you are using (and the same value in `CUDA_VISIBLE_DEVICES`).



Do not reuse another arm’s `run_id`, `output_dir`, or S3 `prefix`. Sync/upload

always use **that config’s** prefix.



## Layout



```

token_selection/

  configs/           # one YAML per scientific run

  olmo_ext/          # EMA, FrozenReference, scorers, TrainModule, metrics

  scripts/           # sync, manifest, freeze_order, validate, train

  tests/

  data/              # local only (gitignored); real artifacts live on S3

```



## Corpus



Shared read-only input: `s3://edullm-dataset-regmix/regmix-10b/tokenized`

(`allenai/dolma2-tokenizer`, EOS `100257`). Seven domain shards are **raw

headerless** `uint32` arrays (despite `.npy`); each has a JSON sidecar. There is

**no** `manifest.json` in the bucket — `build_token_manifest` derives it locally

and verifies sizes/tokenizer/EOS.



Usable one-epoch budget after per-shard sequence truncation: **10,004,799,488**

tokens. Config `max_tokens: 10_000_000_000` fits; preflight rejects a budget that

would wrap into a second epoch. `data.tokenizer` must match the corpus or the

embedding table is wrong.



## How to run a 10B arm



```bash

cd <repo-root>
export PYTHONPATH=experiments/token-selection

pip install -r experiments/token-selection/token_selection/requirements.txt

# Pin OLMo-core (required revision is in each YAML under olmo_core.revision):

#   git clone https://github.com/edu-llm/OLMo-core /opt/OLMo-core

#   git -C /opt/OLMo-core checkout <revision>

#   pip install -e /opt/OLMo-core



CFG=token_selection/configs/run_middle_ppl_10b.yaml   # one config per arm; never retarget a finished run_id

METHOD=middle_ppl                                     # must be listed in that config's methods:



python -m token_selection.scripts.sync_artifacts \

  --config "$CFG" --direction download --what tokens

python -m token_selection.scripts.build_token_manifest --config "$CFG"

python -m token_selection.scripts.freeze_order --config "$CFG"

python -m token_selection.scripts.validate_experiment \

  --config "$CFG" --olmo-root /opt/OLMo-core



# Launch: set train.cuda_visible_devices in the YAML to an idle GPU on this host,

# then pin the same index in the environment before Python starts.

CUDA_VISIBLE_DEVICES=<gpu> python -m torch.distributed.run --standalone \

  --nproc_per_node=1 -m token_selection.scripts.train_olmo_template \

  --config "$CFG" --method "$METHOD" --olmo-root /opt/OLMo-core --launch



# Resume after crash / timeout (fingerprint-guarded):

CUDA_VISIBLE_DEVICES=<gpu> python -m torch.distributed.run --standalone \

  --nproc_per_node=1 -m token_selection.scripts.train_olmo_template \

  --config "$CFG" --method "$METHOD" --olmo-root /opt/OLMo-core --launch --resume



python -m token_selection.scripts.sync_artifacts \

  --config "$CFG" --direction upload --what metrics

python -m token_selection.scripts.sync_artifacts \

  --config "$CFG" --direction upload --what checkpoints

```



Fresh scratch refuses a non-empty checkpoint dir; dataset cache lives **outside**

that dir so a failed build can relaunch. `model.load_path` must stay null

(scratch only). Extending `max_tokens` upward on `--resume` is allowed; changing

seed, arch, order, `k`, reference **bytes**, etc. is not. Relocating

`reference.load_path` across hosts is allowed when `reference_content_sha256`

still matches (Farmshare resume from an AWS run).



**Hardware knob:** `rank_microbatch_size` (default 65536 tokens) is not in the

run fingerprint. Selecting methods that need a second scoring forward (RHO)

and full logits: if the first selecting step OOMs, halve it and `--resume`.

`middle_ppl` has no second forward, so it is lighter at the same microbatch.

Tune `rank_microbatch_size` for your GPU memory (e.g. 16384 on 48 GiB, 65536 on 80 GiB).



## RHO (`rho_excess`)



After warmup, keep top-`k` by `excess = L_curr − L_ref`. Reference is a weight

shadow (`FrozenReference.swap_to`), loaded once from a **local**

`reference.load_path` (`.pt`/`.pth` or a dir with `model.pt`); `s3://` URIs are

refused. Resume fingerprints reference **bytes** (and allows the local path to

move). Preflight refuses a missing path.



Until `reference.load_path` and `train.cuda_visible_devices` are set,

`run_rho_10b.yaml` will not pass validate/launch. Export the RefHQ reference

checkpoint with `experiments/token-selection/reference/aws/export_refhq_reference.py`,

then point `reference.load_path` at the resulting `model.pt`.



```bash

python -m token_selection.scripts.train_method \

  --config token_selection/configs/run_rho_10b.yaml --method rho_excess

```



## Middle perplexity (`middle_ppl`)



Same corpus contract / `k` / `t0_frac` as RHO. After warmup, per sequence

keep the middle `k` of valid tokens by current-model CE (`L_curr` ≈

log-perplexity): drop the easiest and hardest `(1−k)/2` each. No reference —

the score is the train-forward CE already computed for the loss.

Permanent checkpoints every 250 steps (plus T0 milestone).



```bash

python -m token_selection.scripts.train_method \

  --config token_selection/configs/run_middle_ppl_10b.yaml --method middle_ppl

```



## Metrics



Rank-zero schema-v2 train ledger under `metrics/<method>/`: compute counters plus

per-step CE, `selected_frac`, `alpha`, and `mean_rel_kept` /

`mean_rel_dropped` (name is historical; RHO writes excess means and middle_ppl

writes CE means into the same fields). On `--resume`, post-checkpoint rows are

truncated before logging continues. `compare_runs.py` exits until a shared eval

protocol exists.



## Stack



- **Tokens (RO):** `s3://edullm-dataset-regmix/regmix-10b/tokenized`

- **Outputs:** `s3://edullm-dataset-olmo/<prefix>/metrics`,

  `s3://edullm-checkpoints/<prefix>/` (`sbsandbox`, `us-east-1`)

- **Train:** [edu-llm/OLMo-core](https://github.com/edu-llm/OLMo-core) @ pinned

  revision + `TokenSelectTrainModule`

- **GPU:** set `train.cuda_visible_devices` and launch with `torchrun

  --nproc_per_node=N`. Global batch size is fixed; `NUM_GPUS * rank_microbatch_size`

  must divide `global_batch_size`.

- **Attn backend:** `train.attn_backend: auto` (or env `OLMO_ATTN_BACKEND`)

  selects FlashAttention-2 when `flash_attn` works, else torch SDPA. Kernel-only;

  not part of the run fingerprint.



## RHO reference checkpoint



RHO reference is the RefHQ 5.5B CE final planned checkpoint

(`…/edullm-370M-refhq-5p5b/checkpoints/step1315/`), unsharded to `model.pt` by

`experiments/token-selection/reference/aws/export_refhq_reference.py`.


