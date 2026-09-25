# Provenance of this directory

This is the code that actually produced the two Skill-It 370M runs reported in
the paper. It is a **copy**, vendored here so that the paper's "all code is
available" claim is true from this repository alone.

## Where it came from

| Field | Value |
| --- | --- |
| Source | `OLMo-core/.edullm/` in the two FarmShare run folders below |
| Probe run folder | `/scratch/users/nzhao2/agent-runs/skillit-370m-probe-rerun-20260918-011120` |
| Derivative run folder | `/scratch/users/nzhao2/agent-runs/skillit-370m-deriv-20260916-124719` |
| Upstream repo | `https://github.com/edu-llm/OLMo-core` |
| Nearest upstream commit | `f2ded0b6ba6c6a617d48b346c6e4faab0ad15b5a` on `edullm/skillit-370m` (2026-08-03) |
| Copied on | 2026-09-24 |

The two run folders hold identical code: every file under `OLMo-core/.edullm/`,
the launch scripts in `scripts/` and the data manifest `manifest/ready.json`
have the same SHA-256 in both.

The run copies had Windows (CRLF) line endings from being synced off a laptop.
The files here are converted to LF and are otherwise byte-identical to them. The
one checksum the code enforces, `skillit_math.py`'s `RECIPE_SHA256` over
`skillit_recipe.json`, normalizes line endings first, and the copy here matches
it.

## Relationship to upstream

At the time the study ran, this code existed only as an uncommitted working tree.
It has never been committed upstream. Against `f2ded0b6`, the nearest commit:

- 8 of the 20 files here are identical, including `runpod/entrypoint.py`,
  `skillit_loader.py`, `production_contract/` and the task-loss evaluator.
- 8 differ, including `train_skillit_370m.py`, `skillit_controller.py`,
  `skillit_math.py` and `skillit_recipe.json`.
- 4 do not exist upstream: the no-AWS FarmShare launch path and its handoff note.

No commit on any OLMo-core branch matches the differing files.

The OLMo-core library itself is **not** vendored, because it needs no copy. Both
runs installed `OLMo-core/src` in editable mode, and all 450 files under that
`src/` are identical to `f2ded0b6`. Pin the library from that commit.

## How the runs were launched

Both training jobs were submitted with `sbatch scripts/train_no_aws.sbatch`
(recorded in Slurm accounting). That runs `setup_venv_no_aws.sh`, then
`launch_no_aws.sh`, which starts `runpod/entrypoint.py` under `torchrun`. The
entrypoint calls `train_skillit_370m.py`, and `eval_task_loss_olmo_core.py` is
passed in as the task-loss evaluator. Training data came from the local manifest
`manifest/ready.json` (SHA-256
`176b9e79ee21664aa6ad7533769b48d018ea92a3fc8ac2a1634a63f45971392a`), which
lists the tokenized `pretrain/olmo-127b` v1 shards already on FarmShare scratch.
No S3 access was involved.

| Arm | W&B run (`eduLLM/skillit`) | Slurm job | How it ended |
| --- | --- | --- | --- |
| Offline probe | `87ad0201c4b5781a3df50d7bb394776c` | 1730368 | Cancelled after the step-2384 checkpoint and eval were saved; see `../../mixlaw/skill_dag_370m_wandb_curves.json` for where that eval is recorded |
| Online derivative | `c0844ce36f24d6773c7f45cb31d810f4` | 1728144 | Trained all 2,384 steps and saved the final checkpoint; the job then exited with an error from a W&B `finish(quiet=...)` call |

The W&B run metadata for both names `.edullm/runpod/entrypoint.py` in these run
folders as the program.

## Inputs the recipe pins

`skillit_recipe.json` embeds the two inputs the arms were built from, and
`skillit_math.py` checks their source hashes:

- **Derivative fit:** `experiments/skill-dag/mixlaw/mixlaw_fit_chinchilla.json`.
  The pinned SHA-256 matches the file in this repository.
- **Offline adjacency:** `experiments/skill-dag/skillit/artifacts/probes_full/A_offline.json`.
  The embedded matrix is numerically identical to that file's `A`. The pinned
  file hash does not match either committed version of the file, because its
  other fields differ from the copy that was hashed.

## What is included, and why

Only code on the path that produced the reported runs:

| Path | Role | Against `f2ded0b6` |
| --- | --- | --- |
| `runpod/entrypoint.py` | Entrypoint for both runs | same |
| `train_skillit_370m.py` | Model, optimizer, schedule, checkpoints, eval hooks | differs |
| `skillit_controller.py` | Mid-run Skill-It reweighting | differs |
| `skillit_loader.py` | `WeightedDomainDataLoader`, the time-varying domain sampler | same |
| `skillit_math.py` | Update rule, arms, recipe and source-hash checks | differs |
| `skillit_recipe.json` | Frozen recipe: arms, offline matrix, derivative fit | differs |
| `production_contract/*` | Checkpoint ladder, task-loss callback, W&B artifacts | same |
| `eval_task_loss_olmo_core.py` | The 20-label OLMES evaluator behind the reported bpb numbers | same |
| `requirements-skillit-eval.txt` | Evaluator dependencies installed by the venv setup | same |
| `SKILLIT.md` | Upstream description of the methodology | differs |
| `farmshare/train_no_aws.sbatch`, `launch_no_aws.sh`, `setup_venv_no_aws.sh` | The launch path both jobs used | not upstream |
| `farmshare/common.sh`, `config.env` | Shared settings those scripts source | differs |
| `farmshare/README.md`, `HANDOFF.md` | Notes on the no-AWS launch path | differs / not upstream |

The evaluator here is not the same version as the one vendored for token
selection (`experiments/token-selection/olmo_core_token_selection/`); each copy
is the one its own runs used.

SHA-256 of each file here (LF):

```
ba2de34db6a4cf6156c514a9f5bcf463fbd5343c1f34223d5ac459a5a74cb394  runpod/entrypoint.py
e112c2e42c57be4cc027d3552015b0071da84a69c86359a4e57c9d1617d1bb92  train_skillit_370m.py
eb5c41b0dd41ecabaa18e3743240eb30d6440babd5c30f1abf9ca7bfde421590  skillit_controller.py
34c2c58c84217830e43ac2b3d2b628e970840778d5079bae0a1d6518c2a9a9c2  skillit_loader.py
f8d4c1b8792fa56bbbe0733d1f78942179fbd903dbfdd74cb34ba9e6ce94a186  skillit_math.py
28506e7c3e15814c1dd0082c1c3c52dd228083fec83556616019504583b48f91  skillit_recipe.json
c6d6a9d8292c5f48485e7ae170d913cf368cceccc4a8b6285c86ddbc4eaa7f18  production_contract/__init__.py
67d8a787a45e5a1c9f0f2f7463363b994916e4469cadf12ddfc8e930cd5632af  production_contract/checkpoint.py
f633179af942067f859b561056863ca54b2b32479d8bd082db3f0e1331f25283  production_contract/task_loss.py
638770b96742800c5b75e1f7f13fb8e74b177a5ee5b0dac70ffaf98f909ece5b  production_contract/wandb_artifacts.py
a1fcb8c52ee8f438f69d543ea92da39d1b6d5cebb0c5f6f4ec7d187ca5d1dd32  eval_task_loss_olmo_core.py
6baa90c2da9186bba9dd047c8ec50fadec98a644de431b835028ecb614be4b95  requirements-skillit-eval.txt
ea38b1a8b0d8fb0624a8791213c8cbddfee7acbef6550a9d29c297dd8445de8f  SKILLIT.md
7eeabaa290b8cedcf4223953967cf550f424087529c0c1a5960d5e2da392bf88  farmshare/README.md
21e1d681645f8370ddb6513480b749007b994bb2f55e7c65f19c7c3ea7e8c1e3  farmshare/HANDOFF.md
f8580f58284d0e748a1cea1ef12cd2eab0199496f5477abf8bc00a0b084cd4e6  farmshare/common.sh
f387223dbf3eb6282b5295bfa1da8a7319a9784af574d961e0bee055dc6f42a3  farmshare/config.env
819ad5a5a1b507b3873ba4c93db8bf768851c9f71684b8b11ffd7f51651526de  farmshare/setup_venv_no_aws.sh
57fac534adc10cec4a0eef4bdc2c544f7e1638a8ce9df9b3ab6452dd34a413b6  farmshare/launch_no_aws.sh
16f6e6b1e2d6a9e788b6cd257f01e2e18dbc327bf26ab7a8fd4f912d23b51f4e  farmshare/train_no_aws.sbatch
```

## What is deliberately excluded

- The AWS staging and launch path: `farmshare/launch.sh`, `setup_venv.sh`,
  `stage.sh`, `stage_job.sbatch`, `submit_from_laptop.sh`, `sync_repo.sh`,
  `train_job.sbatch`, `runpod/stage_inputs.py` and `staging_plan.py`. Neither
  reported run went through it.
- RunPod-only scaffolding: `runpod/bootstrap.sh`, `runpod/launch.sh`,
  `runpod/README.md` and `Dockerfile`. Both runs were on FarmShare.
- `skillit_entrypoint.py` and `fixtures/`: platform-submission adapters the
  runs did not use.
- `train_on_corpus.py`: the general-purpose corpus trainer, which is not on the
  Skill-It path.
- `tests/` and `rehearsal.md`: they produced no reported number.
- The OLMo-core library (`src/`): identical to `f2ded0b6`, as noted above.

## Caveat

The code has no upstream commit, and the W&B runs record no git commit. So the
correspondence between this copy and the runs rests on two things: the program
path W&B recorded for each run, and the two run folders holding identical copies
of this code when it was copied here. Nothing records whether those folders were
edited after the jobs started.
