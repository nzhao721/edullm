# SmolLM2-135M Co-LMLM training on RunPod

This is the RunPod copy of the FarmShare SmolLM2-135M DDP flow. It trains on
the self-contained ModernBERT annotation shards at:

`s3://edullm-checkpoints/runpod/colmlm-annotate/output/`

The original text is unchanged. For every annotated half-open character span
`[char_start, char_end)`, every SmolLM2 token whose offset intersects the span
gets label `-100`. This includes tokens crossing either boundary. No `<FACT>`
markers are inserted.

## Data contract

- All 19 annotation shards and their manifests are downloaded at startup.
- `pretrain/fineweb-edu-1b/v6` metadata is downloaded for provenance checks.
- The v6 vendor/raw group is not joined to the annotations: `dataset.json`
  explicitly says its raw and token groups are not document-aligned.
- `prepare_annotated_corpus.py` tokenizes the aligned text embedded in each
  annotation record and writes fixed 2048-token `uint32` shards plus parallel
  `uint8` loss-mask shards to local scratch.

## Credential lifecycle

`launch_full_runs.ps1` mints one temporary `sbsandbox` session on the laptop,
creates pods with no AWS credentials in their API configuration, and copies the
session file over SSH. Each pod uses it only to download the bounded inputs,
verifies all 19 shards, deletes the file, unsets credential variables, and then
preprocesses/trains using local scratch and W&B. The laptop copy is deleted
after it has been copied to all requested pods. There is no refresh and no S3
checkpoint upload.

## Launch full runs

The default launches the same full 20B-token configuration on 4x L40S and
4x A100 SXM 80GB:

```powershell
.\scripts\runpod\smollm2_colmlm\launch_full_runs.ps1
```

Launch one hardware type:

```powershell
.\scripts\runpod\smollm2_colmlm\launch_full_runs.ps1 `
  -GpuTypes "NVIDIA L40S" -GpuCount 4
```

The launcher uses the RunPod API key already configured for Cursor, an SSH
public key under `~/.ssh`, and `WANDB_API_KEY` or `~/.wandb_api_key`.

## Throughput

The normal full trainer writes `output/progress/train.jsonl` and W&B metrics:

- `train/recent_tps`: all input tokens per second over the latest log window.
- `train/avg_tps`: all input tokens per second since training started.
- `train/recent_unmasked_tps`: targets that contribute to causal-LM loss per
  second (label position zero excluded because causal loss shifts labels).

There is no benchmark-only trainer or early-stop path. To benchmark, start the
full run, wait for several stable `recent_tps` windows, record them, and
terminate the pod.
