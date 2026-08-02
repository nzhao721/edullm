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
4x A100 SXM 80GB using Secure Cloud:

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

The throughput defaults use:

- BF16 FlashAttention 2.
- Liger fused RMSNorm, RoPE, SwiGLU, and linear cross-entropy. The fused loss
  honors `-100` labels without materializing full vocabulary logits.
- `torch.compile(mode="max-autotune-no-cudagraphs")`; PyTorch 2.8 CUDA graphs
  are disabled because their allocator fails with this compiled DDP graph.
- No gradient checkpointing, fused AdamW, TF32, gradient bucket views, and
  prefetched pinned-memory batches.
- Per-device batch 40 on L40S. Batches 56 and 64 exceed the tested card's
  44.39 GiB usable memory even with the fused loss.

## Throughput

The normal full trainer writes `output/progress/train.jsonl` and W&B metrics:

- `train/recent_tps`: all input tokens per second over the latest log window.
- `train/avg_tps`: all input tokens per second since training started.
- `train/recent_unmasked_tps`: targets that contribute to causal-LM loss per
  second (label position zero excluded because causal loss shifts labels).

There is no benchmark-only trainer or early-stop path. To benchmark, start the
full run, wait for several stable `recent_tps` windows, record them, and
terminate the pod.

### Measured 4-GPU throughput (2026-08-02 UTC)

The original baseline runs used sequence length 2048, per-device batch 16,
BF16, gradient checkpointing, and the same prepared corpus:

- 4x A100 SXM 80GB: median 153,425 input tokens/s and 142,672 loss-bearing
  tokens/s over six stable windows (0.68% population CV).
  [W&B run](https://wandb.ai/eduLLM/edullm-smollm2/runs/4jbi0wgc)
- 4x L40S: median 131,327 input tokens/s and 122,146 loss-bearing tokens/s
  over five stable windows (0.56% population CV).
  [W&B run](https://wandb.ai/eduLLM/edullm-smollm2/runs/v5lmbvdl)

The optimized 4x L40S run used per-device batch 40 and the defaults above:

- Median **370,679 input tokens/s** and **345,027 loss-bearing tokens/s** over
  eight stable windows (0.032% population CV), a **2.82x** improvement over
  the L40S baseline. All four GPUs sustained 100% utilization and used about
  42.4/46.1 GB each.
  [W&B run](https://wandb.ai/eduLLM/edullm-smollm2/runs/j4iyod7p)

The tested L40S host was PCIe-only and split across two NUMA nodes. NCCL P2P
stalled in its first collective despite CUDA peer access reporting available;
the launcher therefore sets `NCCL_P2P_DISABLE=1` for L40S only. A100 uses the
default NCCL transport.
