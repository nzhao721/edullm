# SmolLM2-135M: control (standard CLM) vs Co-LMLM (fact-masked CLM)

This document records the architecture, data, and hyperparameters for the two arms of the
SmolLM2-135M FineWeb-Edu experiment:

| Arm | Goal | Primary W&B project | Reference run name |
|-----|------|---------------------|--------------------|
| **Control** | Standard next-token prediction on a 750M-token FineWeb-Edu slice | `edullm-smollm2` | `smollm2-135m-750m-27ep-fresh` |
| **Co-LMLM** | Same corpus + tokenizer, but **mask loss on ModernBERT fact spans** | `edullm-smollm2-colmlm` | `smollm2-colmlm-8xnvidia-l40s-wandb-resume` (8×L40S resume) |

Both arms train **from random init** (`AutoModelForCausalLM.from_config`), not from the public
SmolLM2-135M checkpoint weights.

---

## Shared model architecture

**Checkpoint id:** `HuggingFaceTB/SmolLM2-135M` (config only; weights initialized randomly)

| Parameter | Value |
|-----------|-------|
| Architecture | Llama-style causal LM (`LlamaForCausalLM`) |
| Parameters | ~134.5M |
| `vocab_size` | 49,152 |
| `hidden_size` | 576 |
| `intermediate_size` | 1,536 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 9 |
| `num_key_value_heads` | 3 (GQA) |
| `max_position_embeddings` | 8,192 (training uses 2,048-token chunks) |
| `hidden_act` | SiLU (SwiGLU MLP) |
| `rms_norm_eps` | 1e-5 |
| `rope_theta` | 100,000 |
| `tie_word_embeddings` | true |
| Precision | **bfloat16** (`torch.autocast` on CUDA) |
| Initialization | `AutoConfig.from_pretrained` → `from_config` (no pretrained weights) |

**Tokenizer:** SmolLM2 fast tokenizer from `HuggingFaceTB/SmolLM2-135M`  
**Training sequence length:** 2,048 tokens (non-overlapping contiguous chunks)

---

## Shared optimization schedule

Both trainers use the same core optimizer recipe (defaults in
`scripts/farmshare/train_smollm2_135m_ddp.py` and
`scripts/runpod/smollm2_colmlm/train_smollm2_135m_colmlm_ddp.py`):

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate (peak) | **3×10⁻⁴** |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| LR schedule | Cosine decay with linear warmup |
| Warmup | **2% of total training steps** (`warmup_ratio=0.02`) |
| Gradient clipping | 1.0 (global norm) |
| Random seed | 42 |
| Token budget cap | **20×10⁹ tokens** (`max_train_tokens=20_000_000_000`) |
| Epoch cap | **27** (whichever limit is hit first) |

Because warmup and cosine are defined as a fraction of **total steps**, and total steps scales
inversely with global batch size for a fixed 20B token cap, the **learning-rate curve as a
function of tokens seen is aligned** between arms (~400M-token warmup, cosine over 20B).

**Important:** Peak LR is **not** linearly scaled with global batch size in either arm.

---

## Shared evaluation protocol

Inline eval during training (both arms):

| Setting | Value |
|---------|-------|
| Tasks | **HellaSwag**, **PIQA**, **OpenBookQA** (ARC tasks disabled in W&B logging) |
| Format | 5-shot multiple-choice, rank-classification |
| Metric | Bits per byte (bpb) + accuracy |
| Interval | Every **250×10⁶ training tokens** |
| Sharding | Eval examples partitioned across all DDP ranks |
| Eval seed | 42 |

Implementation: `scripts/farmshare/eval_arc_task_loss_smollm.py` (shared by both trainers).

Checkpoints are also written at the same 250M-token boundaries (Co-LMLM uploads to W&B artifacts;
control additionally supports S3 under `s3://edullm-checkpoints/smollm2/<run>/`).

---

## Control arm — standard causal language modeling

### Intent

Train a baseline SmolLM2-135M on a **750M-token prefix** of FineWeb-Edu tokenized with the SmolLM2
tokenizer. Every token position (except the usual causal shift) contributes to the cross-entropy loss.

### Reference runs & scripts

| Item | Value |
|------|-------|
| Trainer | `scripts/farmshare/train_smollm2_135m_ddp.py` |
| Launch wrapper | `scripts/farmshare/submit_smollm2_135m_500m_40ep.sh` (with env overrides) |
| W&B run | `smollm2-135m-750m-27ep-fresh` |
| FarmShare run dir | `/scratch/users/nzhao2/agent-runs/smollm2-135m-750m-27ep-fresh` |
| Original job (2026-07-30) | `smollm2-135m-750m-27ep-20260730-162021` |

### Data

| Setting | Value |
|---------|-------|
| Corpus | **750M-token slice** of FineWeb-Edu (SmolLM2 tokenizer) |
| On-disk layout | FineWeb-style memmap: `train_tokens.bin` + `meta.json` |
| FarmShare path | `/scratch/users/nzhao2/agent-runs/fineweb-edu-750m-smollm2-tokenized` |
| Provenance | Prefix of `fineweb-edu-1b-smollm2-tokenized` (HuggingFaceFW `fineweb_edu_100BT-shuffled`) |
| Packing | Contiguous `uint32` token stream, chunked into 2,048-token sequences |
| Loss mask | **None** — standard CLM: `labels = input_ids` |

Published edullm-data datasets (`pretrain/fineweb-edu-500m`, `pretrain/fineweb-edu-1b`, etc.) use
the same tokenizer and memmap contract; the 750M control run used a **local persistent scratch
slice**, not a separate published dataset id.

### Training objective

```text
loss = cross_entropy(logits[:, :-1], input_ids[:, 1:])
```

All non-padding targets are supervised.

### Distributed setup (FarmShare)

| Setting | Value |
|---------|-------|
| Cluster | Stanford FarmShare (Slurm) |
| Layout | **2 nodes × 2 GPUs** (`NUM_NODES=2`, `GPUS_PER_NODE=2`) |
| Total GPUs | 4 |
| Launcher | `torch.distributed.run` with `c10d` rendezvous |
| Slurm resources | 8 CPUs/task, 48G RAM (typical for 750M runs) |

### Batch & step budget

| Setting | Value |
|---------|-------|
| `per_device_batch_size` | **8** |
| Global batch (samples) | **32** (= 8 × 4 GPUs) |
| Global batch (tokens/step) | **65,536** (= 32 × 2,048) |
| `num_epochs` | 27 |
| **Total steps** | **~305,176** (= ⌈20B / 65,536⌉; epoch cap not binding) |
| Steps per epoch | ~11,435 (= ⌈750M / 65,536⌉) |

### Performance & memory choices

| Setting | Value |
|---------|-------|
| Gradient checkpointing | **Enabled** |
| FlashAttention | Not used (default eager/SDPA attention) |
| `torch.compile` | Not used |
| Liger fused kernels | Not used |
| AdamW `fused` | false (default PyTorch) |
| DataLoader | `num_workers=4`, `pin_memory=True`, `drop_last=True` |
| Logging | Every 20 steps |

### Checkpointing

| Setting | Value |
|---------|-------|
| Interval | 250M tokens (also historically 0.5 epoch ≈ 5,722 steps on early 750M jobs) |
| Durable storage | S3 (`CHECKPOINT_S3_URI`) + W&B online artifacts |
| Local retention | FarmShare scratch under `output/checkpoints/` |

---

## Co-LMLM arm — fact-span masked causal LM

### Intent

Replicate the **Co-LMLM annotation objective** (arXiv:2607.07707, annotator stage only): a
ModernBERT span tagger marks factual character intervals; tokens intersecting those spans receive
label **−100** and do not contribute to loss. The underlying document text is **not modified**
(no `<FACT>` markers).

### Reference runs & scripts

| Item | Value |
|------|-------|
| Corpus prep | `scripts/runpod/smollm2_colmlm/prepare_annotated_corpus.py` |
| Trainer | `scripts/runpod/smollm2_colmlm/train_smollm2_135m_colmlm_ddp.py` |
| Launch entrypoint | `scripts/runpod/smollm2_colmlm/run_full_training.sh` |
| Pod launcher | `scripts/runpod/smollm2_colmlm/launch_full_runs.ps1` |
| W&B project | `edullm-smollm2-colmlm` |
| W&B group | `colmlm-fact-masked` |
| Current production run | `smollm2-colmlm-8xnvidia-l40s-wandb-resume` (pod `y83pcj0g00wijz`) |
| Prior 4× run (stopped for migration) | `smollm2-colmlm-4xnvidia-l40s-20260802-032940` |

### Data

| Setting | Value |
|---------|-------|
| Annotation source | `s3://edullm-checkpoints/runpod/colmlm-annotate/output/` (19 worker shards) |
| Annotator | ModernBERT token-classification span tagger (Co-LMLM pipeline step 1 only) |
| Schema | Per-document JSONL: `text` + `annotations[]` with `char_start`/`char_end`/`span` |
| Provenance metadata | `s3://edullm-data/pretrain/fineweb-edu-1b/v6` (`dataset.json` only — **not** joined for tokens) |
| Corpus token cap | **750×10⁶ tokens** (`CORPUS_MAX_TOKENS=750000000`) |
| Packed layout | Parallel memmap shards: `uint32` `input_ids` + `uint8` `loss_mask` per 2,048-token row |
| Typical packed size | ~755M tokens, ~369k sequences (see `output/run_meta.json` on pod) |
| Masked target fraction | ~6–7% of label positions (fact-span intersections; logged as `masked_fraction`) |

**Why re-tokenize?** The v6 edullm-data release documents that its **token and raw text groups are
not document-aligned**. Annotations embed the aligned raw text, so `prepare_annotated_corpus.py`
tokenizes annotation records directly and builds span masks from tokenizer offset mappings.

### Masking rule

For each annotated half-open character span `[char_start, char_end)`:

1. Tokenize the document with `return_offsets_mapping=True`.
2. Any token whose character interval intersects a fact span is marked in `loss_mask`.
3. At training time: `labels = input_ids`; `labels[loss_mask] = -100`.

Boundary-crossing tokens are masked (partial overlap counts).

### Training objective

```text
loss = cross_entropy(logits[:, :-1], labels[:, 1:])   # with labels == -100 ignored
```

Logged throughput includes `recent_unmasked_tps` (loss-bearing tokens per second).

### Distributed setup (RunPod)

| Setting | 4×L40S (original) | 8×L40S (current resume) |
|---------|-------------------|-------------------------|
| Platform | RunPod Secure Cloud | RunPod Secure Cloud |
| GPU | 4 × NVIDIA L40S (48 GB) | 8 × NVIDIA L40S |
| Launcher | `torch.distributed.run --standalone` | same |
| `NPROC` | 4 | 8 |
| NCCL | `NCCL_P2P_DISABLE=1` on tested L40S hosts | same |

### Batch & step budget

Global batch is held at **160 samples** across the 4→8 GPU migration (`GLOBAL_BATCH_SAMPLES=160`).

| Setting | 4×L40S | 8×L40S |
|---------|--------|--------|
| `per_device_batch_size` | 40 | 20 |
| Global batch (samples) | 160 | 160 |
| Global batch (tokens/step) | **327,680** | **327,680** |
| `num_epochs` | 27 | 27 |
| **Total steps** | **61,036** | **61,036** |
| Steps per epoch | 2,304 | 2,304 |

`total_steps = min(27 × steps_per_epoch, ⌈20B / global_batch_tokens⌉)` — the **20B token cap**
binds before the epoch cap.

### Performance & memory choices

| Setting | Value |
|---------|-------|
| Attention | **FlashAttention 2** (`attn_implementation=flash_attention_2`) |
| `torch.compile` | **Enabled**, mode `max-autotune-no-cudagraphs` |
| Liger kernels | **Enabled** (fused RMSNorm, RoPE, SwiGLU, fused linear cross-entropy honoring −100 labels) |
| Gradient checkpointing | **Disabled** (`--no-gradient-checkpointing`) |
| AdamW | `fused=True` on CUDA |
| TF32 | Enabled for matmul/cudnn |
| DDP | `gradient_as_bucket_view=True`, `bucket_cap_mb=100` |
| DataLoader | `num_workers=4`, `prefetch_factor=4`, `pin_memory=True`, `persistent_workers` |
| Logging | Every 20 steps |
| Compile warmup | ~20+ minutes on first steps (Inductor autotune) |

### Checkpointing

| Setting | Value |
|---------|-------|
| Interval | 250M tokens |
| Durable storage | **W&B artifacts only** (no S3 checkpoint upload on RunPod path) |
| Local retention | Last **2** checkpoints (`keep_local_checkpoints=2`) |
| Resume | Supports 4→8 GPU resume at **fixed** `global_batch_tokens` (loads weights, optimizer, step, tokens; rebuilds scheduler) |

---

## Side-by-side summary

| | **Control** | **Co-LMLM** |
|---|-------------|-------------|
| **Objective** | Standard CLM | Fact-span masked CLM (Co-LMLM annotator) |
| **Corpus tokens (one pass)** | ~750M FineWeb-Edu | ~750M annotated FineWeb-Edu (same underlying text, span masks added) |
| **Training token budget** | 20B | 20B |
| **Epoch cap** | 27 | 27 |
| **Seq len** | 2,048 | 2,048 |
| **Peak LR** | 3e-4 | 3e-4 |
| **Global batch (samples)** | 32 | 160 |
| **Global batch (tokens/step)** | 65,536 | 327,680 |
| **Total optimizer steps** | ~305,176 | 61,036 |
| **GPUs** | 4 (FarmShare 2×2) | 4 or 8 (RunPod L40S) |
| **Gradient checkpointing** | On | Off |
| **FlashAttention / compile / Liger** | Off | On |
| **Eval suite** | HellaSwag, PIQA, OpenBookQA @ 250M | Same |
| **W&B project** | `edullm-smollm2` | `edullm-smollm2-colmlm` |

---

## Comparing results fairly

1. **Use tokens seen, not step count**, on the x-axis (305k vs 61k steps is expected).
2. **Align eval milestones** at 250M-token boundaries (0.25B, 0.5B, …, 20B).
3. Treat **global batch size** (32 vs 160) as a minor confound; LR-vs-tokens is matched.
4. The **masking objective** and **annotated data pipeline** are the primary experimental variables.
5. **Hardware/stack differ** (FarmShare vs RunPod; control uses checkpointing, Co-LMLM uses fused kernels). Throughput numbers are not directly comparable; downstream eval metrics are.

---

## Key source files

```
scripts/farmshare/
  train_smollm2_135m_ddp.py          # control trainer
  submit_smollm2_135m_500m_40ep.sh    # control launch (override DATA_DIR / NUM_EPOCHS)
  eval_arc_task_loss_smollm.py        # shared eval suite
  resume_smollm2_750m_fresh.sh        # control resume helper

scripts/runpod/smollm2_colmlm/
  prepare_annotated_corpus.py         # annotation → packed tokens + masks
  train_smollm2_135m_colmlm_ddp.py    # Co-LMLM trainer
  run_full_training.sh                # full 20B RunPod pipeline
  launch_full_runs.ps1                # pod + credential bootstrap
  README.md                           # RunPod ops + throughput notes
```

---

## Revision notes

- Document reflects configs as of **2026-08-02**.
- Co-LMLM 8× run resumed from W&B artifact `step0014497` (4× checkpoint) with unchanged global batch 160.
- For live `run_meta.json` on the 8× pod: `/workspace/smollm2-colmlm-8xnvidia-l40s-wandb-resume/output/run_meta.json`.
