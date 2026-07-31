# P3Math

Filtered Proof-Pile-2 math corpus + Lean 4 Mathlib for proof/reasoning pretraining.

## Settled filters

| Source | Keep rule |
|--------|-----------|
| **OpenWebMath** | `math_score ≥ 0.72`, native TeX signals ≥ 2, **legacy share ≤ 50%**, drop path noise / image-only |
| **AlgebraicStack** | RefHQ import-shell drop, then language ∈ {Agda, Coq, Idris, Isabelle, Lean, TeX} |
| **arXiv** | Primary `math.*` + biblio strip; then PP2 `language ∈ {en,eng,english}`, strip figures/tables/tikz/includegraphics/acks, keep only if **10k–200k chars** and **≥2** `theorem|lemma|proposition|corollary|proof` envs; then **cross-list purity** (all cats `math.*`) + **pure-math subcategory allowlist** (AG, AT, CT, AC, NT, GT, DG, CO, LO, RT, RA, KT, GR, OA, QA, SG, MG, GN, FA, CV, CA, SP, PR; drop ST/OC/NA/AP/MP/GM/HO/DS) |
| **Lean4-Mathlib** | Keep as-is |

## Layout on FarmShare scratch

```
$SCRATCH_ROOT/
  raw/
    proof-pile-2/{arxiv,open-web-math,algebraic-stack}/
    lean4-mathlib/
    arxiv-metadata/
  filtered/
    open-web-math/open-web-math.jsonl.zst
    algebraic-stack/algebraic-stack.jsonl.zst
    arxiv/arxiv-math.jsonl.zst          # math.* + biblio
    arxiv/arxiv-math-refined.jsonl.zst  # + EN / proof / length / noise strip
    arxiv/arxiv-math-pure.jsonl.zst     # + all-math.* cross-list + pure subcat allowlist
    lean4-mathlib/   # parquet/files copied as-is
  manifests/
    arxiv_pure_summary.json             # keep/drop counts for pure pass
  logs/
  venv/
```

## Run

```bash
# from a FarmShare login node with repo synced
export SCRATCH_ROOT=/scratch/users/$USER/agent-runs/p3math-<stamp>
export REPO_ROOT=/scratch/users/$USER/agent-runs/edullm-farmshare-staging
mkdir -p "$SCRATCH_ROOT/logs"
cd "$SCRATCH_ROOT"

jid=$(sbatch --exclude=wheat-01 --export=ALL,SCRATCH_ROOT,REPO_ROOT \
  "$REPO_ROOT/datasets/p3math/download.sbatch" | awk '{print $4}')
sbatch --exclude=wheat-01 --dependency=afterok:$jid --export=ALL,SCRATCH_ROOT,REPO_ROOT \
  "$REPO_ROOT/datasets/p3math/filter.sbatch"
```

## Sources

- https://huggingface.co/datasets/EleutherAI/proof-pile-2
- https://huggingface.co/datasets/phanerozoic/Lean4-Mathlib
- https://huggingface.co/datasets/jackkuo/arXiv-metadata-oai-snapshot
