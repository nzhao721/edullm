# Co-LMLM ModernBERT annotate — RunPod

CLI port of `Co_LMLM_annotate_ModernBERT.ipynb` for filesystem inputs (no Colab / Drive API).

## Smoke (single A100, S3-staged)

Stage locally (model `final/` + one raw shard + script) to
`s3://edullm-checkpoints/runpod/colmlm-annotate-smoke/`, mint a short-lived AWS
session, then:

```powershell
$awsEnv = Join-Path $env:TEMP "aws-session-colmlm-smoke.env"
.\scripts\farmshare\mint_aws_session_local.ps1 -Profile sbsandbox -OutputPath $awsEnv
node scripts/runpod/colmlm_annotate/create_annotate_smoke_s3.js $awsEnv
```

Validated smoke (2026-08-01): **200 docs → 3,046 spans, 0 offset mismatches**, ~0.2 min on A100 after S3 sync. Output also written to `…/colmlm-annotate-smoke/output/`.

## Multi-worker (later)

Each pod owns a disjoint subset of input files:

```bash
python3 annotate_modernbert.py \
  --model-dir /data/model/final \
  --input-dir /data/fineweb-edu-1b-smollm2-raw \
  --output-dir /data/annotations \
  --path-prefix shards/ \
  --id-field doc_id \
  --worker-index 3 \
  --num-workers 19
```

Resume unit is the input file name (recorded per worker in `_manifest.json`).
