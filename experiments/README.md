# Experiments

| Directory | Description |
|-----------|-------------|
| [`baseline/`](baseline/) | Legacy OLMo-ladder 370M CPT on ~30B (FarmShare + AWS entrypoints) |
| [`skill-dag/mixlaw/`](skill-dag/mixlaw/) | Data mixing laws pilot (24× DataDecide-60M mixtures over olmohq domains) |
| [`token-selection/`](token-selection/) | Token-selection training framework and experiment arms |
| [`proposals/`](proposals/) | P1 experiment proposals (markdown + images) |

Shared FarmShare platform utilities (`bootstrap.sh`, `prepare_aws_session*.sh`, `write_aws_session_env.py`, etc.) remain in [`scripts/farmshare/`](../scripts/farmshare/). Cross-corpus dataset helpers (`olmo_shard_utils.py`, `download_s3_shard.py`, …) live in [`datasets/`](../datasets/).
