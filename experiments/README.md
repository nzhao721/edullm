# Experiments

| Directory | Description |
|-----------|-------------|
| [`skill-dag/mixlaw/`](skill-dag/mixlaw/) | Data mixing laws pilot (24× DataDecide-60M). Chinchilla fits use in-run jsonl only (no step-1451 anchor). |
| [`skill-dag/skillit/`](skill-dag/skillit/) | Skill-It probes + dual-arm OLMo2-370M (offline A vs mixing-law derivatives). See [`skill-dag/README.md`](skill-dag/README.md) for artifact paths. |
| [`token-selection/`](token-selection/) | Token-selection training framework and experiment arms |
| [`curriculum/`](curriculum/) | Curriculum learning on RegMix 10B (OLMo2-370M; 17-arm pacing × metric matrix) |
| [`proposals/`](proposals/) | P1 experiment proposals (markdown + images) |

Shared FarmShare platform utilities (`bootstrap.sh`, `prepare_aws_session*.sh`, `write_aws_session_env.py`, etc.) remain in [`scripts/farmshare/`](../scripts/farmshare/). Cross-corpus dataset helpers (`olmo_shard_utils.py`, `download_s3_shard.py`, …) live in [`datasets/`](../datasets/).
