#!/usr/bin/env bash
# Launch a fully staged REL-RefHQ Instruct v3 run without AWS credentials.
set -Eeuo pipefail

EDULLM_ROOT=/workspace/edullm
TS_ROOT="${EDULLM_ROOT}/experiments/token-selection"
OLMO_CORE_DIR=/workspace/OLMo-core-token-selection
RUN_DIR=/workspace/edullm-runs/token-selection/rel-ema-refhq-instruct-v3
CFG_REL=rel-ema-refhq/configs/run_rel_ema_refhq_instruct_v3_10b.yaml
CFG="${TS_ROOT}/${CFG_REL}"
REF_PT="${RUN_DIR}/ref-cache/refhq_instruct_v3_step940_model.pt"
REFERENCE_S3=s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-instruct-v3/checkpoints/step940/
TRAIN_LOG=/workspace/rel-ema-refhq-instruct-v3-train.log
WANDB_SESSION_ENV=/workspace/wandb-session-token-selection.env
TASK_LOSS_EVAL_SCRIPT=/workspace/OLMo-core/.edullm/task_loss/eval_task_loss_olmo_core.py

[[ -f "${REF_PT}" ]] || { echo "missing staged reference: ${REF_PT}" >&2; exit 2; }
[[ -f "${RUN_DIR}/data/rel_ema_refhq_10b/tokens/manifest.json" ]] || {
  echo "missing staged RegMix manifest" >&2
  exit 2
}
[[ ! -e "${RUN_DIR}/aws-session.env" ]] || {
  echo "refusing to train while temporary AWS session exists" >&2
  exit 2
}
[[ -f "${TASK_LOSS_EVAL_SCRIPT}" ]] || {
  echo "missing task-loss eval script: ${TASK_LOSS_EVAL_SCRIPT}" >&2
  exit 2
}
if pgrep -f '[p]ython.*-m token_selection\.scripts\.train_olmo_template' >/dev/null; then
  echo "refusing duplicate token-selection trainer" >&2
  exit 3
fi

CFG="${CFG}" python3 - <<'PY'
import os
from pathlib import Path
import yaml

p = Path(os.environ["CFG"])
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
data = cfg.setdefault("data", {})
data["dataset_version"] = "v1"
data["offline_staged"] = True
train = cfg.setdefault("train", {})
train["pre_train_checkpoint"] = False
train["activation_checkpoint_interval"] = 2
cfg.setdefault("eval", {}).setdefault("task_loss", {})["eval_script"] = (
    "/workspace/OLMo-core/.edullm/task_loss/eval_task_loss_olmo_core.py"
)
p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(
    f"offline staged config: {data['dataset_id']}/{data['dataset_version']} "
    f"reference={cfg['reference']['s3_uri']}"
)
PY

BASE_WANDB=/workspace/wandb-session.env WANDB_SESSION_ENV="${WANDB_SESSION_ENV}" \
python3 - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["BASE_WANDB"])
target = Path(os.environ["WANDB_SESSION_ENV"])
lines = []
seen = set()
for line in source.read_text(encoding="utf-8").splitlines():
    key = line.split("=", 1)[0].replace("export ", "").strip() if "=" in line else ""
    if key in {"WANDB_PROJECT", "EDULLM_WANDB_PROJECT"}:
        lines.append(f"export {key}='token-selection'")
        seen.add(key)
    else:
        lines.append(line)
for key in ("WANDB_PROJECT", "EDULLM_WANDB_PROJECT"):
    if key not in seen:
        lines.append(f"export {key}='token-selection'")
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
target.chmod(0o600)
print("prepared token-selection W&B session")
PY

if [[ -f "${TRAIN_LOG}" ]]; then
  mv "${TRAIN_LOG}" "${TRAIN_LOG%.log}.attempt1.log"
fi
stamp="$(date -u +'%Y%m%d-%H%M%S')"
name="rel-ema-refhq-instruct-v3-10b-runpod-${stamp}"
cat > "${RUN_DIR}/run.env" <<EOF
export EDULLM_RUN_ID='${name}'
export REFERENCE_S3='${REFERENCE_S3}'
export REF_PT='${REF_PT}'
EOF

cd "${EDULLM_ROOT}"
nohup env \
  EDULLM_ROOT="${EDULLM_ROOT}" \
  OLMO_CORE_DIR="${OLMO_CORE_DIR}" \
  RUN_DIR="${RUN_DIR}" \
  WORK="${RUN_DIR}" \
  CFG_REL="${CFG_REL}" \
  NUM_GPUS=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NUM_WORKERS=8 \
  RANK_MICROBATCH_SIZE=32768 \
  TOKEN_SELECTION_REF_CACHE="${RUN_DIR}/ref-cache" \
  REF_PT="${REF_PT}" \
  WANDB_SESSION_ENV="${WANDB_SESSION_ENV}" \
  WANDB_PROJECT=token-selection \
  WANDB_MODE=online \
  WANDB_RUN_NAME="${name}" \
  TASK_LOSS_EVAL_SCRIPT="${TASK_LOSS_EVAL_SCRIPT}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash "${TS_ROOT}/rel-ema-refhq/launch_train.sh" train --resume \
  > "${TRAIN_LOG}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" > "${RUN_DIR}/trainer.pid"
echo "TRAIN_PID=${pid} NAME=${name} LOG=${TRAIN_LOG}"

