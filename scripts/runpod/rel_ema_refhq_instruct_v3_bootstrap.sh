#!/usr/bin/env bash
# Stage RegMix + the Aug-04 RefHQ Instruct v3 EMA seed, then launch REL-RefHQ.
set -Eeuo pipefail

EDULLM_ROOT="${EDULLM_ROOT:-/workspace/edullm}"
TS_ROOT="${EDULLM_ROOT}/experiments/token-selection"
OLMO_CORE_DIR="${OLMO_CORE_DIR:-/workspace/OLMo-core-token-selection}"
RUN_DIR="${RUN_DIR:-/workspace/edullm-runs/token-selection/rel-ema-refhq-instruct-v3}"
AWS_SESSION_ENV="${AWS_SESSION_ENV:-${RUN_DIR}/aws-session.env}"
WANDB_SESSION_ENV="${WANDB_SESSION_ENV:-/workspace/wandb-session.env}"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-/workspace/rel-ema-refhq-instruct-v3-bootstrap.log}"
TRAIN_LOG="${TRAIN_LOG:-/workspace/rel-ema-refhq-instruct-v3-train.log}"
REFERENCE_S3="s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-instruct-v3/checkpoints/step940/"
REF_CACHE="${RUN_DIR}/ref-cache"
REF_PT="${REF_CACHE}/refhq_instruct_v3_step940_model.pt"
BASE_CFG="${TS_ROOT}/rel-ema-refhq/configs/run_rel_ema_refhq_10b.yaml"
CFG_REL="rel-ema-refhq/configs/run_rel_ema_refhq_instruct_v3_10b.yaml"
RUN_CFG="${TS_ROOT}/${CFG_REL}"
export PATH="/workspace/bin:${PATH}"

log() { printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${BOOTSTRAP_LOG}"; }

cleanup_session() {
  rm -f "${AWS_SESSION_ENV}"
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
}
trap cleanup_session EXIT

[[ -f "${AWS_SESSION_ENV}" ]] || { log "missing temporary AWS session: ${AWS_SESSION_ENV}"; exit 2; }
[[ -f "${WANDB_SESSION_ENV}" ]] || { log "missing W&B session: ${WANDB_SESSION_ENV}"; exit 2; }
[[ -f "${BASE_CFG}" ]] || { log "missing REL-RefHQ config: ${BASE_CFG}"; exit 2; }
if pgrep -f '[p]ython.*-m token_selection\.scripts\.train_olmo_template' >/dev/null; then
  log "refusing duplicate token-selection trainer"
  exit 3
fi

mkdir -p "${RUN_DIR}" "${REF_CACHE}"

log "installing latest edullm-data client"
python3 -m pip install --quiet --upgrade --force-reinstall --break-system-packages \
  "edullm-data @ git+https://github.com/edu-llm/edullm-data@main"
python3 - <<'PY'
import edullm_data
print("edullm_data", getattr(edullm_data, "__version__", "?"), edullm_data.__file__)
PY

log "writing run-specific config with Instruct Reference v3 step940"
BASE_CFG="${BASE_CFG}" RUN_CFG="${RUN_CFG}" REFERENCE_S3="${REFERENCE_S3}" python3 - <<'PY'
import os
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(os.environ["BASE_CFG"]).read_text(encoding="utf-8"))
cfg["run_id"] = "rel-ema-refhq-instruct-v3-10b-runpod"
data = cfg.setdefault("data", {})
data["dataset_version"] = "v1"
data["offline_staged"] = True
ref = cfg.setdefault("reference", {})
ref.update(
    {
        "load_path": None,
        "s3_uri": os.environ["REFERENCE_S3"],
        "step": 940,
        "planned_total_steps": 940,
        "dataset": "pretrain/refhq-instruct/v3",
    }
)
# The pod receives a bounded environment session, not a named local profile.
cfg.setdefault("s3", {})["profile"] = "none"
Path(os.environ["RUN_CFG"]).write_text(
    yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
)
PY

# Session is valid only for these two bounded read stages.
# shellcheck disable=SC1090
source "${AWS_SESSION_ENV}"
unset AWS_PROFILE
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export EDULLM_ROOT OLMO_CORE_DIR RUN_DIR
export WORK="${RUN_DIR}"
export CFG_REL
export NUM_GPUS=8
export NUM_WORKERS=8
export RANK_MICROBATCH_SIZE=16384
export TOKEN_SELECTION_REF_CACHE="${REF_CACHE}"
export REF_PT

log "staging pretrain/regmix-10b/v1 train tokens (~40 GB)"
bash "${TS_ROOT}/rel-ema-refhq/launch_train.sh" prepare

log "staging and unsharding Instruct Reference v3 step940 (~7.6 GB DistCP)"
PYTHONPATH="${TS_ROOT}:${OLMO_CORE_DIR}/src" \
REFERENCE_S3="${REFERENCE_S3}" REF_CACHE="${REF_CACHE}" REF_PT="${REF_PT}" \
python3 - <<'PY'
import json
import os
from pathlib import Path
from token_selection.olmo_ext.refhq_materialize import ensure_distcp_pt

source = os.environ["REFERENCE_S3"]
output = ensure_distcp_pt(
    source,
    cache_dir=Path(os.environ["REF_CACHE"]),
    output_name=Path(os.environ["REF_PT"]).name,
)
meta = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
assert meta["source_s3"] == source, (meta["source_s3"], source)
assert output.stat().st_size > 1_000_000_000, output.stat().st_size
print(f"reference_verified path={output} bytes={output.stat().st_size} source={source}")
PY

log "verifying staged dataset contract"
RUN_DIR="${RUN_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ["RUN_DIR"]) / "data/rel_ema_refhq_10b/tokens/manifest.json"
m = json.loads(p.read_text(encoding="utf-8"))
assert m["dataset_id"] == "pretrain/regmix-10b", m["dataset_id"]
assert m["dataset_version"] == "v1", m["dataset_version"]
assert int(m["n_tokens"]) == 9_989_799_834, m["n_tokens"]
missing = [s["path"] for s in m["shards"] if not (p.parent / s["path"]).is_file()]
assert not missing, missing[:3]
print(
    f"dataset_verified id={m['dataset_id']}/{m['dataset_version']} "
    f"n_tokens={m['n_tokens']} shards={len(m['shards'])}"
)
PY

log "removing temporary AWS session before training"
cleanup_session
trap - EXIT

RUN_STAMP="$(date -u +'%Y%m%d-%H%M%S')"
WANDB_RUN_NAME="rel-ema-refhq-instruct-v3-10b-runpod-${RUN_STAMP}"
cat > "${RUN_DIR}/run.env" <<EOF
export EDULLM_RUN_ID='${WANDB_RUN_NAME}'
export REFERENCE_S3='${REFERENCE_S3}'
export REF_PT='${REF_PT}'
EOF

log "launching 8-GPU REL-RefHQ training: ${WANDB_RUN_NAME}"
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
  RANK_MICROBATCH_SIZE=16384 \
  TOKEN_SELECTION_REF_CACHE="${REF_CACHE}" \
  REF_PT="${REF_PT}" \
  WANDB_SESSION_ENV="${WANDB_SESSION_ENV}" \
  WANDB_PROJECT=token-selection \
  WANDB_MODE=online \
  WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  bash "${TS_ROOT}/rel-ema-refhq/launch_train.sh" train \
  >> "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${RUN_DIR}/trainer.pid"
log "trainer launcher pid=${TRAIN_PID}; log=${TRAIN_LOG}"

