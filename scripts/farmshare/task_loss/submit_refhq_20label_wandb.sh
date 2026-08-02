#!/usr/bin/env bash
# FarmShare: re-eval all RefHQ checkpoints with the MixLaw 20-label suite,
# then log results to W&B project ``refhq``.
#
# Run on a FarmShare login node (via control socket). Does not use wheat-01.
set -Eeuo pipefail

SUNET="${SUNET:-nzhao2}"
SCRATCH="/scratch/users/${SUNET}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRATCH}/agent-runs/refhq-20label-wandb-${STAMP}}"

REFHQ_ROOT="${REFHQ_ROOT:-${SCRATCH}/agent-runs/refhq-models-all-20260727T220851Z/unsharded}"
VENV="${VENV:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217/venv}"
BASE_CONFIG="${BASE_CONFIG:-${SCRATCH}/agent-runs/olmo-ladder-370m-20260722-185217/checkpoints/edullm-370M-30B/step5000-unsharded/config.yaml}"

WANDB_PROJECT="${WANDB_PROJECT:-refhq}"
WANDB_ENTITY="${WANDB_ENTITY:-eduLLM}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-refhq-370m-posthoc-20label-${STAMP}}"
EVAL_TIME="${EVAL_TIME:-01:15:00}"
DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE:-4}"

SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${RUN_DIR}/scripts" "${RUN_DIR}/logs" "${RUN_DIR}/eval/refhq"
chmod 700 "${RUN_DIR}"

if [[ "${SKIP_SCRIPT_COPY:-0}" != "1" ]]; then
  cp -f "${SCRIPT_SRC}/eval_task_loss_olmo_core.py" "${RUN_DIR}/scripts/"
  cp -f "${SCRIPT_SRC}/prepare_model_eval_pt.py" "${RUN_DIR}/scripts/"
  cp -f "${SCRIPT_SRC}/task_loss_eval_step.sbatch" "${RUN_DIR}/scripts/"
  cp -f "${SCRIPT_SRC}/log_refhq_task_loss_to_wandb.py" "${RUN_DIR}/scripts/"
  sed -i 's/\r$//' "${RUN_DIR}/scripts/"*.{py,sbatch,sh} 2>/dev/null || true
  chmod +x "${RUN_DIR}/scripts/"*.sbatch
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "missing venv ${VENV}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "missing base config ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${REFHQ_ROOT}" ]]; then
  echo "missing RefHQ root ${REFHQ_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/wandb-session.env" ]]; then
  echo "missing ${RUN_DIR}/wandb-session.env (push via push_wandb_session_to_farmshare.sh)" >&2
  exit 2
fi

# Fail closed: ai2-olmo must expose the OLMES BPB labels MixLaw uses.
"${VENV}/bin/python" - <<'PY'
from olmo.eval.downstream import label_to_task_map
need = [
    "arc_challenge_val_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
]
missing = [k for k in need if k not in label_to_task_map]
if missing:
    raise SystemExit(f"venv olmo missing OLMES labels: {missing}")
print("olmes_labels_ok", len(label_to_task_map))
PY

list_steps() {
  find "${REFHQ_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step*' \
    | sed 's#.*/##' \
    | sed 's/^step//' \
    | sort -n
}

# Reject legacy incomplete JSON so we never SKIP an 11-label artifact.
purge_incomplete() {
  local step="$1"
  local out="${RUN_DIR}/eval/refhq/step${step}_task_loss.json"
  [[ -f "${out}" ]] || return 0
  if "${VENV}/bin/python" - "${out}" <<'PY'
import json, sys
from pathlib import Path
RAW = {
    "arc_challenge_val_rc_5shot_bpb","arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb","arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb","csqa_val_rc_5shot_bpb","hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb","openbookqa_test_rc_5shot_bpb","piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb","winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb","mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb","mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb","mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb","mmlu_other_test_rc_5shot_bpb",
}
p = Path(sys.argv[1])
d = json.loads(p.read_text())
vals = {}
for field in ("labels", "task_loss_bpb"):
    src = d.get(field) or {}
    if isinstance(src, dict):
        for k, v in src.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals[str(k)] = float(v)
sys.exit(0 if RAW.issubset(vals) else 1)
PY
  then
    echo "KEEP complete ${out}"
  else
    echo "PURGE incomplete ${out}"
    rm -f "${out}"
  fi
}

: > "${RUN_DIR}/eval_jobs.txt"
n_eval=0
deps=()

for step in $(list_steps); do
  ckpt="${REFHQ_ROOT}/step${step}"
  if [[ ! -d "${ckpt}" ]]; then
    echo "skip missing ${ckpt}" >&2
    continue
  fi
  purge_incomplete "${step}"
  job=$(sbatch --exclude=wheat-01 --parsable \
    --job-name="tl-refhq-s${step}" \
    --time="${EVAL_TIME}" \
    --chdir="${RUN_DIR}" \
    --export=ALL,RUN_DIR="${RUN_DIR}",MODEL_NAME="refhq",STEP="${step}",CHECKPOINT_DIR="${ckpt}",CKPT_FORMAT="model_pt",VENV="${VENV}",BASE_CONFIG="${BASE_CONFIG}",DEVICE_EVAL_BATCH_SIZE="${DEVICE_EVAL_BATCH_SIZE}",LADDER_BASE_CONFIG="${BASE_CONFIG}" \
    "${RUN_DIR}/scripts/task_loss_eval_step.sbatch")
  echo "${job}" >> "${RUN_DIR}/eval_jobs.txt"
  deps+=("${job}")
  n_eval=$((n_eval + 1))
  echo "SUBMITTED tl-refhq-s${step} job=${job} ckpt=${ckpt}"
done

if [[ "${n_eval}" -eq 0 ]]; then
  echo "no RefHQ checkpoints found under ${REFHQ_ROOT}" >&2
  exit 1
fi

dep_csv=$(IFS=,; echo "${deps[*]}")
WB_SBATCH="${RUN_DIR}/logs/wandb_log_refhq.sbatch"
cat > "${WB_SBATCH}" <<EOF
#!/bin/bash
#SBATCH --job-name=refhq-wb-log
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:45:00
#SBATCH --exclude=wheat-01
#SBATCH --chdir=${RUN_DIR}
#SBATCH --output=${RUN_DIR}/logs/wandb-log-%j.out
#SBATCH --error=${RUN_DIR}/logs/wandb-log-%j.err
#SBATCH --dependency=afterok:${dep_csv}

set -Eeuo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
# shellcheck disable=SC1091
source "${RUN_DIR}/wandb-session.env"
export WANDB_START_METHOD=thread
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_ENTITY="${WANDB_ENTITY}"

python "${RUN_DIR}/scripts/log_refhq_task_loss_to_wandb.py" \\
  --eval-dir "${RUN_DIR}/eval/refhq" \\
  --wandb-project "${WANDB_PROJECT}" \\
  --wandb-entity "${WANDB_ENTITY}" \\
  --wandb-run-name "${WANDB_RUN_NAME}" \\
  --run-id-out "${RUN_DIR}/wandb_run_id.txt" \\
  --upload-artifacts \\
  --require-all-complete
EOF

wb_job=$(sbatch --parsable "${WB_SBATCH}")
echo "${wb_job}" > "${RUN_DIR}/logs/wandb_log_job_id.txt"

cat > "${RUN_DIR}/run_meta.json" <<EOF
{
  "schema_version": 1,
  "arm": "refhq",
  "suite": "olmes_rc_5shot_bpb_20label",
  "comparable_to": "mixlaw-370m-validation",
  "refhq_root": "${REFHQ_ROOT}",
  "eval_dir": "${RUN_DIR}/eval/refhq",
  "n_eval_jobs": ${n_eval},
  "eval_job_ids": [$(IFS=,; echo "${deps[*]}" | sed 's/[^,]*/"&"/g')],
  "wandb_log_job_id": "${wb_job}",
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_entity": "${WANDB_ENTITY}",
  "wandb_run_name": "${WANDB_RUN_NAME}"
}
EOF

echo "RUN_DIR=${RUN_DIR}"
echo "eval_jobs=${n_eval} dependency=afterok:${dep_csv}"
echo "wandb_log_job=${wb_job} project=${WANDB_PROJECT}"
squeue -u "${SUNET}" -o '%.18i %.12P %.20j %.8T %.10M %.6D %R' || true
