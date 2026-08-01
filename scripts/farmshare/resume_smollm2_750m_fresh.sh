#!/usr/bin/env bash
# Resume smollm2-135m-750m-27ep-fresh from latest local checkpoint.
set -Eeuo pipefail

RUN="${RUN:-/scratch/users/nzhao2/agent-runs/smollm2-135m-750m-27ep-fresh}"
CKPT="${CKPT:-${RUN}/output/checkpoints/step0068670}"
SHARED_HF="${SHARED_HF:-/scratch/users/nzhao2/.cache/huggingface}"

if [[ ! -d "${CKPT}" ]]; then
  echo "missing checkpoint ${CKPT}" >&2
  exit 2
fi

mkdir -p "${RUN}/hf-datasets" "${SHARED_HF}/hub"

python3 - <<'PY' "${RUN}/launch_train.sh" "${SHARED_HF}"
import pathlib, sys

path = pathlib.Path(sys.argv[1])
shared_hf = sys.argv[2]
text = path.read_text()

# Model/tokenizer cache: shared scratch (already populated). Eval datasets: run-local.
block = f'''export HF_HOME="{shared_hf}"
export TRANSFORMERS_CACHE="${{HF_HOME}}/hub"
export HF_DATASETS_CACHE="${{RUN_DIR}}/hf-datasets"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1'''

import re
text = re.sub(
    r'export HF_HOME=.*\nexport TRANSFORMERS_CACHE=.*\n(?:export HF_DATASETS_CACHE=.*\n)?export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1',
    block,
    text,
    count=2,
)

if 'mkdir -p "${HF_DATASETS_CACHE}"' not in text:
    text = text.replace(
        'mkdir -p "${WANDB_DIR}"',
        'mkdir -p "${WANDB_DIR}" "${HF_DATASETS_CACHE}"',
        1,
    )
    text = text.replace(
        'mkdir -p \\"${WANDB_DIR}\\"',
        'mkdir -p \\"${WANDB_DIR}\\" \\"${RUN_DIR}/hf-datasets\\"',
        1,
    )

path.write_text(text)
PY

if grep -q '^RESUME_FROM=' "${RUN}/env.sh"; then
  sed -i "s|^RESUME_FROM=.*|RESUME_FROM=${CKPT}|" "${RUN}/env.sh"
else
  echo "RESUME_FROM=${CKPT}" >> "${RUN}/env.sh"
fi

echo "RESUME_FROM=${CKPT}"
echo "HF_HOME=${SHARED_HF}"
echo "HF_DATASETS_CACHE=${RUN}/hf-datasets"

cd "${RUN}"
JOB=$(sbatch --parsable train.sbatch)
echo "job_id=${JOB}"
echo "${JOB}" > job_id.txt
