#!/bin/bash
# Shared setup for Farmshare batch jobs.

slurm_project_setup() {
    # Resolve project root: farmshare.env > SLURM_SUBMIT_DIR > fail.
    local root=""
    if [[ -f "${SLURM_SUBMIT_DIR:-}/farmshare.env" ]]; then
        # shellcheck disable=SC1091
        source "${SLURM_SUBMIT_DIR}/farmshare.env"
        root="${EDULLM_ROOT:-}"
    elif [[ -f "${HOME}/edullm/farmshare.env" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/edullm/farmshare.env"
        root="${EDULLM_ROOT:-}"
    fi
    if [[ -z "${root}" ]]; then
        root="${SLURM_SUBMIT_DIR:-}"
    fi
    if [[ -z "${root}" || ! -d "${root}" ]]; then
        echo "ERROR: cannot resolve EDULLM_ROOT."
        echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
        echo "  Set EDULLM_ROOT in ${SLURM_SUBMIT_DIR:-~/edullm}/farmshare.env"
        exit 1
    fi

    cd "${root}" || exit 1
    mkdir -p logs

    if [[ ! -f "${root}/.venv/bin/activate" ]]; then
        echo "ERROR: missing venv at ${root}/.venv"
        exit 1
    fi

    # Do not run module purge — it can break the interpreter the venv was built with.
    # shellcheck disable=SC1091
    source "${root}/.venv/bin/activate"
    export PYTHONPATH="${root}"
    export EDULLM_ROOT="${root}"
    export PYTHONUNBUFFERED=1

    echo "=== SLURM job ${SLURM_JOB_ID:-local} on $(hostname) ==="
    echo "Partition:       ${SLURM_JOB_PARTITION:-n/a}"
    echo "EDULLM_ROOT:     ${root}"
    echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<unset>}"
    echo "Python:          $(which python)"
    python -V

    if [[ "${SLURM_JOB_PARTITION:-}" == "gpu" ]]; then
        if ! nvidia-smi; then
            echo "ERROR: job is on gpu partition but nvidia-smi failed."
            exit 1
        fi
        python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: torch.cuda.is_available() is False on a GPU job")
PY
    fi
}
