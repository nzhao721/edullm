# The platform resolves and passes the reviewed digest from its repository registry.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip==26.2 setuptools==83.0.0 wheel==0.47.0

WORKDIR /opt/edullm
COPY . /opt/edullm

RUN python -m pip install --no-cache-dir \
        --requirement .edullm/requirements-linux-cu128.lock

ENV EDULLM_ROOT=/opt/edullm \
    LADDER_BASE_CONFIG=/opt/edullm/experiments/skill-dag/mixlaw/ladder_base_config.yaml \
    PYTHONPATH=/opt/edullm/experiments/skill-dag/mixlaw:/opt/edullm/experiments/token-selection:/opt/edullm/experiments/curriculum \
    OLMO_FLASH_ATTENTION=1 \
    OLMO_ATTN_BACKEND=flash_2 \
    OLMO_FUSED_LOSS=0 \
    TOKENIZERS_PARALLELISM=false

# Build-time contract: CUDA-linked torch, every MixLaw runtime import, the
# checked-in ladder config, and the exact 20-label OLMES BPB suite.
RUN python .edullm/verify_image.py

CMD ["python", "/opt/edullm/experiments/skill-dag/mixlaw/platform_array_entrypoint.py"]
