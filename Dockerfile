FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache
ENV HF_DATASETS_TIMEOUT=120
ENV TRITON_CACHE_DIR=/tmp/.triton

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

ARG GITHUB_TOKEN
RUN apt-get update && apt-get install -y --no-install-recommends git gcc \
    && rm -rf /var/lib/apt/lists/*

# Install chessgpt dependency first (needs GitHub auth if repo is private)
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN=$(cat /run/secrets/github_token 2>/dev/null || echo "${GITHUB_TOKEN}") && \
    pip install --no-cache-dir "chessgpt @ git+https://${GITHUB_TOKEN}@github.com/malcouffe/ChessGPT.git"

COPY pyproject.toml .
COPY chessgpt_distill/ chessgpt_distill/
RUN pip install --no-cache-dir --no-deps .

COPY scripts/ scripts/
COPY configs/ configs/

ENTRYPOINT ["python", "scripts/train_distill.py"]
CMD ["--config", "configs/distill_h100.yaml"]
