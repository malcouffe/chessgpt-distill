FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/hf_cache
ENV HF_DATASETS_TIMEOUT=120
ENV TRITON_CACHE_DIR=/tmp/.triton

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN apt-get update && apt-get install -y --no-install-recommends git gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY chessgpt_distill/ chessgpt_distill/
RUN pip install --no-cache-dir .

COPY scripts/ scripts/
COPY configs/ configs/

ENTRYPOINT ["python", "scripts/train_distill.py"]
CMD ["--config", "configs/distill_h100.yaml"]
