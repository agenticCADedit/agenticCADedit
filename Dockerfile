# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=1

# ---------------------------------------------------------------------------
# System dependencies
#   - build tools / git / archive utils
#   - Python 3.12 via deadsnakes
#   - OpenGL + xvfb for headless CadQuery/VTK rendering
#   - ffmpeg + libsm6/libxext6 for video decoding (decord2)
# ---------------------------------------------------------------------------
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        ninja-build \
        unzip \
        pigz \
        p7zip-full \
        software-properties-common \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libxi6 \
        libxkbcommon0 \
        xvfb \
        && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Python dependencies (installed before the source copy for better caching)
# ---------------------------------------------------------------------------
RUN pip install \
        itables \
        pandas \
        pyarrow \
        mongita \
        ray \
        vllm \
        decord2 \
        cadquery \
        openai \
        anthropic \
        google-genai \
        json-repair \
        scikit-learn \
        open3d \
        mcp

# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
ARG PROJECT_DIR=/workspace/AgenticCADedit
ENV PROJECT_DIR=${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}
COPY . ${PROJECT_DIR}

# Make the repo importable and define generic, mount-friendly data locations.
# Override any of these at run time with `-e NAME=value`.
ENV PYTHONPATH=${PROJECT_DIR} \
    DATA_DIR=/data \
    OUTPUT_DIR=/data/outputs \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    DISPLAY=:99

RUN mkdir -p /data /data/outputs /cache

# Optional: install the project itself if it ships a pyproject/setup.py
# RUN pip install -e .

CMD ["bash"]