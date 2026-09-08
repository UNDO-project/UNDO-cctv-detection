# CPU-only image for the CCTV detection Gradio UI.
#
# Model weights, training runs and example images are NOT baked in; mount them
# read-only at runtime (see docker-compose.yml). torch resolves to the CPU wheel
# on Linux via the pytorch-cpu index declared in pyproject.toml.

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# opencv-python (pulled in by ultralytics) needs libGL and glib at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    CCTV_LOG_LEVEL=INFO

WORKDIR /app

# Dependency layer: cached until pyproject.toml or uv.lock change.
# --no-install-project keeps the project itself out of this layer so source
# edits do not invalidate the (large) dependency install. The cache mount keeps
# uv's download cache (~1.7 GB) out of the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Ultralytics writes a settings file here at import; pre-create it so it does
# not warn and fall back to /tmp
RUN mkdir -p /root/.config/Ultralytics

# Source layer
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py data.yaml ./

EXPOSE 7860

CMD ["uv", "run", "--no-sync", "python", "app.py"]
