# satnogs-signal live-service image. The container IS the environment — no host virtualenv.
FROM python:3.14-slim

# libgomp1: OpenMP runtime used by numpy/torch on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch/torchvision FIRST, from the PyTorch CPU index. The default PyPI
# wheels bundle the full NVIDIA CUDA stack (multi-GB, useless on this CPU-only image); by
# pre-satisfying the torch requirement with the +cpu builds, the editable install below
# reuses them instead of pulling the CUDA variants.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the runtime package only (no dev/test extras). torch/torchvision are already
# satisfied above. The source tree is bind-mounted at runtime (see docker-compose.yml), so
# the editable install resolves to the live code without rebuilding.
COPY pyproject.toml ./
COPY satnogs_signal ./satnogs_signal
# scripts too: deployments that build this image straight from git (no bind
# mount) run /app/scripts/run_poller.py — the image must carry its own tools.
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

CMD ["python", "-c", "import satnogs_signal; print('satnogs-signal container ready (no venv)')"]
