# satnogs-signal dev/run image. The container IS the environment — no host virtualenv.
FROM python:3.14-slim

# libgomp1: OpenMP runtime used by numpy/torch on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package + dev tools (pytest) into the image. The source tree is
# bind-mounted at runtime (see docker-compose.yml), so the editable install resolves
# to the live code and tests/scripts are available without rebuilding.
COPY pyproject.toml ./
COPY satnogs_signal ./satnogs_signal
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-c", "import satnogs_signal; print('satnogs-signal container ready (no venv)')"]
