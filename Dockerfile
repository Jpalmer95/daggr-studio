# Daggr Studio runs on the plain `docker` SDK so we control Python and gradio versions
# instead of inheriting them from the Space runtime (daggr pins gradio>=6 and the
# gradio-sdk runtime also pre-binds :7860, which fights our own uvicorn).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/tmp/huggingface

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Probe fixtures are generated at build time so the first registry check is fast.
RUN python -c "import sys; sys.path.insert(0,'.'); from scripts.verify_registry import probe_fixtures; print(probe_fixtures())" || true

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:7860/healthz || exit 1

CMD ["python", "app.py"]