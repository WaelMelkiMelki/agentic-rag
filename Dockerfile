FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_DIR=/opt/fastembed

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src/ ./src/
COPY configs/ ./configs/
RUN FASTEMBED_CACHE_DIR=/opt/fastembed \
    .venv/bin/python -c 'from fastembed import TextEmbedding; TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir="/opt/fastembed")'

FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/configs /app/configs
COPY --from=builder --chown=appuser:appuser /opt/fastembed /opt/fastembed

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_DIR=/opt/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import json,os,urllib.request; url='http://127.0.0.1:'+os.getenv('PORT','8000')+'/health'; assert json.load(urllib.request.urlopen(url, timeout=4))['status']=='ok'" || exit 1

CMD ["sh", "-c", "exec uvicorn agentic_rag.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
