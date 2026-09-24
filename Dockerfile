FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/.config \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/models \
    HF_HOME=/app/.cache/huggingface \
    HF_XET_CACHE=/app/.cache/huggingface/xet

WORKDIR /app

RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --no-create-home --shell /usr/sbin/nologin appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the free embedding model into the image. Runtime containers do not need
# Hugging Face network access or a writable shared model volume.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/.cache/models')"

COPY data ./data
COPY src ./src
COPY scripts ./scripts
COPY infra ./infra

RUN mkdir -p /app/data/index /app/.cache/models /app/.cache/huggingface/xet \
    && chown -R appuser:appgroup /app

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER 10001:10001

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

  CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]