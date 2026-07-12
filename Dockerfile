# Synapse API image (Phase 11/12 — one-command stack).
# Runs the FastAPI app; reaches host Ollama via host.docker.internal, neo4j/redis over the
# compose network. Deps pinned from requirements.lock.txt (no torch/sentence-transformers).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Pinned deps first (layer-cached across code changes). Drop pywin32 — it's a Windows-only
# pin in the lock (the venv was frozen on Windows) and has no Linux distribution.
COPY requirements.lock.txt .
RUN grep -ivE '^pywin32' requirements.lock.txt > /tmp/reqs.txt && pip install -r /tmp/reqs.txt

# App code (shadowed by a bind-mount in dev compose; present for a standalone image).
COPY synapse ./synapse

EXPOSE 8848
CMD ["uvicorn", "synapse.api.main:app", "--host", "0.0.0.0", "--port", "8848"]
