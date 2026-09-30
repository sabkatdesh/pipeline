# Slim image: no PyTorch (embeddings run on ONNX via fastembed), so it builds in
# a couple of minutes and stays well under 1 GB.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic/ ./alembic/
COPY alembic.ini .
COPY app/ ./app/
COPY scripts/ ./scripts/

# BM25 index, concept graph and embedding model cache live here (a compose volume).
RUN useradd -m appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
