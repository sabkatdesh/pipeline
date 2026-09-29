# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building native extensions (psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Runtime system deps only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder.
COPY --from=builder /install /usr/local

# Copy application source.
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY app/ ./app/
COPY scripts/ ./scripts/

# Directory for BM25 index and concept graph (mounted as a volume in compose).
RUN mkdir -p /app/data

# Non-root user for security.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
