import os

# Settings are read at import time; give tests a DSN that is never actually connected to.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://pipeline:password@localhost:5432/pipeline")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("RESET_CONFIRMATION_TOKEN", "test-token")
