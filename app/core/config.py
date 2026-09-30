from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(..., description="Async PostgreSQL DSN (postgresql+asyncpg://...)")
    db_pool_size: int = Field(10, ge=1, le=50)
    db_max_overflow: int = Field(20, ge=0, le=100)

    # ── Embedding ────────────────────────────────────────────────────────────
    # local  = fastembed (ONNX, no API key, 384-dim)  <- default
    # openai = OpenAI API        azure = Azure OpenAI deployment
    embedding_provider: str = Field("local", pattern="^(openai|azure|local)$")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(384, ge=1)
    embedding_cache_dir: str = Field("data/models")
    openai_api_key: str | None = Field(None)

    # ── Azure OpenAI (optional: embeddings and/or chat fallback) ─────────────
    azure_openai_api_base: str | None = Field(None)
    azure_openai_api_key: str | None = Field(None)
    azure_openai_api_version: str = Field("2024-10-21")
    azure_openai_embedding_deployment: str | None = Field(None)
    azure_openai_chat_deployment: str | None = Field(None)

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Anthropic is used when ANTHROPIC_API_KEY is set; otherwise Azure chat.
    anthropic_api_key: str | None = Field(None)
    llm_model: str = Field("claude-haiku-4-5-20251001")
    llm_max_tokens: int = Field(1024, ge=1)

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_top_k: int = Field(5, ge=1, le=20)
    # Cosine-similarity cut-offs (calibrated for MiniLM / text-embedding-3-small).
    rag_similarity_threshold: float = Field(0.30, ge=0.0, le=1.0)  # below = "no match"
    rag_medium_threshold: float = Field(0.45, ge=0.0, le=1.0)
    rag_high_threshold: float = Field(0.60, ge=0.0, le=1.0)
    rag_max_retry_iterations: int = Field(2, ge=1)
    rag_bm25_index_path: str = Field("data/bm25_index.pkl")
    rag_graph_path: str = Field("data/concept_graph.pkl")

    # ── Ingestion ────────────────────────────────────────────────────────────
    arxiv_rate_limit_seconds: float = Field(3.5, ge=0.0)
    arxiv_batch_size: int = Field(500, ge=1, le=2000)
    arxiv_categories: str = Field("cs.AI,cs.LG,cs.CL")
    arxiv_max_papers: int = Field(3500, ge=1)
    ingest_date_from: str = Field("2026-01-01")
    ingest_date_to: str = Field("2026-09-29")
    embedding_batch_size: int = Field(100, ge=1)
    concept_extraction_batch_size: int = Field(20, ge=1)

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1, le=65535)
    log_level: str = Field("INFO")
    reset_confirmation_token: str = Field("my-secret-reset-token")

    @property
    def azure_configured(self) -> bool:
        return bool(self.azure_openai_api_base and self.azure_openai_api_key)

    @property
    def arxiv_categories_list(self) -> list[str]:
        return [c.strip() for c in self.arxiv_categories.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance. Safe to call from anywhere."""
    return Settings()
