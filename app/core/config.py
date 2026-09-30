from functools import lru_cache

from pydantic import Field, model_validator
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
    embedding_provider: str = Field("openai", pattern="^(openai|local)$")
    openai_api_key: str | None = Field(None)
    # Azure OpenAI / Foundry (optional). If present, these values will be used
    # to configure the OpenAI client to talk to an Azure OpenAI endpoint.
    azure_openai_api_base: str | None = Field(None)
    azure_openai_api_key: str | None = Field(None)
    azure_openai_api_version: str | None = Field(None)
    embedding_model: str = Field("text-embedding-3-small")
    embedding_dim: int = Field(1536, ge=1)

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str | None = Field(None)
    llm_model: str = Field("claude-haiku-4-5-20251001")
    llm_max_tokens: int = Field(1024, ge=1)

    # ── RAG ──────────────────────────────────────────────────────────────────
    rag_top_k: int = Field(5, ge=1, le=20)
    rag_similarity_threshold: float = Field(0.65, ge=0.0, le=1.0)
    rag_max_retry_iterations: int = Field(3, ge=1)
    rag_reranker_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rag_bm25_index_path: str = Field("/app/data/bm25_index.pkl")
    rag_graph_path: str = Field("/app/data/concept_graph.pkl")

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

    @model_validator(mode="after")
    def _check_openai_key(self) -> "Settings":
        # Allow either an OpenAI API key (openai.com) or Azure OpenAI settings
        # (base + key) when EMBEDDING_PROVIDER=openai.
        if self.embedding_provider == "openai" and not (
            self.openai_api_key or (self.azure_openai_api_key and self.azure_openai_api_base)
        ):
            raise ValueError(
                "OPENAI_API_KEY or AZURE_OPENAI_API_KEY (with AZURE_OPENAI_API_BASE) is required when EMBEDDING_PROVIDER=openai"
            )
        return self

    @property
    def arxiv_categories_list(self) -> list[str]:
        return [c.strip() for c in self.arxiv_categories.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance. Safe to call from anywhere."""
    return Settings()
