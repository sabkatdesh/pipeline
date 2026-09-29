"""
Embedding providers.

  OpenAIProvider  - text-embedding-3-small (default), 1536-dim
  LocalProvider   - sentence-transformers (e.g. all-MiniLM-L6-v2), 384-dim

Both expose `async embed(texts) -> list[list[float]]` (same order as input).
Retry policy (spec section 7): rate limit -> exponential backoff, 5 attempts;
5xx / connection errors -> 3 attempts; then EmbeddingError (caller skips the batch).
Auth errors and dimension mismatches raise EmbeddingConfigError (fatal, never skipped).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BACKOFF_BASE_SECONDS = 3.5  # same base as arxiv_client
_RATE_LIMIT_ATTEMPTS = 5
_SERVER_ERROR_ATTEMPTS = 3


class EmbeddingError(Exception):
    """Recoverable: this batch failed, later batches may still work."""


class EmbeddingConfigError(EmbeddingError):
    """Fatal: bad credentials, wrong model name, or dimension mismatch."""


class EmbeddingProvider(ABC):
    model: str
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class OpenAIProvider(EmbeddingProvider):
    def __init__(self) -> None:
        import openai

        cfg = get_settings()
        self.model = cfg.embedding_model
        self.dim = cfg.embedding_dim
        # text-embedding-3-* accept a `dimensions` param; ada-002 does not.
        self._send_dimensions = self.model.startswith("text-embedding-3")
        # max_retries=0: we own the retry policy so it matches the spec.
        self._client = openai.AsyncOpenAI(
            api_key=cfg.openai_api_key, max_retries=0, timeout=60.0
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import openai

        if not texts:
            return []

        kwargs: dict = {"model": self.model, "input": texts}
        if self._send_dimensions:
            kwargs["dimensions"] = self.dim

        rate_attempts = 0
        server_attempts = 0
        while True:
            try:
                resp = await self._client.embeddings.create(**kwargs)
                break
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise EmbeddingConfigError(f"OpenAI rejected credentials: {exc}") from exc
            except openai.NotFoundError as exc:
                raise EmbeddingConfigError(f"Unknown embedding model {self.model!r}") from exc
            except openai.RateLimitError as exc:
                rate_attempts += 1
                if rate_attempts >= _RATE_LIMIT_ATTEMPTS:
                    raise EmbeddingError(f"rate-limited {rate_attempts}x: {exc}") from exc
                delay = _BACKOFF_BASE_SECONDS * 2 ** (rate_attempts - 1)
                logger.warning("embedding_rate_limited", attempt=rate_attempts, sleep=delay)
                await asyncio.sleep(delay)
            except (openai.APIConnectionError, openai.InternalServerError) as exc:
                server_attempts += 1
                if server_attempts >= _SERVER_ERROR_ATTEMPTS:
                    raise EmbeddingError(f"OpenAI unavailable {server_attempts}x: {exc}") from exc
                delay = _BACKOFF_BASE_SECONDS * 2 ** (server_attempts - 1)
                logger.warning("embedding_server_error", attempt=server_attempts, sleep=delay)
                await asyncio.sleep(delay)
            except openai.APIError as exc:  # e.g. 400 bad input
                raise EmbeddingError(f"OpenAI API error: {exc}") from exc

        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} vectors, got {len(vectors)}")
        if vectors and len(vectors[0]) != self.dim:
            raise EmbeddingConfigError(
                f"{self.model} returned {len(vectors[0])}-dim vectors but EMBEDDING_DIM={self.dim}"
            )
        return vectors


class LocalProvider(EmbeddingProvider):
    def __init__(self) -> None:
        cfg = get_settings()
        self.model = cfg.embedding_model
        if self.model.startswith("text-embedding"):
            raise EmbeddingConfigError(
                "EMBEDDING_PROVIDER=local needs a sentence-transformers model, e.g. "
                "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 and EMBEDDING_DIM=384"
            )
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self._st = SentenceTransformer(self.model)
        self.dim = int(self._st.get_sentence_embedding_dimension())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Encoding is CPU-bound and synchronous: keep it off the event loop.
        vectors = await asyncio.to_thread(
            self._st.encode,
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    cfg = get_settings()
    provider: EmbeddingProvider = (
        OpenAIProvider() if cfg.embedding_provider == "openai" else LocalProvider()
    )
    if provider.dim != cfg.embedding_dim:
        raise EmbeddingConfigError(
            f"Provider produces {provider.dim}-dim vectors but EMBEDDING_DIM={cfg.embedding_dim}"
        )
    return provider