"""
Embedding providers.

  LocalProvider   - fastembed/ONNX all-MiniLM-L6-v2 (default, no API key, 384-dim)
  OpenAIProvider  - OpenAI text-embedding-3-small (1536-dim) or an Azure OpenAI deployment

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
    """OpenAI or Azure OpenAI embeddings (EMBEDDING_PROVIDER=openai|azure)."""

    def __init__(self, azure: bool = False) -> None:
        import openai

        cfg = get_settings()
        self.dim = cfg.embedding_dim
        if azure:
            if not cfg.azure_configured:
                raise EmbeddingConfigError(
                    "EMBEDDING_PROVIDER=azure needs AZURE_OPENAI_API_BASE and AZURE_OPENAI_API_KEY"
                )
            # On Azure the "model" is the deployment name.
            self.model = cfg.azure_openai_embedding_deployment or cfg.embedding_model
            self._client = openai.AsyncAzureOpenAI(
                azure_endpoint=cfg.azure_openai_api_base.rstrip("/"),
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.azure_openai_api_version,
                max_retries=0,
                timeout=60.0,
            )
        else:
            if not cfg.openai_api_key:
                raise EmbeddingConfigError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY")
            self.model = cfg.embedding_model
            self._client = openai.AsyncOpenAI(api_key=cfg.openai_api_key, max_retries=0, timeout=60.0)
        # text-embedding-3-* accept a `dimensions` param; ada-002 does not.
        self._send_dimensions = "text-embedding-3" in (cfg.embedding_model or self.model)

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
    """fastembed (ONNX runtime). Model weights are downloaded once into EMBEDDING_CACHE_DIR."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.model = cfg.embedding_model
        self.dim = cfg.embedding_dim
        if self.model.startswith("text-embedding"):
            raise EmbeddingConfigError(
                "EMBEDDING_PROVIDER=local needs a fastembed model, e.g. "
                "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 and EMBEDDING_DIM=384"
            )
        try:
            from fastembed import TextEmbedding  # lazy: loads onnxruntime

            self._model = TextEmbedding(model_name=self.model, cache_dir=cfg.embedding_cache_dir)
        except Exception as exc:
            raise EmbeddingConfigError(f"could not load local embedding model {self.model!r}: {exc}") from exc

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _run() -> list[list[float]]:
            return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

        # CPU-bound and synchronous: keep it off the event loop.
        return await asyncio.to_thread(_run)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    cfg = get_settings()
    if cfg.embedding_provider == "openai":
        provider: EmbeddingProvider = OpenAIProvider(azure=False)
    elif cfg.embedding_provider == "azure":
        provider = OpenAIProvider(azure=True)
    else:
        provider = LocalProvider()

    if provider.dim != cfg.embedding_dim:
        raise EmbeddingConfigError(
            f"Provider produces {provider.dim}-dim vectors but EMBEDDING_DIM={cfg.embedding_dim}"
        )
    return provider
