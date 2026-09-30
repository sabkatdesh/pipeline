import asyncio
import time

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://export.arxiv.org/api/query"
_MAX_RETRIES = 5
_RETRYABLE_STATUSES = {429, 503}


class ArxivClient:
    """
    Async HTTP client for the arXiv Atom API.

    Handles:
      - Rate limiting (configurable gap between requests)
      - Exponential backoff on 429 / 503 and timeouts
      - Up to _MAX_RETRIES attempts before raising

    Usage (async context manager):
        async with ArxivClient() as client:
            xml_bytes = await client.fetch(...)
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._http: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> "ArxivClient":
        self._http = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "arxiv-rag-pipeline/1.0 (take-home assessment; httpx)"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._http:
            await self._http.aclose()

    async def fetch(
        self,
        category: str,
        date_from: str,
        date_to: str,
        start: int = 0,
        max_results: int | None = None,
    ) -> bytes:
        """
        Fetch one page of arXiv results and return the raw Atom/XML bytes.

        Args:
            category:    arXiv category code, e.g. 'cs.AI'
            date_from:   ISO date string, e.g. '2026-01-01'
            date_to:     ISO date string, e.g. '2026-01-07'
            start:       Pagination offset.
            max_results: Papers per page; defaults to ARXIV_BATCH_SIZE.
        """
        if max_results is None:
            max_results = self._cfg.arxiv_batch_size

        params = {
            # httpx percent-encodes the spaces; do not pre-encode with "+".
            "search_query": (
                f"cat:{category} AND "
                f"submittedDate:[{_compact(date_from)}0000 TO {_compact(date_to)}2359]"
            ),
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }

        await self._wait_for_rate_limit()

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._http.get(_BASE_URL, params=params)
                self._last_request_time = time.monotonic()

                if response.status_code in _RETRYABLE_STATUSES:
                    wait = _backoff(attempt, self._cfg.arxiv_rate_limit_seconds)
                    retry_after = response.headers.get("retry-after", "")
                    if retry_after.isdigit():
                        wait = max(wait, min(float(retry_after), 120.0))
                    logger.warning(
                        "arxiv_throttled",
                        status=response.status_code,
                        attempt=attempt,
                        retry_in=wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                logger.debug(
                    "arxiv_fetched",
                    category=category,
                    start=start,
                    max_results=max_results,
                    bytes=len(response.content),
                )
                return response.content

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"arXiv request failed: {exc!r}") from exc
                wait = _backoff(attempt, self._cfg.arxiv_rate_limit_seconds)
                logger.warning("arxiv_timeout", attempt=attempt, retry_in=wait)
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"arXiv API throttled/unreachable after {_MAX_RETRIES} attempts "
            f"(category={category}, start={start})"
        )

    async def _wait_for_rate_limit(self) -> None:
        """Sleep if needed to respect the configured request interval."""
        if self._last_request_time:
            elapsed = time.monotonic() - self._last_request_time
            gap = self._cfg.arxiv_rate_limit_seconds - elapsed
            if gap > 0:
                await asyncio.sleep(gap)


def _compact(date_str: str) -> str:
    """'2026-01-01' -> '20260101' (date part of arXiv's YYYYMMDDHHMM format)."""
    return date_str.replace("-", "")


def _backoff(attempt: int, base: float) -> float:
    """Exponential backoff: base, 2×base, 4×base, 8×base, 16×base."""
    return base * (2 ** (attempt - 1))
