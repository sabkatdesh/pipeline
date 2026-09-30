"""
EmbeddingIndexer - embeds papers that have no (or a stale) embedding.

A paper is pending when:
  * embedding IS NULL, or
  * it was re-ingested after arXiv revised it (embedding_updated_at < updated_at), or
  * it was embedded with a different model than the one currently configured.

This module also owns the BM25 sparse index (build_bm25_index / load_bm25_index):
it is another index built over the ingested papers, serialized to RAG_BM25_INDEX_PATH.

Batch failures are skipped (papers stay pending for the next run); bad credentials
or a dimension mismatch abort immediately. After embedding, the IVFFlat index is
rebuilt against the real data (see rebuild_ivfflat_index for why).
"""

from __future__ import annotations

import asyncio
import math
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import bindparam, func, text, update

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.embeddings.provider import (
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingProvider,
    get_embedding_provider,
)
from app.models.ingestion_run import IngestionRun
from app.models.paper import Paper

logger = get_logger(__name__)

_MAX_CONSECUTIVE_FAILURES = 3
_INDEX_NAME = "idx_papers_embedding"

_PENDING_SQL = text(
    """
    SELECT arxiv_id, title, abstract_clean
    FROM papers
    WHERE arxiv_id > :last_id
      AND (
            embedding IS NULL
         OR embedding_model IS DISTINCT FROM :model
         OR embedding_updated_at < updated_at
      )
    ORDER BY arxiv_id
    LIMIT :limit
    """
)


async def assert_embedding_dim(expected_dim: int) -> None:
    """
    Fail fast if EMBEDDING_DIM disagrees with the pgvector column
    (spec: 'Raise at startup if EMBEDDING_DIM does not match pgvector column size').
    Also safe to call from the FastAPI lifespan hook.
    """
    async with AsyncSessionLocal() as session:
        column_dim = (
            await session.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'papers'::regclass AND attname = 'embedding'"
                )
            )
        ).scalar_one_or_none()

    if column_dim is None:
        raise EmbeddingConfigError("papers.embedding column not found - run migration 0003")
    if column_dim != expected_dim:
        raise EmbeddingConfigError(
            f"papers.embedding is vector({column_dim}) but EMBEDDING_DIM={expected_dim}. "
            "Change the env var or re-create the column."
        )


class EmbeddingIndexer:
    def __init__(
        self,
        run_id: int | None = None,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self._run_id = run_id
        self._provider = provider
        self._cfg = get_settings()
        self.batches_failed = 0

    async def run(self) -> int:
        """Embed every pending paper. Returns the number embedded."""
        provider = self._provider or get_embedding_provider()
        await assert_embedding_dim(provider.dim)

        batch_size = self._cfg.embedding_batch_size
        total = 0
        last_id = ""
        consecutive_failures = 0

        while True:
            async with AsyncSessionLocal() as session:
                rows = (
                    await session.execute(
                        _PENDING_SQL,
                        {"last_id": last_id, "model": provider.model, "limit": batch_size},
                    )
                ).all()
            if not rows:
                break
            # Keyset pagination: a skipped batch can't be re-selected in this run.
            last_id = rows[-1].arxiv_id

            ids: list[str] = []
            texts: list[str] = []
            for r in rows:
                doc = f"{r.title}\n\n{r.abstract_clean}".strip()
                if doc:  # OpenAI rejects empty strings
                    ids.append(r.arxiv_id)
                    texts.append(doc)
            if not ids:
                continue

            try:
                vectors = await provider.embed(texts)
            except EmbeddingConfigError:
                raise
            except EmbeddingError as exc:
                self.batches_failed += 1
                consecutive_failures += 1
                logger.error("embedding_batch_skipped", error=str(exc), size=len(ids))
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    raise EmbeddingError(
                        f"{consecutive_failures} consecutive embedding batches failed; aborting"
                    ) from exc
                continue
            consecutive_failures = 0

            await self._store(provider, ids, vectors)
            total += len(ids)
            logger.info("embedding_batch_done", size=len(ids), total=total)

        if total > 0:
            lists = await rebuild_ivfflat_index()
            logger.info("ivfflat_rebuilt", lists=lists, embedded=total)
        return total

    async def _store(
        self, provider: EmbeddingProvider, ids: list[str], vectors: list[list[float]]
    ) -> None:
        papers = Paper.__table__
        stmt = (
            update(papers)
            .where(papers.c.arxiv_id == bindparam("b_id"))
            .values(
                embedding=bindparam("b_vec"),
                embedding_model=provider.model,
                embedding_updated_at=func.now(),
            )
        )
        async with AsyncSessionLocal() as session:
            await session.execute(
                stmt, [{"b_id": i, "b_vec": v} for i, v in zip(ids, vectors, strict=True)]
            )
            if self._run_id is not None:
                await session.execute(
                    update(IngestionRun)
                    .where(IngestionRun.id == self._run_id)
                    .values(papers_embedded=IngestionRun.papers_embedded + len(ids))
                )
            await session.commit()


async def rebuild_ivfflat_index() -> int | None:
    """
    Drop and recreate the IVFFlat index sized to the actual row count.

    Migration 0003 creates the index on an EMPTY table, so its centroids are
    meaningless and recall is poor until it's rebuilt on real data. Also the
    spec's lists=370 is for 135k rows; for our ~3.5k rows it's ~sqrt(n).
    Returns the lists value used, or None if nothing is embedded yet.
    """
    async with AsyncSessionLocal() as session:
        n = (
            await session.execute(text("SELECT count(*) FROM papers WHERE embedding IS NOT NULL"))
        ).scalar_one()
        if n == 0:
            return None
        lists = max(1, round(math.sqrt(n)))
        await session.execute(text(f"DROP INDEX IF EXISTS {_INDEX_NAME}"))
        await session.execute(
            text(
                f"CREATE INDEX {_INDEX_NAME} ON papers "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {int(lists)})"
            )
        )
        await session.commit()
    return lists


# ── BM25 sparse index ─────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "was were we with which our their these those using use used based via can also".split()
)


def tokenize(text_: str) -> list[str]:
    """Lowercase alnum tokens, minus stopwords. Used for BOTH docs and queries."""
    return [t for t in _TOKEN_RE.findall(text_.lower()) if t not in _STOPWORDS]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    arxiv_ids: list[str]
    built_at: str

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Top-k (arxiv_id, score), best first. Zero-score docs are dropped."""
        tokens = tokenize(query)
        if not tokens or not self.arxiv_ids:
            return []
        scores = self.bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.arxiv_ids[i], float(scores[i])) for i in top if scores[i] > 0]


def build_bm25_index_from_rows(rows: list[tuple[str, str, str]]) -> BM25Index:
    """Pure function: (arxiv_id, title, abstract) rows -> BM25 index."""
    return BM25Index(
        bm25=BM25Okapi([tokenize(f"{title}\n{abstract}") for _, title, abstract in rows]),
        arxiv_ids=[pid for pid, _, _ in rows],
        built_at=datetime.now(timezone.utc).isoformat(),
    )


async def build_bm25_index(path: str | None = None) -> int:
    """Rebuild the index from every paper in the DB. Returns document count."""
    path = path or get_settings().rag_bm25_index_path

    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                text("SELECT arxiv_id, title, abstract_clean FROM papers ORDER BY arxiv_id")
            )
        ).all()
    if not rows:
        logger.warning("bm25_skipped_no_papers")
        return 0

    index = await asyncio.to_thread(
        build_bm25_index_from_rows, [(r.arxiv_id, r.title, r.abstract_clean) for r in rows]
    )  # CPU-bound
    await asyncio.to_thread(_atomic_pickle_dump, index, path)
    logger.info("bm25_built", docs=len(rows), path=path)
    return len(rows)


def load_bm25_index(path: str | None = None) -> BM25Index | None:
    path = path or get_settings().rag_bm25_index_path
    index = _safe_pickle_load(path)
    if not isinstance(index, BM25Index):
        logger.warning("bm25_index_unavailable", path=path)
        return None
    return index


# ── Pickle helpers (private) ──────────────────────────────────────────────────

def _atomic_pickle_dump(obj: Any, path: str | Path) -> None:
    """Write to a temp file then os.replace(), so readers never see a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _safe_pickle_load(path: str | Path) -> Any | None:
    """
    Load a pickle this app wrote earlier. Returns None if missing or unreadable
    so callers can degrade (spec: no BM25 / no graph -> skip that retrieval leg).
    Only ever point this at files produced by _atomic_pickle_dump.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        with target.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # corrupt / version-skewed pickle
        logger.warning("pickle_load_failed", path=str(target), error=str(exc))
        return None