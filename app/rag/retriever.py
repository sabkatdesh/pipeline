"""
HybridRetriever: pgvector (dense) + BM25 (sparse) + concept graph, fused with RRF.

Also holds the shared RetrievedDoc type and the confidence/threshold logic.
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import literal, select, text

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.embeddings.indexer import load_bm25_index
from app.embeddings.provider import EmbeddingError, get_embedding_provider
from app.models.paper import Paper
from app.rag.graph.builder import load_concept_graph
from app.rag.graph.traverser import traverse
from app.schemas.rag import ConfidenceLevel

logger = get_logger(__name__)

RRF_K = 60
DENSE_TOP = 20
SPARSE_TOP = 20
GRAPH_TOP = 10
FUSED_TOP = 20
IVFFLAT_PROBES = 20
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.70
_MAX_QUERY_CHARS = 2000


@dataclass
class RetrievedDoc:
    arxiv_id: str
    title: str
    abstract_clean: str
    published_date: date
    primary_category: str | None = None
    similarity_score: float = 0.0          # cosine similarity to the question
    reranker_score: float | None = None    # 0-1 (sigmoid of cross-encoder logit)
    retrieval_source: str = "dense"        # dense | sparse | graph
    rrf_score: float = 0.0
    concept_overlap: int | None = None


# ── Pure logic (unit-tested without a DB) ─────────────────────────────────────

def confidence_from_score(score: float | None, threshold: float | None = None) -> ConfidenceLevel:
    """
    >= 0.80 high, >= 0.70 medium, >= threshold (RAG_SIMILARITY_THRESHOLD, 0.65) low,
    below that none ("no relevant papers").
    """
    if threshold is None:
        threshold = get_settings().rag_similarity_threshold
    if score is None:
        return ConfidenceLevel.none
    if score >= HIGH_THRESHOLD:
        return ConfidenceLevel.high
    if score >= MEDIUM_THRESHOLD:
        return ConfidenceLevel.medium
    if score >= threshold:
        return ConfidenceLevel.low
    return ConfidenceLevel.none


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]],
    top_n: int = FUSED_TOP,
    k: int = RRF_K,
) -> list[tuple[str, float, str]]:
    """
    Reciprocal Rank Fusion. Returns (id, rrf_score, source) best first, where
    source is the retrieval leg that ranked the paper highest (ties: dict order).
    """
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, tuple[int, int, str]] = {}
    for order, (leg, ids) in enumerate(rankings.items()):
        for rank, pid in enumerate(ids, start=1):
            scores[pid] += 1.0 / (k + rank)
            cand = (rank, order, leg)
            if pid not in best or cand < best[pid]:
                best[pid] = cand
    ranked = sorted(scores, key=lambda p: (-scores[p], p))[:top_n]
    return [(p, scores[p], best[p][2]) for p in ranked]


# ── Index caches (reload when ingestion rewrites the file) ────────────────────

class _FileCache:
    def __init__(self, path_fn: Callable[[], str], loader: Callable[[str], Any]) -> None:
        self._path_fn = path_fn
        self._loader = loader
        self._value: Any = None
        self._mtime: float | None = None
        self._missing_logged = False

    def get(self) -> Any:
        path = self._path_fn()
        try:
            mtime = os.stat(path).st_mtime
        except FileNotFoundError:
            if not self._missing_logged:
                logger.warning("index_file_missing", path=path)
                self._missing_logged = True
            self._value, self._mtime = None, None
            return None
        self._missing_logged = False
        if mtime != self._mtime:
            self._value = self._loader(path)
            self._mtime = mtime
        return self._value


_bm25_cache = _FileCache(lambda: get_settings().rag_bm25_index_path, load_bm25_index)
_graph_cache = _FileCache(lambda: get_settings().rag_graph_path, load_concept_graph)


async def has_embeddings() -> bool:
    async with AsyncSessionLocal() as session:
        return bool(
            await session.scalar(
                text("SELECT EXISTS (SELECT 1 FROM papers WHERE embedding IS NOT NULL)")
            )
        )


class HybridRetriever:
    async def retrieve(
        self, query: str, variants: list[str], hyde: str = ""
    ) -> list[RetrievedDoc]:
        """Dense + sparse + graph -> RRF -> up to 20 unique candidates."""
        query = query[:_MAX_QUERY_CHARS]
        variants = [v[:_MAX_QUERY_CHARS] for v in variants]
        texts = [query, *variants, *([hyde[:_MAX_QUERY_CHARS]] if hyde else [])]

        vectors: list[list[float]] = []
        try:
            vectors = await get_embedding_provider().embed(texts)
        except EmbeddingError as exc:  # degrade to sparse + graph
            logger.warning("query_embedding_failed", error=str(exc))

        keyword_texts = [query, *variants]
        dense, sparse, graph = await asyncio.gather(
            self._dense(vectors),
            asyncio.to_thread(self._sparse, keyword_texts),
            asyncio.to_thread(self._graph, keyword_texts),
        )
        overlaps = dict(graph)
        fused = rrf_fuse(
            {"dense": dense, "sparse": sparse, "graph": [pid for pid, _ in graph]}
        )
        return await self._hydrate(fused, vectors[0] if vectors else None, overlaps)

    async def _dense(self, vectors: list[list[float]]) -> list[str]:
        if not vectors:
            return []
        best: dict[str, float] = {}
        async with AsyncSessionLocal() as session:
            await session.execute(text(f"SET LOCAL ivfflat.probes = {IVFFLAT_PROBES}"))
            for vec in vectors:
                dist = Paper.embedding.cosine_distance(vec).label("dist")
                rows = await session.execute(
                    select(Paper.arxiv_id, dist)
                    .where(Paper.embedding.is_not(None))
                    .order_by(dist)
                    .limit(DENSE_TOP)
                )
                for pid, d in rows:
                    best[pid] = max(best.get(pid, -1.0), 1.0 - float(d))
        return sorted(best, key=lambda p: (-best[p], p))[:DENSE_TOP]

    def _sparse(self, texts: list[str]) -> list[str]:
        index = _bm25_cache.get()
        if index is None:  # spec: fall back to dense-only
            return []
        best: dict[str, float] = {}
        for t in texts:
            for pid, score in index.search(t, SPARSE_TOP):
                best[pid] = max(best.get(pid, 0.0), score)
        return sorted(best, key=lambda p: (-best[p], p))[:SPARSE_TOP]

    def _graph(self, texts: list[str]) -> list[tuple[str, int]]:
        graph = _graph_cache.get()
        if graph is None:  # spec: skip graph hop
            return []
        return traverse(graph, texts, max_results=GRAPH_TOP)

    async def _hydrate(
        self,
        fused: list[tuple[str, float, str]],
        question_vec: list[float] | None,
        overlaps: dict[str, int],
    ) -> list[RetrievedDoc]:
        if not fused:
            return []
        sim = (
            (1 - Paper.embedding.cosine_distance(question_vec))
            if question_vec is not None
            else literal(0.0)
        )
        ids = [pid for pid, _, _ in fused]
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    select(
                        Paper.arxiv_id,
                        Paper.title,
                        Paper.abstract_clean,
                        Paper.published_date,
                        Paper.primary_category_code,
                        sim.label("sim"),
                    ).where(Paper.arxiv_id.in_(ids))
                )
            ).all()
        by_id = {r.arxiv_id: r for r in rows}
        docs: list[RetrievedDoc] = []
        for pid, rrf, source in fused:
            r = by_id.get(pid)
            if r is None:
                continue
            docs.append(
                RetrievedDoc(
                    arxiv_id=pid,
                    title=r.title,
                    abstract_clean=r.abstract_clean,
                    published_date=r.published_date.date(),
                    primary_category=r.primary_category_code,
                    similarity_score=float(r.sim or 0.0),
                    retrieval_source=source,
                    rrf_score=rrf,
                    concept_overlap=overlaps.get(pid),
                )
            )
        return docs