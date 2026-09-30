"""RAG API endpoints: /api/v1/rag/search and /api/v1/rag/ask

The `/ask` endpoint supports optional Server-Sent Events streaming when
`AskRequest.stream` is true.
"""
from __future__ import annotations

import json
from time import perf_counter
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.langgraph_pipeline import SimpleRAGPipeline
from app.rag.retriever import HybridRetriever
from app.rag.answerer import LLMUnavailableError, LLMRateLimitError
from app.schemas.rag import AskRequest, AskResponse, Source, ConfidenceLevel

router = APIRouter(prefix="/rag", tags=["rag"])
logger = get_logger(__name__)


def _sse_event(event: str | None, payload: dict) -> str:
    body = json.dumps(payload, default=str, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {body}\n\n"
    return f"data: {body}\n\n"


@router.get("/search")
async def search(q: str, top_k: int = 10):
    """Simple retrieval-only endpoint (dense+sparse+graph fusion)."""
    retriever = HybridRetriever()
    t0 = perf_counter()
    docs = await retriever.retrieve(q, variants=[])
    retrieval_ms = int((perf_counter() - t0) * 1000)

    results = []
    for d in docs[:top_k]:
        results.append(
            {
                "arxiv_id": d.arxiv_id,
                "title": d.title,
                "similarity_score": d.similarity_score,
                "retrieval_source": d.retrieval_source,
                "published_date": d.published_date.isoformat(),
                "url": f"https://arxiv.org/abs/{d.arxiv_id}",
                "concept_overlap": d.concept_overlap,
            }
        )
    return {"query": q, "retrieval_ms": retrieval_ms, "results": results}


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """Run CRAG loop -> retrieve -> generate -> faithfulness check.

    If `body.stream` is True, returns an SSE stream of staged events.
    """
    cfg = get_settings()
    pipeline = SimpleRAGPipeline()

    # Streaming SSE flow
    if body.stream:
        async def _stream() -> AsyncIterator[str]:
            try:
                out = await pipeline.ask(body.question, top_k=body.top_k)
                docs = out.get("docs", [])
                retrieval_ms = out.get("retrieval_ms", 0)
                yield _sse_event(
                    "retrieval",
                    {
                        "retrieval_ms": retrieval_ms,
                        "docs": [
                            {
                                "arxiv_id": d.arxiv_id,
                                "title": d.title,
                                "similarity_score": d.similarity_score,
                                "retrieval_source": d.retrieval_source,
                                "published_date": d.published_date.isoformat(),
                                "url": f"https://arxiv.org/abs/{d.arxiv_id}",
                                "concept_overlap": d.concept_overlap,
                            }
                            for d in docs
                        ],
                    },
                )

                yield _sse_event(
                    "answer",
                    {
                        "answer": out.get("answer", ""),
                        "model_used": out.get("model_used"),
                        "llm_ms": out.get("llm_ms", 0),
                    },
                )

                yield _sse_event("faithfulness", {"passed": out.get("faithfulness_passed")})

                yield _sse_event(
                    "done",
                    {
                        "confidence": (out.get("confidence") or ConfidenceLevel.none).value,
                        "iterations": out.get("iterations", 0),
                    },
                )
            except LLMRateLimitError as exc:
                yield _sse_event("error", {"status": 429, "detail": str(exc)})
            except LLMUnavailableError as exc:
                yield _sse_event("error", {"status": 503, "detail": str(exc)})
            except Exception as exc:
                logger.exception("ask_stream_failed", err=str(exc))
                yield _sse_event("error", {"status": 500, "detail": "internal error"})

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # Non-streaming (regular) flow
    try:
        out = await pipeline.ask(body.question, top_k=body.top_k)
    except LLMRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    docs = out.get("docs", [])
    sources = []
    for d in docs[: body.top_k]:
        sources.append(
            Source(
                arxiv_id=d.arxiv_id,
                title=d.title,
                similarity_score=d.similarity_score,
                reranker_score=d.reranker_score,
                retrieval_source=d.retrieval_source,
                published_date=d.published_date,
                url=f"https://arxiv.org/abs/{d.arxiv_id}",
                concept_overlap=d.concept_overlap,
            )
        )

    confidence = out.get("confidence") or ConfidenceLevel.none

    return AskResponse(
        answer=out.get("answer", ""),
        sources=sources,
        confidence=confidence,
        iterations=out.get("iterations", 0),
        faithfulness_passed=out.get("faithfulness_passed"),
        model_used=out.get("model_used", cfg.llm_model),
        retrieval_ms=out.get("retrieval_ms", 0),
        llm_ms=out.get("llm_ms", 0),
    )