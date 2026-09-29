"""Pydantic v2 request/response models for the RAG endpoints."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    none = "none"


class AskRequest(BaseModel):
    # No max_length on purpose: spec error matrix says >1000 chars is truncated
    # silently before embedding (done in the pipeline), not rejected.
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    session_id: str | None = Field(None, max_length=128)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Source(BaseModel):
    arxiv_id: str
    title: str
    similarity_score: float
    reranker_score: float | None = None
    retrieval_source: Literal["dense", "sparse", "graph"]
    published_date: date
    url: str
    concept_overlap: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: ConfidenceLevel
    iterations: int
    faithfulness_passed: bool | None = None  # None when no answer was generated
    model_used: str
    retrieval_ms: int
    llm_ms: int


class EvalSample(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)


class EvaluateRequest(BaseModel):
    test_set: list[EvalSample] = Field(..., min_length=1, max_length=50)


class EvaluateResponse(BaseModel):
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    num_samples: int