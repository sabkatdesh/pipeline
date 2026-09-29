from datetime import datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    date_from: str = Field("2026-01-01", description="Start date (YYYY-MM-DD)")
    date_to: str = Field("2026-09-29", description="End date (YYYY-MM-DD)")
    categories: list[str] | None = Field(
        None,
        description="arXiv category codes; defaults to ARXIV_CATEGORIES env var",
    )


class IngestStarted(BaseModel):
    run_id: int
    status: str
    message: str


class IngestStatus(BaseModel):
    run_id: int
    status: str
    papers_fetched: int
    papers_inserted: int
    papers_updated: int
    papers_embedded: int
    started_at: datetime
    completed_at: datetime | None
    elapsed_seconds: float | None
    error_message: str | None


class ResetRequest(BaseModel):
    confirmation_token: str = Field(..., description="Must match RESET_CONFIRMATION_TOKEN")


class ResetResponse(BaseModel):
    message: str


"""Paper-facing response models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    abstract_clean: str
    published_date: date
    primary_category: str | None = None


class PaperSearchResult(PaperOut):
    similarity_score: float
    reranker_score: float | None = None
    concepts: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[PaperSearchResult]