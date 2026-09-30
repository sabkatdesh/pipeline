"""Ingestion and paper-facing request/response models."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    # If omitted, the handler will fall back to the ingest_date_from / ingest_date_to
    # environment-configured defaults in `app/core/config.py`.
    date_from: str | None = Field(None, description="Start date (YYYY-MM-DD)")
    date_to: str | None = Field(None, description="End date (YYYY-MM-DD)")
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
    # Date-window currently being processed (ISO date string, e.g. "2026-03-09").
    # This mirrors the checkpoint format stored in IngestionRun.last_checkpoint["current"]["date_window"].
    current_week: str | None = None


class ResetRequest(BaseModel):
    confirmation_token: str = Field(..., description="Must match RESET_CONFIRMATION_TOKEN")


class ResetResponse(BaseModel):
    message: str


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
