from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CategoryCount(BaseModel):
    code: str
    name: str | None = None
    count: int


class StatsMeta(BaseModel):
    total: int
    limit: int
    generated_at: datetime


class SummaryResponse(BaseModel):
    total_papers: int
    total_authors: int
    total_categories: int
    date_range: dict[str, str] | None = None
    top_categories: list[CategoryCount]


class TopCategoriesResponse(BaseModel):
    data: list[CategoryCount]
    meta: StatsMeta


class TimeSeriesPoint(BaseModel):
    period: str
    category: str
    count: int


class TimeSeriesResponse(BaseModel):
    data: list[TimeSeriesPoint]


class AuthorCount(BaseModel):
    author: str
    paper_count: int


class TopAuthorsResponse(BaseModel):
    data: list[AuthorCount]


class VelocityPoint(BaseModel):
    period: str
    count: int


class PublicationVelocityResponse(BaseModel):
    data: list[VelocityPoint]


class CoOccurrenceEntry(BaseModel):
    category_a: str
    category_b: str
    cooccurrences: int


class CategoryCoOccurrenceResponse(BaseModel):
    data: list[CoOccurrenceEntry]
