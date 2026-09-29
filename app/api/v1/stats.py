"""
Visualization / Stats API endpoints.

GET /api/v1/stats/summary
GET /api/v1/stats/top-categories
GET /api/v1/stats/papers-by-category-over-time
GET /api/v1/stats/top-authors
GET /api/v1/stats/publication-velocity
GET /api/v1/stats/category-cooccurrence
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.stats import (
    AuthorCount,
    CategoryCoOccurrenceResponse,
    CategoryCount,
    CoOccurrenceEntry,
    PublicationVelocityResponse,
    StatsMeta,
    SummaryResponse,
    TimeSeriesPoint,
    TimeSeriesResponse,
    TopAuthorsResponse,
    TopCategoriesResponse,
    VelocityPoint,
)

router = APIRouter(prefix="/stats", tags=["stats"])


class Granularity(str, Enum):
    month = "month"
    year = "year"
    week = "week"


_GRANULARITY_FMT: dict[Granularity, str] = {
    Granularity.month: "YYYY-MM",
    Granularity.year: "YYYY",
    Granularity.week: 'IYYY-"W"IW',
}


@router.get(
    "/summary",
    response_model=SummaryResponse,
    summary="Dataset summary statistics",
)
async def get_summary(db: AsyncSession = Depends(get_db)) -> SummaryResponse:
    total_papers = (await db.scalar(text("SELECT COUNT(*) FROM papers"))) or 0
    total_authors = (await db.scalar(text("SELECT COUNT(*) FROM authors"))) or 0
    total_categories = (await db.scalar(text("SELECT COUNT(*) FROM categories"))) or 0

    date_row = (
        await db.execute(
            text("SELECT MIN(published_date)::date, MAX(published_date)::date FROM papers")
        )
    ).one()
    date_min, date_max = date_row

    top_rows = (
        await db.execute(
            text("""
                SELECT c.code, c.display_name, COUNT(pc.paper_id) AS cnt
                FROM categories c
                JOIN paper_categories pc ON c.code = pc.category_code
                GROUP BY c.code, c.display_name
                ORDER BY cnt DESC
                LIMIT 5
            """)
        )
    ).all()

    return SummaryResponse(
        total_papers=total_papers,
        total_authors=total_authors,
        total_categories=total_categories,
        date_range=(
            {"from": str(date_min), "to": str(date_max)}
            if date_min is not None
            else None
        ),
        top_categories=[
            CategoryCount(code=r.code, name=r.display_name, count=r.cnt)
            for r in top_rows
        ],
    )


@router.get(
    "/top-categories",
    response_model=TopCategoriesResponse,
    summary="Top categories by paper count, optionally filtered by date range",
)
async def get_top_categories(
    limit: int = Query(10, ge=1, le=100),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    db: AsyncSession = Depends(get_db),
) -> TopCategoriesResponse:
    params: dict = {"limit": limit, "from_date": from_date, "to_date": to_date}

    # Always join papers so date filters are available; NULL params are treated as "no filter".
    data_rows = (
        await db.execute(
            text("""
                SELECT c.code, c.display_name, COUNT(pc.paper_id) AS cnt
                FROM categories c
                JOIN paper_categories pc ON c.code = pc.category_code
                JOIN papers p ON pc.paper_id = p.arxiv_id
                WHERE (:from_date IS NULL OR p.published_date >= :from_date)
                  AND (:to_date   IS NULL OR p.published_date <= :to_date)
                GROUP BY c.code, c.display_name
                ORDER BY cnt DESC
                LIMIT :limit
            """),
            params,
        )
    ).all()

    total = (
        await db.scalar(
            text("""
                SELECT COUNT(DISTINCT c.code)
                FROM categories c
                JOIN paper_categories pc ON c.code = pc.category_code
                JOIN papers p ON pc.paper_id = p.arxiv_id
                WHERE (:from_date IS NULL OR p.published_date >= :from_date)
                  AND (:to_date   IS NULL OR p.published_date <= :to_date)
            """),
            params,
        )
    ) or 0

    return TopCategoriesResponse(
        data=[
            CategoryCount(code=r.code, name=r.display_name, count=r.cnt)
            for r in data_rows
        ],
        meta=StatsMeta(
            total=total,
            limit=limit,
            generated_at=datetime.now(timezone.utc),
        ),
    )


@router.get(
    "/papers-by-category-over-time",
    response_model=TimeSeriesResponse,
    summary="Paper counts per category grouped by time period",
)
async def get_papers_by_category_over_time(
    granularity: Granularity = Query(Granularity.month),
    category: str | None = Query(None, description="Filter to a single category code"),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    db: AsyncSession = Depends(get_db),
) -> TimeSeriesResponse:
    fmt = _GRANULARITY_FMT[granularity]
    params: dict = {
        "fmt": fmt,
        "from_date": from_date,
        "to_date": to_date,
        "category": category,
    }

    rows = (
        await db.execute(
            text("""
                SELECT TO_CHAR(p.published_date, :fmt) AS period,
                       pc.category_code                AS category,
                       COUNT(*)                        AS cnt
                FROM papers p
                JOIN paper_categories pc ON p.arxiv_id = pc.paper_id
                WHERE (:from_date IS NULL OR p.published_date >= :from_date)
                  AND (:to_date   IS NULL OR p.published_date <= :to_date)
                  AND (:category  IS NULL OR pc.category_code = :category)
                GROUP BY TO_CHAR(p.published_date, :fmt), pc.category_code
                ORDER BY period, category
            """),
            params,
        )
    ).all()

    return TimeSeriesResponse(
        data=[
            TimeSeriesPoint(period=r.period, category=r.category, count=r.cnt)
            for r in rows
        ]
    )


@router.get(
    "/top-authors",
    response_model=TopAuthorsResponse,
    summary="Top authors by paper count, optionally filtered by date range",
)
async def get_top_authors(
    limit: int = Query(20, ge=1, le=100),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    db: AsyncSession = Depends(get_db),
) -> TopAuthorsResponse:
    params: dict = {"limit": limit, "from_date": from_date, "to_date": to_date}

    rows = (
        await db.execute(
            text("""
                SELECT a.name                   AS author,
                       COUNT(pa.paper_id)       AS paper_count
                FROM authors a
                JOIN paper_authors pa ON a.id = pa.author_id
                JOIN papers p         ON pa.paper_id = p.arxiv_id
                WHERE (:from_date IS NULL OR p.published_date >= :from_date)
                  AND (:to_date   IS NULL OR p.published_date <= :to_date)
                GROUP BY a.id, a.name
                ORDER BY paper_count DESC
                LIMIT :limit
            """),
            params,
        )
    ).all()

    return TopAuthorsResponse(
        data=[AuthorCount(author=r.author, paper_count=r.paper_count) for r in rows]
    )


@router.get(
    "/publication-velocity",
    response_model=PublicationVelocityResponse,
    summary="Total paper submissions per time period across all categories",
)
async def get_publication_velocity(
    granularity: Granularity = Query(Granularity.week),
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    db: AsyncSession = Depends(get_db),
) -> PublicationVelocityResponse:
    fmt = _GRANULARITY_FMT[granularity]
    params: dict = {"fmt": fmt, "from_date": from_date, "to_date": to_date}

    rows = (
        await db.execute(
            text("""
                SELECT TO_CHAR(published_date, :fmt) AS period,
                       COUNT(*)                      AS cnt
                FROM papers
                WHERE (:from_date IS NULL OR published_date >= :from_date)
                  AND (:to_date   IS NULL OR published_date <= :to_date)
                GROUP BY TO_CHAR(published_date, :fmt)
                ORDER BY period
            """),
            params,
        )
    ).all()

    return PublicationVelocityResponse(
        data=[VelocityPoint(period=r.period, count=r.cnt) for r in rows]
    )


@router.get(
    "/category-cooccurrence",
    response_model=CategoryCoOccurrenceResponse,
    summary="How often pairs of categories appear together on the same paper",
)
async def get_category_cooccurrence(
    limit: int = Query(20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> CategoryCoOccurrenceResponse:
    rows = (
        await db.execute(
            text("""
                SELECT pc1.category_code AS category_a,
                       pc2.category_code AS category_b,
                       COUNT(*)          AS cooccurrences
                FROM paper_categories pc1
                JOIN paper_categories pc2
                  ON pc1.paper_id = pc2.paper_id
                 AND pc1.category_code < pc2.category_code
                GROUP BY pc1.category_code, pc2.category_code
                ORDER BY cooccurrences DESC
                LIMIT :limit
            """),
            {"limit": limit},
        )
    ).all()

    return CategoryCoOccurrenceResponse(
        data=[
            CoOccurrenceEntry(
                category_a=r.category_a,
                category_b=r.category_b,
                cooccurrences=r.cooccurrences,
            )
            for r in rows
        ]
    )
