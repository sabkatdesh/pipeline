"""
Ingestion API endpoints.

POST /api/v1/ingest/run    — start a background ingestion run
GET  /api/v1/ingest/status — poll status of the latest run
POST /api/v1/ingest/reset  — wipe all data for a clean re-import
"""

import asyncio
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.ingestion.pipeline import IngestionPipeline
from app.models.ingestion_run import IngestionRun
from app.schemas.paper import IngestRequest, IngestStarted, IngestStatus, ResetRequest, ResetResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = get_logger(__name__)

# Tracks the currently running asyncio task (if any).
_active_task: asyncio.Task | None = None


@router.post(
    "/run",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestStarted,
    summary="Start an ingestion run",
)
async def run_ingestion(
    body: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestStarted:
    """
    Launch a background ingestion job that fetches papers from arXiv,
    cleans them, and upserts them into the database.

    Returns 409 if an ingestion is already running.
    """
    global _active_task

    if _active_task and not _active_task.done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion already running. Check GET /api/v1/ingest/status.",
        )

    cfg = get_settings()
    categories = body.categories or cfg.arxiv_categories_list
    date_from = date.fromisoformat(body.date_from)
    date_to = date.fromisoformat(body.date_to)

    # Create the audit row before launching the task so callers always get a run_id.
    run = IngestionRun(
        status="running",
        date_from=datetime.combine(date_from, datetime.min.time()).replace(tzinfo=timezone.utc),
        date_to=datetime.combine(date_to, datetime.min.time()).replace(tzinfo=timezone.utc),
        categories=categories,
    )
    db.add(run)
    await db.flush()
    run_id = run.id

    pipeline = IngestionPipeline(run_id=run_id)
    _active_task = asyncio.create_task(
        pipeline.run(categories=categories, date_from=date_from, date_to=date_to),
        name=f"ingestion-run-{run_id}",
    )

    logger.info("ingestion_task_created", run_id=run_id, categories=categories)

    return IngestStarted(
        run_id=run_id,
        status="running",
        message=f"Ingestion started in background (run_id={run_id})",
    )


@router.get(
    "/status",
    response_model=IngestStatus,
    summary="Poll the latest ingestion run",
)
async def get_status(db: AsyncSession = Depends(get_db)) -> IngestStatus:
    """Return the status and progress counters of the most recent ingestion run."""
    result = await db.execute(
        select(IngestionRun).order_by(IngestionRun.id.desc()).limit(1)
    )
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ingestion run found. POST /api/v1/ingest/run to start one.",
        )

    now = datetime.now(timezone.utc)
    elapsed = (run.completed_at or now) - run.started_at

    return IngestStatus(
        run_id=run.id,
        status=run.status,
        papers_fetched=run.papers_fetched,
        papers_inserted=run.papers_inserted,
        papers_updated=run.papers_updated,
        papers_embedded=run.papers_embedded,
        started_at=run.started_at,
        completed_at=run.completed_at,
        elapsed_seconds=elapsed.total_seconds(),
        error_message=run.error_message,
    )


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Wipe all ingested data for a clean re-import",
)
async def reset_database(
    body: ResetRequest,
    db: AsyncSession = Depends(get_db),
) -> ResetResponse:
    """
    Truncates all paper data and clears embeddings so a fresh import can run.

    Requires `confirmation_token` to match the RESET_CONFIRMATION_TOKEN env var.
    Returns 409 if an ingestion is currently running.
    """
    cfg = get_settings()

    if body.confirmation_token != cfg.reset_confirmation_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid confirmation token.",
        )

    if _active_task and not _active_task.done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot reset while ingestion is running.",
        )

    # CASCADE on FK constraints handles dependent rows automatically.
    for table in ("paper_concepts", "paper_categories", "paper_authors", "papers", "authors", "concepts", "ingestion_runs"):
        await db.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))

    logger.info("database_reset")
    return ResetResponse(message="Database wiped. Ready for fresh import.")
