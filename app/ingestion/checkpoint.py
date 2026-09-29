"""
Checkpoint helpers for resumable ingestion.

The `last_checkpoint` JSONB column on `ingestion_runs` stores:
{
    "done":    ["cs.AI:2026-01-01", "cs.AI:2026-01-08", ...],   # completed windows
    "current": {"category": "cs.AI", "date_window": "2026-01-15", "offset": 500}
}

This lets the pipeline skip already-finished windows and resume mid-window
after a crash or restart.
"""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.ingestion_run import IngestionRun

logger = get_logger(__name__)


async def get_checkpoint(session: AsyncSession, run_id: int) -> dict | None:
    """Return the current checkpoint dict for the given run, or None."""
    result = await session.execute(
        select(IngestionRun.last_checkpoint).where(IngestionRun.id == run_id)
    )
    return result.scalar_one_or_none()


async def save_progress(
    session: AsyncSession,
    run_id: int,
    category: str,
    date_window: str,
    offset: int,
) -> None:
    """Record how far we got inside a window so we can resume after a restart."""
    cp = await get_checkpoint(session, run_id) or {}
    cp["current"] = {"category": category, "date_window": date_window, "offset": offset}
    await _write(session, run_id, cp)
    logger.debug("checkpoint_progress", run_id=run_id, category=category, window=date_window, offset=offset)


async def complete_window(
    session: AsyncSession,
    run_id: int,
    category: str,
    date_window: str,
) -> None:
    """Mark a date window as fully processed so it is skipped on resume."""
    cp = await get_checkpoint(session, run_id) or {}
    done: list[str] = cp.get("done", [])
    key = f"{category}:{date_window}"
    if key not in done:
        done.append(key)
    cp["done"] = done
    cp.pop("current", None)
    await _write(session, run_id, cp)
    logger.debug("window_complete", run_id=run_id, key=key)


def is_window_done(checkpoint: dict | None, category: str, date_window: str) -> bool:
    """Return True if this window was already completed in a previous run."""
    if not checkpoint:
        return False
    return f"{category}:{date_window}" in checkpoint.get("done", [])


def resume_offset(checkpoint: dict | None, category: str, date_window: str) -> int:
    """Return the offset to start from when resuming a mid-window run."""
    if not checkpoint:
        return 0
    current = checkpoint.get("current", {})
    if (
        current.get("category") == category
        and current.get("date_window") == date_window
    ):
        return int(current.get("offset", 0))
    return 0


# ── Private ───────────────────────────────────────────────────────────────────

async def _write(session: AsyncSession, run_id: int, cp: dict) -> None:
    await session.execute(
        update(IngestionRun)
        .where(IngestionRun.id == run_id)
        .values(last_checkpoint=cp)
    )
    await session.commit()
