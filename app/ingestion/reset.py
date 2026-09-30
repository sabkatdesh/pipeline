"""Wipe the dataset (papers, authors, categories, concepts, embeddings, indexes)."""

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Embeddings live on papers.embedding, so truncating papers removes them too.
_TABLES = (
    "paper_concepts", "paper_categories", "paper_authors",
    "papers", "authors", "concepts", "ingestion_runs",
)


async def wipe_dataset(db: AsyncSession) -> None:
    """Delete all ingested data. The caller commits (get_db does on success)."""
    for table in _TABLES:
        await db.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    # Keep the seeded reference categories (they have display names); drop the rest
    # that ingestion discovered.
    await db.execute(text("DELETE FROM categories WHERE display_name IS NULL"))

    cfg = get_settings()
    for path in (cfg.rag_bm25_index_path, cfg.rag_graph_path):
        Path(path).unlink(missing_ok=True)
    logger.info("dataset_wiped")
