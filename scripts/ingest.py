"""
CLI entry point for running ingestion outside of the HTTP API.

Usage:
    python -m scripts.ingest [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--categories cs.AI,cs.LG] [--reset]

The script creates an ingestion_run record, runs the pipeline synchronously,
and exits with a non-zero code on failure.
"""

import argparse
import asyncio
import sys
from datetime import date, datetime, timezone

from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger, setup_logging
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.reset import wipe_dataset
from app.models.ingestion_run import IngestionRun

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the pipeline database.")
    parser.add_argument(
        "--from",
        dest="date_from",
        default=cfg.ingest_date_from,
        metavar="YYYY-MM-DD",
        help=f"Start date (default: {cfg.ingest_date_from})",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        default=cfg.ingest_date_to,
        metavar="YYYY-MM-DD",
        help=f"End date (default: {cfg.ingest_date_to})",
    )
    parser.add_argument(
        "--categories",
        default=cfg.arxiv_categories,
        help=f"Comma-separated arXiv categories (default: {cfg.arxiv_categories})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all papers, authors, categories, embeddings and indexes, then exit.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    cfg = get_settings()
    setup_logging(cfg.log_level)

    if args.reset:
        async with AsyncSessionLocal() as session:
            await wipe_dataset(session)
            await session.commit()
        print("Dataset wiped. Run the ingest command again for a fresh import.")
        return

    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    logger.info(
        "cli_ingestion_start",
        date_from=str(date_from),
        date_to=str(date_to),
        categories=categories,
    )

    async with AsyncSessionLocal() as session:
        run = IngestionRun(
            status="running",
            date_from=datetime.combine(date_from, datetime.min.time()).replace(tzinfo=timezone.utc),
            date_to=datetime.combine(date_to, datetime.min.time()).replace(tzinfo=timezone.utc),
            categories=categories,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        run_id = run.id

    logger.info("ingestion_run_created", run_id=run_id)

    pipeline = IngestionPipeline(run_id=run_id)
    await pipeline.run(categories=categories, date_from=date_from, date_to=date_to)

    # Print a summary from the DB.
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(IngestionRun).where(IngestionRun.id == run_id))
        finished = result.scalar_one()

    print(
        f"\nIngestion complete\n"
        f"  Run ID  : {finished.id}\n"
        f"  Status  : {finished.status}\n"
        f"  Fetched : {finished.papers_fetched}\n"
        f"  Inserted: {finished.papers_inserted}\n"
        f"  Updated : {finished.papers_updated}\n"
    )


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nIngestion cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nIngestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
