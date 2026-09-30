"""
IngestionPipeline — arXiv fetch → parse → clean → DB upsert.

Flow per category × date window:
  1. Skip if checkpoint says this window is done.
  2. Fetch pages from arXiv (rate-limited, retried).
  3. Parse XML → clean abstracts → normalize dates/authors.
  4. Bulk-upsert papers, authors, categories (idempotent).
  5. Save progress checkpoint; sleep between requests.
  6. Mark window complete.

After all windows, post-ingest stages run (each isolated - one failing never fails
the run, since the papers are already safely stored):
  a. EmbeddingIndexer      - embed new/changed papers, rebuild IVFFlat index
  b. build_bm25_index      - rebuild sparse index -> disk
  c. ConceptGraphBuilder   - LLM concept extraction -> concept graph -> disk

Then update ingestion_run stats and status. Stage problems are recorded in
error_message on a run that still finishes as "completed".
"""

import asyncio
import math
from datetime import date, timedelta

from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.embeddings.indexer import EmbeddingIndexer, build_bm25_index
from app.ingestion import checkpoint as chk
from app.ingestion.arxiv_client import ArxivClient
from app.ingestion.parser import RawPaper, parse_feed
from app.models.author import Author, PaperAuthor
from app.models.category import Category, PaperCategory
from app.models.ingestion_run import IngestionRun
from app.models.paper import Paper
from app.preprocessing.cleaner import clean_abstract
from app.preprocessing.normalizer import normalize_author_name, normalize_date
from app.rag.graph.builder import ConceptGraphBuilder

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates a full arXiv ingestion run.

    Each instance is tied to a single ingestion_run row (identified by run_id).
    """

    def __init__(self, run_id: int) -> None:
        self._run_id = run_id
        self._cfg = get_settings()
        self._total_fetched = 0
        self._total_inserted = 0
        self._total_updated = 0

    async def run(
        self,
        categories: list[str],
        date_from: date,
        date_to: date,
    ) -> None:
        logger.info("ingestion_started", run_id=self._run_id, categories=categories)

        try:
            async with ArxivClient() as client:
                # Windows outer, categories inner: if ARXIV_MAX_PAPERS cuts the run short,
                # every category is sampled across the whole date range.
                windows = _weekly_windows(date_from, date_to)
                # Per category-week quota so the paper cap yields an even sample of the range.
                quota = max(1, math.ceil(self._cfg.arxiv_max_papers / max(1, len(windows) * len(categories))))
                for win_start, win_end in windows:
                    if self._total_fetched >= self._cfg.arxiv_max_papers:
                        logger.info("max_papers_cap_reached", cap=self._cfg.arxiv_max_papers)
                        break
                    for category in categories:
                        if self._total_fetched >= self._cfg.arxiv_max_papers:
                            break
                        await self._process_window(client, category, win_start, win_end, quota)

            warnings = await self._run_post_ingest()
            await self._update_run(
                status="completed",
                error="; ".join(warnings)[:2000] if warnings else None,
            )
            logger.info(
                "ingestion_finished",
                run_id=self._run_id,
                fetched=self._total_fetched,
                inserted=self._total_inserted,
                updated=self._total_updated,
            )

        except Exception as exc:
            logger.exception("ingestion_failed", run_id=self._run_id, error=str(exc))
            await self._update_run(status="failed", error=str(exc))
            raise

    # ── Post-ingest stages ────────────────────────────────────────────────────

    async def _run_post_ingest(self) -> list[str]:
        """Run embeddings -> BM25 -> concept graph. Never raises; returns warnings."""
        stages = (
            ("embeddings", self._stage_embeddings),
            ("bm25", self._stage_bm25),
            ("concept_graph", self._stage_concept_graph),
        )
        warnings: list[str] = []
        for name, stage in stages:
            try:
                note = await stage()
            except Exception as exc:
                logger.exception("post_ingest_stage_failed", run_id=self._run_id, stage=name)
                warnings.append(f"{name} failed: {str(exc)[:300]}")
            else:
                logger.info("post_ingest_stage_done", run_id=self._run_id, stage=name)
                if note:
                    warnings.append(f"{name}: {note}")
        return warnings

    async def _stage_embeddings(self) -> str | None:
        indexer = EmbeddingIndexer(run_id=self._run_id)
        await indexer.run()
        if indexer.batches_failed:
            return f"{indexer.batches_failed} batch(es) skipped, will retry next run"
        return None

    async def _stage_bm25(self) -> str | None:
        await build_bm25_index()
        return None

    async def _stage_concept_graph(self) -> str | None:
        result = await ConceptGraphBuilder().run()
        notes = []
        if result.extraction_skipped:
            notes.append(f"extraction skipped ({result.extraction_skipped})")
        if result.batches_failed:
            notes.append(f"{result.batches_failed} extraction batch(es) failed")
        return ", ".join(notes) or None

    # ── Window processing ─────────────────────────────────────────────────────

    async def _process_window(
        self,
        client: ArxivClient,
        category: str,
        win_start: date,
        win_end: date,
        quota: int,
    ) -> None:
        window_key = str(win_start)

        async with AsyncSessionLocal() as session:
            cp = await chk.get_checkpoint(session, self._run_id)

        if chk.is_window_done(cp, category, window_key):
            logger.debug("window_skipped", category=category, window=window_key)
            return

        offset = chk.resume_offset(cp, category, window_key)
        logger.info("window_start", category=category, window=window_key, resume_offset=offset)

        while True:
            remaining = min(self._cfg.arxiv_max_papers - self._total_fetched, quota - offset)
            if remaining <= 0:
                break

            batch_size = min(self._cfg.arxiv_batch_size, remaining)

            try:
                xml_bytes = await client.fetch(
                    category=category,
                    date_from=str(win_start),
                    date_to=str(win_end),
                    start=offset,
                    max_results=batch_size,
                )
            except Exception as exc:
                # On an unrecoverable fetch failure, persist our current
                # checkpoint (so we can resume at the current offset) then
                # re-raise the exception to let the outer run() handler mark
                # the ingestion as failed.
                logger.exception("arxiv_fetch_failed_saving_checkpoint", category=category, window=window_key, offset=offset, error=str(exc))
                async with AsyncSessionLocal() as session:
                    await chk.save_progress(session, self._run_id, category, window_key, offset)
                raise

            try:
                feed = parse_feed(xml_bytes)
            except ValueError as exc:
                logger.error("xml_parse_error", error=str(exc), category=category, window=window_key)
                break

            if not feed.papers:
                break

            inserted, updated = await self._upsert_batch(feed.papers)

            self._total_fetched += len(feed.papers)
            self._total_inserted += inserted
            self._total_updated += updated
            offset += len(feed.papers)

            async with AsyncSessionLocal() as session:
                await chk.save_progress(session, self._run_id, category, window_key, offset)
                await self._flush_stats(session)

            logger.info(
                "batch_done",
                category=category,
                window=window_key,
                offset=offset,
                inserted=inserted,
                updated=updated,
            )

            if len(feed.papers) < batch_size:
                break  # last page for this window

            await asyncio.sleep(self._cfg.arxiv_rate_limit_seconds)

        async with AsyncSessionLocal() as session:
            await chk.complete_window(session, self._run_id, category, window_key)

    # ── Bulk DB upsert ────────────────────────────────────────────────────────

    async def _upsert_batch(self, raw_papers: list[RawPaper]) -> tuple[int, int]:
        """
        Bulk-upsert a batch of papers + their authors/categories.

        Returns (inserted_count, updated_count).
        """
        valid = [p for p in raw_papers if p.arxiv_id]
        if not valid:
            return 0, 0

        async with AsyncSessionLocal() as session:
            inserted, updated = await _upsert_papers(session, valid)
            await _upsert_authors(session, valid)
            await _upsert_categories(session, valid)
            await session.commit()

        return inserted, updated

    # ── Stats helpers ─────────────────────────────────────────────────────────

    async def _flush_stats(self, session: AsyncSession) -> None:
        await session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == self._run_id)
            .values(
                papers_fetched=self._total_fetched,
                papers_inserted=self._total_inserted,
                papers_updated=self._total_updated,
            )
        )

    async def _update_run(
        self,
        status: str,
        error: str | None = None,
    ) -> None:
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(IngestionRun)
                .where(IngestionRun.id == self._run_id)
                .values(
                    status=status,
                    completed_at=func.now(),
                    papers_fetched=self._total_fetched,
                    papers_inserted=self._total_inserted,
                    papers_updated=self._total_updated,
                    error_message=error,
                )
            )
            await session.commit()


# ── Bulk helpers (module-level for testability) ───────────────────────────────

async def _upsert_papers(
    session: AsyncSession, papers: list[RawPaper]
) -> tuple[int, int]:
    """
    Bulk INSERT … ON CONFLICT DO UPDATE for the papers table.

    Uses the PostgreSQL xmax trick to count inserted vs updated rows:
    xmax = 0 → new row (inserted); xmax != 0 → existing row (updated).
    """
    records = []
    for p in papers:
        pub = normalize_date(p.published)
        upd = normalize_date(p.updated, fallback=pub)
        primary = p.primary_category or (p.categories[0] if p.categories else "cs.AI")
        records.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title or "",
                "abstract": p.abstract or "",
                "abstract_clean": clean_abstract(p.abstract),
                "published_date": pub,
                "updated_date": upd,
                "doi": p.doi,
                "journal_ref": p.journal_ref,
                "primary_category_code": primary,
            }
        )

    stmt = (
        pg_insert(Paper)
        .values(records)
        .on_conflict_do_update(
            index_elements=["arxiv_id"],
            set_={
                "title": text("EXCLUDED.title"),
                "abstract": text("EXCLUDED.abstract"),
                "abstract_clean": text("EXCLUDED.abstract_clean"),
                "updated_date": text("EXCLUDED.updated_date"),
                "doi": text("EXCLUDED.doi"),
                "journal_ref": text("EXCLUDED.journal_ref"),
                "primary_category_code": text("EXCLUDED.primary_category_code"),
                "updated_at": func.now(),
            },
            # Only overwrite when arXiv has a newer version of the paper.
            where=text("papers.updated_date < EXCLUDED.updated_date"),
        )
        .returning(text("(xmax = 0)::boolean AS is_new"))
    )

    result = await session.execute(stmt)
    flags = [row[0] for row in result]
    inserted = sum(1 for f in flags if f)
    return inserted, len(flags) - inserted


async def _upsert_authors(session: AsyncSession, papers: list[RawPaper]) -> None:
    """
    Bulk-upsert all unique authors in the batch, then link them to their papers.

    Two queries per batch regardless of how many authors there are.
    """
    # Collect unique normalized names across all papers.
    name_map: dict[str, str] = {}  # normalized → display
    for p in papers:
        for name in p.authors:
            if name := name.strip():
                norm = normalize_author_name(name)
                name_map.setdefault(norm, name)

    if not name_map:
        return

    # 1. Bulk insert authors (skip duplicates).
    await session.execute(
        pg_insert(Author)
        .values([{"name": display, "name_normalized": norm} for norm, display in name_map.items()])
        .on_conflict_do_nothing(index_elements=["name_normalized"])
    )

    # 2. Fetch all IDs in one query.
    rows = await session.execute(
        select(Author.id, Author.name_normalized).where(
            Author.name_normalized.in_(name_map.keys())
        )
    )
    author_id_map: dict[str, int] = {row.name_normalized: row.id for row in rows}

    # 3. Bulk insert paper_authors.
    pa_records = [
        {"paper_id": p.arxiv_id, "author_id": author_id_map[norm], "position": idx}
        for p in papers
        if p.arxiv_id
        for idx, name in enumerate(p.authors)
        if name.strip()
        and (norm := normalize_author_name(name.strip())) in author_id_map
    ]
    if pa_records:
        await session.execute(
            pg_insert(PaperAuthor).values(pa_records).on_conflict_do_nothing()
        )


async def _upsert_categories(session: AsyncSession, papers: list[RawPaper]) -> None:
    """
    Ensure all category codes exist, then link papers to their categories.
    """
    all_codes = {code for p in papers for code in p.categories if code}
    if not all_codes:
        return

    # Ensure every category code exists (display_name filled in by seed migration).
    await session.execute(
        pg_insert(Category)
        .values([{"code": c, "display_name": None} for c in all_codes])
        .on_conflict_do_nothing(index_elements=["code"])
    )

    pc_records = [
        {
            "paper_id": p.arxiv_id,
            "category_code": code,
            "is_primary": code == (p.primary_category or (p.categories[0] if p.categories else "")),
        }
        for p in papers
        if p.arxiv_id
        for code in p.categories
        if code
    ]
    if pc_records:
        await session.execute(
            pg_insert(PaperCategory).values(pc_records).on_conflict_do_nothing()
        )


# ── Date windowing ────────────────────────────────────────────────────────────

def _weekly_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into sequential 7-day windows."""
    windows: list[tuple[date, date]] = []
    current = start
    while current <= end:
        win_end = min(current + timedelta(days=6), end)
        windows.append((current, win_end))
        current = win_end + timedelta(days=1)
    return windows
