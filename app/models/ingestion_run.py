from __future__ import annotations

from datetime import datetime

from sqlalchemy import ARRAY, BigInteger, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class IngestionRun(Base):
    """Audit log for every ingestion run, including resume checkpoints."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column()
    # running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(Text, nullable=False)
    date_from: Mapped[datetime] = mapped_column(nullable=False)
    date_to: Mapped[datetime] = mapped_column(nullable=False)
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    papers_fetched: Mapped[int] = mapped_column(Integer, default=0)
    papers_inserted: Mapped[int] = mapped_column(Integer, default=0)
    papers_updated: Mapped[int] = mapped_column(Integer, default=0)
    papers_embedded: Mapped[int] = mapped_column(Integer, default=0)
    # JSON checkpoint used to resume interrupted runs.
    # Schema: {"done": ["cs.AI:2026-01-01", ...], "current": {"category": ..., "offset": ...}}
    last_checkpoint: Mapped[dict | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
