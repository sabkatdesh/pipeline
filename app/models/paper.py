from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.author import PaperAuthor
    from app.models.category import Category, PaperCategory
    from app.models.concept import PaperConcept


def _embedding_dim() -> int:
    # Lazy import so settings are only resolved at first use, not import time.
    from app.core.config import get_settings

    return get_settings().embedding_dim


class Paper(Base):
    """An arXiv paper with its metadata and optional embedding vector."""

    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=False)
    abstract_clean: Mapped[str] = mapped_column(Text, nullable=False)
    published_date: Mapped[datetime] = mapped_column(nullable=False)
    updated_date: Mapped[datetime] = mapped_column(nullable=False)
    doi: Mapped[str | None] = mapped_column(Text)
    journal_ref: Mapped[str | None] = mapped_column(Text)
    primary_category_code: Mapped[str] = mapped_column(
        Text, ForeignKey("categories.code"), nullable=False
    )

    # NULL until the embedding indexer runs; dimension set from EMBEDDING_DIM env var.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_embedding_dim()))
    embedding_model: Mapped[str | None] = mapped_column(Text)
    embedding_updated_at: Mapped[datetime | None] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    primary_category: Mapped[Category] = relationship(
        "Category",
        foreign_keys=[primary_category_code],
        back_populates="papers",
    )
    paper_authors: Mapped[list[PaperAuthor]] = relationship(
        "PaperAuthor", back_populates="paper", cascade="all, delete-orphan"
    )
    paper_categories: Mapped[list[PaperCategory]] = relationship(
        "PaperCategory", back_populates="paper", cascade="all, delete-orphan"
    )
    paper_concepts: Mapped[list[PaperConcept]] = relationship(
        "PaperConcept", back_populates="paper", cascade="all, delete-orphan"
    )
