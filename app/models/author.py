from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, SmallInteger, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.paper import Paper


class Author(Base):
    """A unique paper author, deduplicated via a normalized name key."""

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    # Lowercase + unicode-NFC + whitespace-collapsed — used for deduplication.
    name_normalized: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    paper_authors: Mapped[list[PaperAuthor]] = relationship(
        "PaperAuthor", back_populates="author"
    )


class PaperAuthor(Base):
    """Join table: paper ↔ author with author order preserved."""

    __tablename__ = "paper_authors"

    paper_id: Mapped[str] = mapped_column(
        Text, ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    author_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("authors.id"), primary_key=True
    )
    # 0-based position; 0 = first / corresponding author.
    position: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    paper: Mapped[Paper] = relationship("Paper", back_populates="paper_authors")
    author: Mapped[Author] = relationship("Author", back_populates="paper_authors")
