from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.paper import Paper


class Category(Base):
    """An arXiv subject category (e.g. cs.AI, cs.LG)."""

    __tablename__ = "categories"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(Text)

    papers: Mapped[list[Paper]] = relationship(
        "Paper",
        foreign_keys="Paper.primary_category_code",
        back_populates="primary_category",
    )
    paper_categories: Mapped[list[PaperCategory]] = relationship(
        "PaperCategory", back_populates="category"
    )


class PaperCategory(Base):
    """Join table linking papers to all their categories (including cross-listed)."""

    __tablename__ = "paper_categories"

    paper_id: Mapped[str] = mapped_column(
        Text, ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    category_code: Mapped[str] = mapped_column(
        Text, ForeignKey("categories.code"), primary_key=True
    )
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)

    paper: Mapped[Paper] = relationship("Paper", back_populates="paper_categories")
    category: Mapped[Category] = relationship("Category", back_populates="paper_categories")
