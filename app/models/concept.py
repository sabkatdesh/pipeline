"""ORM models for concepts + paper_concepts (migration 0005). Skip if you already have this."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.paper import Paper


class Concept(Base):
    __tablename__ = "concepts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    concept_type: Mapped[str | None] = mapped_column(Text)  # method|model|dataset|task|other

    paper_concepts: Mapped[list[PaperConcept]] = relationship(
        "PaperConcept", back_populates="concept"
    )


class PaperConcept(Base):
    __tablename__ = "paper_concepts"

    paper_id: Mapped[str] = mapped_column(
        Text, ForeignKey("papers.arxiv_id", ondelete="CASCADE"), primary_key=True
    )
    concept_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("concepts.id"), primary_key=True
    )

    paper: Mapped[Paper] = relationship("Paper", back_populates="paper_concepts")
    concept: Mapped[Concept] = relationship("Concept", back_populates="paper_concepts")