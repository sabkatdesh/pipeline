"""Add concepts and paper_concepts tables for the concept graph

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "concepts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        # method | model | dataset | task | other
        sa.Column("concept_type", sa.Text, nullable=True),
    )
    op.create_index("idx_concepts_name", "concepts", ["name"])

    op.create_table(
        "paper_concepts",
        sa.Column(
            "paper_id",
            sa.Text,
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "concept_id",
            sa.BigInteger,
            sa.ForeignKey("concepts.id"),
            primary_key=True,
        ),
    )
    op.create_index("idx_paper_concepts_paper_id", "paper_concepts", ["paper_id"])
    op.create_index("idx_paper_concepts_concept_id", "paper_concepts", ["concept_id"])


def downgrade() -> None:
    op.drop_table("paper_concepts")
    op.drop_table("concepts")
