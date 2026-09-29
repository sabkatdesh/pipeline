"""Initial schema: categories, papers, authors, paper_authors, paper_categories

Revision ID: 0001
Revises:
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "categories",
        sa.Column("code", sa.Text, primary_key=True),
        sa.Column("display_name", sa.Text, nullable=True),
    )

    op.create_table(
        "papers",
        sa.Column("arxiv_id", sa.Text, primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("abstract", sa.Text, nullable=False),
        sa.Column("abstract_clean", sa.Text, nullable=False),
        sa.Column("published_date", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_date", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("doi", sa.Text, nullable=True),
        sa.Column("journal_ref", sa.Text, nullable=True),
        sa.Column(
            "primary_category_code",
            sa.Text,
            sa.ForeignKey("categories.code"),
            nullable=False,
        ),
        sa.Column("embedding_model", sa.Text, nullable=True),
        sa.Column("embedding_updated_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("idx_papers_published_date", "papers", ["published_date"])
    op.create_index("idx_papers_primary_category", "papers", ["primary_category_code"])
    op.create_index("idx_papers_updated_date", "papers", ["updated_date"])

    op.create_table(
        "authors",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("name_normalized", sa.Text, nullable=False, unique=True),
    )
    op.create_index("idx_authors_normalized", "authors", ["name_normalized"])

    op.create_table(
        "paper_authors",
        sa.Column(
            "paper_id",
            sa.Text,
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "author_id",
            sa.BigInteger,
            sa.ForeignKey("authors.id"),
            primary_key=True,
        ),
        sa.Column("position", sa.SmallInteger, nullable=False),
    )
    op.create_index("idx_paper_authors_paper_id", "paper_authors", ["paper_id"])
    op.create_index("idx_paper_authors_author_id", "paper_authors", ["author_id"])

    op.create_table(
        "paper_categories",
        sa.Column(
            "paper_id",
            sa.Text,
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "category_code",
            sa.Text,
            sa.ForeignKey("categories.code"),
            primary_key=True,
        ),
        sa.Column("is_primary", sa.Boolean, nullable=False, server_default="false"),
    )
    op.create_index("idx_paper_categories_code", "paper_categories", ["category_code"])
    op.create_index("idx_paper_categories_paper_id", "paper_categories", ["paper_id"])
    op.create_index(
        "idx_paper_categories_primary",
        "paper_categories",
        ["category_code"],
        postgresql_where=sa.text("is_primary = TRUE"),
    )


def downgrade() -> None:
    op.drop_table("paper_categories")
    op.drop_table("paper_authors")
    op.drop_table("authors")
    op.drop_table("papers")
    op.drop_table("categories")
