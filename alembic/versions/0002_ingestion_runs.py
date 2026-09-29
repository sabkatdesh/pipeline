"""Add ingestion_runs audit table

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("date_from", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("date_to", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("categories", ARRAY(sa.Text), nullable=False),
        sa.Column("papers_fetched", sa.Integer, nullable=False, server_default="0"),
        sa.Column("papers_inserted", sa.Integer, nullable=False, server_default="0"),
        sa.Column("papers_updated", sa.Integer, nullable=False, server_default="0"),
        sa.Column("papers_embedded", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_checkpoint", JSONB, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
