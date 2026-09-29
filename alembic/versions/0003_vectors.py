"""Enable pgvector extension and add embedding column + IVFFlat index

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29
"""

import os
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Read dimension from env at migration time so it matches the model.
_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# IVFFlat lists ≈ sqrt(expected row count).  Sized for ~3 500 rows initially;
# rebuild the index after a large bulk load with CREATE INDEX CONCURRENTLY.
_IVFFLAT_LISTS = 60


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "papers",
        sa.Column(
            "embedding",
            sa.Text,  # placeholder type; replaced below with raw DDL
            nullable=True,
        ),
    )
    # Replace the placeholder TEXT column with the real vector type.
    op.execute(f"ALTER TABLE papers ALTER COLUMN embedding TYPE vector({_EMBEDDING_DIM}) USING NULL")

    op.execute(
        f"""
        CREATE INDEX idx_papers_embedding
        ON papers
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = {_IVFFLAT_LISTS})
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_papers_embedding")
    op.drop_column("papers", "embedding")
