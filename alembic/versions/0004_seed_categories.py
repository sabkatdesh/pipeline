"""Seed the categories table with known arXiv CS subcategories

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Authoritative list of arXiv CS subcategories.
_CATEGORIES: list[tuple[str, str]] = [
    ("cs.AI", "Artificial Intelligence"),
    ("cs.AR", "Hardware Architecture"),
    ("cs.CC", "Computational Complexity"),
    ("cs.CE", "Computational Engineering, Finance, and Science"),
    ("cs.CG", "Computational Geometry"),
    ("cs.CL", "Computation and Language"),
    ("cs.CR", "Cryptography and Security"),
    ("cs.CV", "Computer Vision and Pattern Recognition"),
    ("cs.CY", "Computers and Society"),
    ("cs.DB", "Databases"),
    ("cs.DC", "Distributed, Parallel, and Cluster Computing"),
    ("cs.DL", "Digital Libraries"),
    ("cs.DM", "Discrete Mathematics"),
    ("cs.DS", "Data Structures and Algorithms"),
    ("cs.ET", "Emerging Technologies"),
    ("cs.FL", "Formal Languages and Automata Theory"),
    ("cs.GL", "General Literature"),
    ("cs.GR", "Graphics"),
    ("cs.GT", "Computer Science and Game Theory"),
    ("cs.HC", "Human-Computer Interaction"),
    ("cs.IR", "Information Retrieval"),
    ("cs.IT", "Information Theory"),
    ("cs.LG", "Machine Learning"),
    ("cs.LO", "Logic in Computer Science"),
    ("cs.MA", "Multiagent Systems"),
    ("cs.MM", "Multimedia"),
    ("cs.MS", "Mathematical Software"),
    ("cs.NA", "Numerical Analysis"),
    ("cs.NE", "Neural and Evolutionary Computing"),
    ("cs.NI", "Networking and Internet Architecture"),
    ("cs.OH", "Other Computer Science"),
    ("cs.OS", "Operating Systems"),
    ("cs.PF", "Performance"),
    ("cs.PL", "Programming Languages"),
    ("cs.RO", "Robotics"),
    ("cs.SC", "Symbolic Computation"),
    ("cs.SD", "Sound"),
    ("cs.SE", "Software Engineering"),
    ("cs.SI", "Social and Information Networks"),
    ("cs.SY", "Systems and Control"),
    ("eess.AS", "Audio and Speech Processing"),
    ("eess.IV", "Image and Video Processing"),
    ("eess.SP", "Signal Processing"),
    ("eess.SY", "Systems and Control"),
    ("math.ST", "Statistics Theory"),
    ("stat.ML", "Machine Learning"),
    ("stat.AP", "Applications"),
    ("stat.CO", "Computation"),
]


def upgrade() -> None:
    for code, name in _CATEGORIES:
        op.execute(
            f"INSERT INTO categories (code, display_name) "
            f"VALUES ('{code}', '{name}') "
            f"ON CONFLICT (code) DO UPDATE SET display_name = EXCLUDED.display_name"
        )


def downgrade() -> None:
    codes = ", ".join(f"'{c}'" for c, _ in _CATEGORIES)
    op.execute(f"DELETE FROM categories WHERE code IN ({codes})")
