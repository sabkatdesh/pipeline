# Import all models here so SQLAlchemy's mapper and Alembic's autogenerate
# can discover every table from a single import.
from app.models.author import Author, PaperAuthor
from app.models.category import Category, PaperCategory
from app.models.concept import Concept, PaperConcept
from app.models.ingestion_run import IngestionRun
from app.models.paper import Paper

__all__ = [
    "Author",
    "Category",
    "Concept",
    "IngestionRun",
    "Paper",
    "PaperAuthor",
    "PaperCategory",
    "PaperConcept",
]
