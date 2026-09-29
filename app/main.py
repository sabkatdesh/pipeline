from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import ingestion, stats
from app.core.config import get_settings
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    setup_logging(cfg.log_level)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="arXiv Intelligence Pipeline",
        description="Ingest, store, and query arXiv papers with a RAG pipeline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingestion.router, prefix="/api/v1")
    app.include_router(stats.router, prefix="/api/v1")

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    async def ready():
        from sqlalchemy import text
        from app.core.database import AsyncSessionLocal
        from app.models.paper import Paper
        from sqlalchemy import func, select

        try:
            async with AsyncSessionLocal() as session:
                total = await session.scalar(select(func.count(Paper.arxiv_id)))
                embedded = await session.scalar(
                    select(func.count(Paper.arxiv_id)).where(Paper.embedding.is_not(None))
                )
            return {"status": "ready", "papers_count": total, "embedded_count": embedded}
        except Exception as exc:
            from fastapi import Response
            return Response(
                content='{"status":"not_ready","reason":"database unreachable"}',
                status_code=503,
                media_type="application/json",
            )

    return app


app = create_app()
