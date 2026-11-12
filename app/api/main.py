"""FastAPI Application Entry Point for CircuitMind AI.

Exposes RESTful endpoints for:
- System Health & Readiness (/health)
- Grounded Datasheet Retrieval (/api/v1/retrieve)
- LangGraph Multi-Agent Copilot Analysis (/api/v1/analyze)
- Interactive OpenAPI & Swagger Documentation (/docs, /redoc)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import get_settings

# Configure structured logging
settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("circuitmind.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to pre-warm the hybrid retriever on startup."""
    try:
        from app.rag.retrieval import get_retriever
        logger.info("Pre-warming hybrid datasheet retriever...")
        retriever = get_retriever()
        _ = retriever.embeddings
        _ = retriever.get_vector_store()
        _ = retriever.get_keyword_retriever()
        logger.info("Hybrid retriever pre-warmed successfully.")
    except Exception as exc:
        logger.debug("Retriever startup pre-warming notice: %s", exc)
    yield


def create_app() -> FastAPI:
    """Factory creating and configuring the CircuitMind AI FastAPI application."""
    app = FastAPI(
        title="CircuitMind AI — Electronics Engineering Copilot API",
        description=(
            "Backend REST API for CircuitMind AI: An Agentic AI copilot for electronics engineers. "
            "Orchestrates hybrid RAG datasheet retrieval, parameter extraction, deterministic electrical "
            "safety validation, and structured engineering report synthesis."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    allowed_origins = settings.cors_origins
    logger.info("Configuring CORS with allowed origins: %s", allowed_origins)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Routers
    app.include_router(api_router)

    @app.get("/", tags=["System"], summary="API Root Overview")
    def read_root() -> Dict[str, str]:
        """Root endpoint returning API service information and documentation links."""
        return {
            "name": "CircuitMind AI — Electronics Engineering Copilot API",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
            "health_check": "/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True if settings.environment == "development" else False,
    )

