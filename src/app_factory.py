"""KN-FAPI-01, -03: builds the FastAPI app. `main.py` is the process entry point and calls this at import.

One app serves every HTTP route: the internal retrieval route and the operational probes. API Gateway decides which
routes are exposed; the probes are never routed.

__author__ = "Van Vo"
"""

from __future__ import annotations

from fastapi import FastAPI
from fintracker_internal_http import install_internal_exception_handlers

from src.core.config import KnowledgeSettings, load_knowledge_settings
from src.core.container import Container
from src.core.error_handlers import register_error_handlers
from src.health.router import router as health_router
from src.retrieval.router import router as retrieval_router


def create_app(settings: KnowledgeSettings | None = None, *, container: Container | None = None) -> FastAPI:
    """Raises:
    ConfigError / InternalConfigError: invalid configuration; the process must not serve.
    """
    settings = settings or load_knowledge_settings()
    local = settings.internal.is_local  # API docs only in the local container
    app = FastAPI(
        title="FinTracker Knowledge Service",
        version="1.0.0",
        description="Vector store, retrieval and indexing of help documents.",
        docs_url="/docs" if local else None,
        redoc_url="/redoc" if local else None,
        openapi_url="/openapi.json" if local else None,
        redirect_slashes=False,
    )
    app.state.container = container or Container(settings)
    app.include_router(retrieval_router)
    app.include_router(health_router)
    install_internal_exception_handlers(app)
    register_error_handlers(app)
    return app
