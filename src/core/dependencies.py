"""FastAPI dependencies: the container and what is built from it (KN-FAPI-03).

A test replaces any of them with `app.dependency_overrides[...]`.

__author__ = "Van Vo"
"""

from __future__ import annotations

from fastapi import Depends, Request

from src.core.container import Container
from src.retrieval.metrics import RetrievalMetrics
from src.retrieval.service import RetrievalService


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_retrieval_service(container: Container = Depends(get_container)) -> RetrievalService:
    return container.retrieval_service()


def get_metrics(container: Container = Depends(get_container)) -> RetrievalMetrics:
    return container.metrics()
