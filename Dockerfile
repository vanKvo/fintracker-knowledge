# KN-FAPI-08: one image for Knowledge. The same code runs in Lambda behind Mangum (src/main.py `handler`).
#
# The build context is the monorepo root (not this directory) because the shared internal HTTP library is a path
# dependency at ../../libs/fintracker-internal-http (LB-IHTTP-09). Build with docker-compose.yml, or:
#   docker build -f services/fintracker-knowledge/Dockerfile .
#
# Two stages: Poetry and the build tools stay in the first, so the runtime image holds only the virtualenv and the code.
# Only the source and the dependency files are copied: no tests, golden data, .env files or credentials.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN pip install poetry==2.3.3 && python -m venv /opt/venv

# Same relative layout as the repository, so the path dependency in pyproject.toml resolves.
WORKDIR /repo/services/fintracker-knowledge
COPY libs/fintracker-internal-http/pyproject.toml libs/fintracker-internal-http/README.md /repo/libs/fintracker-internal-http/
COPY libs/fintracker-internal-http/src /repo/libs/fintracker-internal-http/src
COPY services/fintracker-knowledge/pyproject.toml services/fintracker-knowledge/poetry.lock services/fintracker-knowledge/README.md ./
# Dependencies first (cached layer); the service's own code is copied after.
RUN poetry install --only main --no-root
COPY services/fintracker-knowledge/src ./src
RUN poetry install --only main

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /repo /repo
WORKDIR /repo/services/fintracker-knowledge
USER app
EXPOSE 8087
# Liveness only: it touches no dependency (KN-FAPI-08). The orchestrator's readiness check is GET /ready.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8087/health', timeout=2).status == 200 else 1)"
# INTERNAL_AUTH_MODE=local with an AWS runtime marker stops the process at start (LB-IHTTP-02); nothing here sets it.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8087", "--no-server-header"]
