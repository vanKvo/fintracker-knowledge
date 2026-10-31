# fintracker-knowledge

Knowledge service: the vector store, retrieval and indexing of help docs (later user documents).
Architecture: `docs/fintracker-knowledge/architecture/knowledge-architecture.md`.

## Implemented so far

KN-VS (vector store): schema, database roles, Row-Level Security, HNSW index, isolation tests.

- `db/migration/` — Flyway migrations for the `knowledge` database. Run them as a separate deploy
  step (`flyway migrate`) before a new service version, never at service startup.
  Order matters: tables, index, roles and grants, then RLS last.
- `src/vector_store/session.py` — SQL every session runs in the same transaction as its query
  (user context, HNSW search settings).
- Roles `kn_ingest`, `kn_retrieve`, `kn_delete` are created `NOLOGIN`; provision their credentials
  (Secrets Manager / Data API) outside the schema. They must never be superusers or have BYPASSRLS.

KN-CHUNK (chunking and embedding):

- `src/config/knowledge_config.py` — Appendix A defaults (chunk sizes, `active_embedding_model`, dimensions).
- `src/chunking/` — heading-based Markdown chunker (pure, deterministic).
- `src/embedding/` — Bedrock embedder for chunks and queries, with retries and `embedding_unavailable`.
- `src/model_switch/` — switches the active embedding model after a full rebuild and a passing quality gate.
- `src/retrieval_quality/` — recall@5 gate over `eval/golden-questions.yaml` (not yet written) with a stored baseline.

KN-SDOC (system docs indexing):

- `src/system_docs/publisher.py` — CI side: uploads changed help docs, then the manifest (`python -m src.system_docs.publisher`).
- `src/system_docs/trigger.py` — manifest parsing and the idempotent execution start (execution named after the commit).
- `src/system_docs/service.py` — one isolated branch per doc: hash skip, chunk, embed, atomic replace, delete,
  retries (2 s, 4 s, 8 s), full rebuild, run record. Branch logic is plain Python so a Step Functions Map state can call it.
- `src/system_docs/status.py`, `access.py` — index status, alarm rules, caller allow-list (fails closed).
- `db/migration/V5__create_index_runs.sql` — `kn_index_runs`, readable and writable by `kn_ingest` only.
- Not built yet (no IaC for this service): the S3 bucket policy, the S3 event and Step Functions wiring, the
  CI workflow that calls the publisher, the alarms, and the `GET /internal/index-status` HTTP route.

KN-RET (retrieval):

- `src/retrieval/schemas.py` — strict request validation (Appendix A) and response shape (B); errors (C) are the shared `src/core/exceptions.py`, returned as RFC 9457 problem+json by `src/core/error_handlers.py`.
- `src/retrieval/service.py` — validate, embed, scoped search, ownership re-check, per-document cap of 3, 3 s budget
  (worker thread abandoned on timeout), one DB retry after 1 s.
- `src/retrieval/repository.py` — search as `kn_retrieve`; user context and HNSW settings set in the same transaction.
  `index_commit` is the latest indexed system doc's `source_commit` (`kn_retrieve` cannot read `kn_index_runs`).
- `src/retrieval/router.py` — the FastAPI route for `POST /internal/retrieve`: caller allow-list (fails closed), user from
  the `x-internal-user-id` header, never the body. The caller check and the user context are the shared
  [`fintracker-internal-http`](../../libs/fintracker-internal-http/README.md) library's, wired in `src/core/internal_auth.py`:
  `INTERNAL_ALLOWED_CALLER_ARNS` (comma-separated role ARN prefixes; unset rejects every call) and, for a laptop only,
  `INTERNAL_AUTH_MODE=local` (the caller is a header; refused in an AWS runtime, never set by IaC).
- `src/retrieval/metrics.py` — per-request metrics and `kn.retrieve.ownership_mismatch` as structured logs, no content.
- Not built yet (no IaC): API Gateway `knowledge-internal` route with IAM auth, the metric filters, dashboard and
  alarms (ownership mismatch on first occurrence, empty-result rate), and the deployment (Lambda or container service).

KN-FAPI (FastAPI container):

- `src/main.py` — `app` (uvicorn) and `handler` (Mangum, for Lambda). Reads and checks the configuration at import:
  a missing variable stops the process naming the variable (never its value). `src/app_factory.py` builds the app.
- Routes: `POST /internal/retrieve`, and `GET /health` (touches nothing) and `GET /ready` (database as `kn_retrieve` and
  Bedrock credentials; 503 otherwise). API docs (`/docs`) only with `INTERNAL_AUTH_MODE=local`.
- `src/core/config.py` — settings (below). `src/core/container.py` builds each part once, on first use;
  `src/core/dependencies.py` exposes them to `app.dependency_overrides`.
- `src/vector_store/connections.py` — one connection factory per database role: `postgres` mode (login role that is a member
  of the NOLOGIN role; at start the app refuses a superuser or `BYPASSRLS` role) and `data_api` mode (RDS Data API).
- `src/system_docs/cli.py` — `python -m src.system_docs.cli index|rebuild|status --root docs`: indexes `docs/help/**/*.md`
  as `kn_ingest`, and refuses a database host that is not local. With `KNOWLEDGE_EMBEDDER=replay` it embeds from the
  recorded snapshot; with real Bedrock it goes through the owner guard (see "Live RAG evaluation").
- `Dockerfile` — one image, non-root, port 8087; build context is the monorepo root (the shared library is a path dependency).

| Variable | Notes |
|---|---|
| `INTERNAL_AUTH_MODE`, `INTERNAL_ALLOWED_CALLER_ARNS` | The shared library's. `local` only in the compose file |
| `AWS_REGION` | Required. Credentials from the default chain. A global `AWS_ENDPOINT_URL` stops the start (it would redirect Bedrock) |
| `KNOWLEDGE_EMBEDDER`, `KNOWLEDGE_REPLAY_DIR` | `bedrock` (default) or `replay`. Replay embeds only the recorded golden questions and help chunks from `KNOWLEDGE_REPLAY_DIR/recording.json.gz`; any other text is `embedding_unavailable` (503) at once. Refused when an AWS runtime variable is set. The compose file uses replay |
| `KNOWLEDGE_DB_MODE` | `postgres` or `data_api`. Required |
| `KNOWLEDGE_DB_HOST`, `_PORT`, `_NAME`, `KNOWLEDGE_DB_RETRIEVE_USER`, `_PASSWORD` | `postgres` mode, the API. The index command reads `KNOWLEDGE_DB_INGEST_USER`, `_PASSWORD` instead |
| `KNOWLEDGE_DB_CLUSTER_ARN`, `KNOWLEDGE_DB_RETRIEVE_SECRET_ARN` | `data_api` mode |
| `LOG_LEVEL` | Default `INFO` |

## Run it locally

```bash
cp .env.local.example .env.local        # set the passwords; the file is git-ignored
docker compose up -d                    # Flyway (one-shot), the app on :8087
docker compose --profile index run --rm knowledge-index   # indexes the recorded snapshot; no AWS, no cost
poetry run python scripts/smoke.py      # health, 401/403, retrieval, cross-user check
```

The database is `knowledge` in the shared PostgreSQL container (`infrastructure/local-postgres/`, also used by the Ledger).
Start that container first (`docker compose up -d` there, once). Its first start creates the `knowledge` database and the
login roles, so `local/setup-local-database.sh` is not needed. The app connects as `kn_app_retrieve`, so Row-Level Security
applies as in AWS. Full steps: `docs/instructions/knowledge_service_build_test_deploy.md`.

The local stack holds **no AWS credentials** and mounts no `~/.aws`. It runs in replay mode on `tests/fixtures/rag/`
(mounted read-only at `/rag`): it can answer the golden questions and nothing else, and it can never call Bedrock.

## Tests

```bash
poetry install
poetry run pytest      # needs Docker: starts pgvector/pgvector:0.8.0-pg16 via Testcontainers
poetry run mypy src
```

The isolation and recall suite (`tests/vector_store/`) runs in CI on every change (KN-VS-05). The image tests
(`tests/container/`) build the image and run only with `KNOWLEDGE_IMAGE_TESTS=1`:

```bash
KNOWLEDGE_IMAGE_TESTS=1 poetry run pytest tests/container
```

## Live RAG evaluation (KN-CHUNK-05)

Three layers: unit tests (fake client), the replay run (real chunker, pgvector, retrieval and gate with vectors from
`tests/fixtures/rag/recording.json.gz`), and the live run. Only the live run calls Bedrock, and only the repo owner
starts it.

```bash
python -m src.rag_eval dry                                   # plan and worst-case price; calls nothing, needs no AWS
AWS_REGION=us-east-1 RAG_EVAL_RESULTS_BUCKET=<bucket> \
  python -m src.rag_eval live --aws-profile <your-profile>   # asks you to type "I approve N"
```

- Re-uses vectors already recorded, so a second run costs nothing and a changed help page costs only its changed chunks.
- Default cost cap $0.50 for a run; a higher worst case needs `--cap-usd` and `--confirm-over-cap`.
- Indexes into a throwaway Testcontainers database (needs Docker), never your local one. Only a passing run replaces
  the fixture and baseline, and only a pass writes the S3 pass record a pull request needs.
- Commit the refreshed `tests/fixtures/rag/` after a pass. A pull request that changes the chunker, embedded text,
  retrieval query, golden set or recording is blocked by the `eval-required` job until a pass exists for its exact version.
- The same run is available as the manual `rag-eval` workflow (protected environment, OIDC role). Terraform:
  `infrastructure/terraform/README.md`.

**AI agents never run it.** Tests clear AWS credentials and make creating a Bedrock client raise
(`tests/conftest.py`); a test fails if any module other than `core/container.py` and `rag_eval/live.py` can name a
Bedrock service; the live command refuses under pytest, when an agent marker is set, without a terminal, and with static
AWS credentials in the environment. These are speed bumps: the real control is that no agent session holds credentials.
`.claude/settings.json` denies the command and AWS tools to Claude Code.
