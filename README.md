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

- `src/retrieval/models.py` — strict request validation (Appendix A), response shape (B), documented errors (C).
- `src/retrieval/service.py` — validate, embed, scoped search, ownership re-check, per-document cap of 3, 3 s budget
  (worker thread abandoned on timeout), one DB retry after 1 s.
- `src/retrieval/repository.py` — search as `kn_retrieve`; user context and HNSW settings set in the same transaction.
  `index_commit` is the latest indexed system doc's `source_commit` (`kn_retrieve` cannot read `kn_index_runs`).
- `src/retrieval/router.py` — Lambda handler for `POST /internal/retrieve`: caller allow-list (fails closed), user from
  the `x-internal-user-id` header, never the body.
- `src/retrieval/metrics.py` — per-request metrics and `kn.retrieve.ownership_mismatch` as structured logs, no content.
- Not built yet (no IaC): API Gateway `knowledge-internal` route with IAM auth, the metric filters, dashboard and
  alarms (ownership mismatch on first occurrence, empty-result rate), and the Lambda wiring (Bedrock client with its
  own timeouts, `kn_retrieve` Data API connection).

## Tests

```bash
poetry install
poetry run pytest      # needs Docker: starts pgvector/pgvector:0.8.0-pg16 via Testcontainers
poetry run mypy src
```

The isolation and recall suite (`tests/vector_store/`) runs in CI on every change (KN-VS-05).
