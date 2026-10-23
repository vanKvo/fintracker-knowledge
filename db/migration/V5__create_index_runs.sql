-- KN-SDOC-06: one row per ingestion run (spec Appendix C).
--
-- Holds commits, counts and timestamps only, never doc text or user data, so it has no scope or
-- Row-Level Security. Only kn_ingest touches it: the workflow writes a run, and the
-- Knowledge-internal index-status route reads the latest one through the same role.

CREATE TABLE kn_index_runs (
    run_id          uuid PRIMARY KEY,
    mode            text        NOT NULL CHECK (mode IN ('incremental', 'rebuild')),
    "commit"        text        NOT NULL,
    embedding_model text        NOT NULL,
    status          text        NOT NULL CHECK (status IN ('succeeded', 'partial', 'failed')),
    docs_indexed    int         NOT NULL,
    docs_skipped    int         NOT NULL,
    docs_deleted    int         NOT NULL,
    docs_failed     int         NOT NULL,
    chunks_written  int         NOT NULL,
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz NOT NULL
);

CREATE INDEX kn_index_runs_finished_at ON kn_index_runs (finished_at DESC);

REVOKE ALL ON kn_index_runs FROM PUBLIC;
GRANT SELECT, INSERT ON kn_index_runs TO kn_ingest;
