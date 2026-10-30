-- KN-VS-01: shared document and chunk tables for all Knowledge workflows (spec Appendix A).
--
-- A document version is (scope, owner, source path, embedding model); its content hash says
-- whether the text changed. Chunks copy scope/user_id so Row-Level Security (V4) needs no join.
-- Runs as a deploy step before the new service version, never at service startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE kn_documents (
    document_id     uuid PRIMARY KEY,
    scope           text        NOT NULL,
    user_id         text,
    source_path     text        NOT NULL,
    content_hash    char(64)    NOT NULL,
    source_commit   text,
    embedding_model text        NOT NULL,
    chunk_count     int         NOT NULL,
    indexed_at      timestamptz NOT NULL,
    -- system rows have no owner; user rows must have a non-empty one ('' is what an unset
    -- user context reads as, so it must never be a real owner).
    CONSTRAINT kn_documents_scope_owner CHECK (
        (scope = 'system' AND user_id IS NULL)
        OR (scope = 'user' AND user_id IS NOT NULL AND user_id <> '')
    ),
    -- Targets for kn_chunks' composite foreign keys, so a chunk can't disagree with its document.
    CONSTRAINT kn_documents_id_scope UNIQUE (document_id, scope),
    CONSTRAINT kn_documents_id_user UNIQUE (document_id, user_id),
    CONSTRAINT kn_documents_id_model UNIQUE (document_id, embedding_model)
);

CREATE UNIQUE INDEX kn_documents_version_uq
    ON kn_documents (scope, coalesce(user_id, ''), source_path, embedding_model);

CREATE TABLE kn_chunks (
    chunk_id        uuid PRIMARY KEY,
    document_id     uuid        NOT NULL REFERENCES kn_documents (document_id) ON DELETE CASCADE,
    scope           text        NOT NULL,
    user_id         text,
    ordinal         int         NOT NULL,
    heading_path    text        NOT NULL,
    text            text        NOT NULL,
    token_count     int         NOT NULL,
    -- 1024 = embedding_dimensions in KN-CHUNK Appendix A; changing it needs a new migration.
    embedding       vector(1024) NOT NULL,
    embedding_model text        NOT NULL,
    CONSTRAINT kn_chunks_scope_owner CHECK (
        (scope = 'system' AND user_id IS NULL)
        OR (scope = 'user' AND user_id IS NOT NULL AND user_id <> '')
    ),
    -- scope always matches the document's; user_id matches whenever the chunk has one (a
    -- system chunk's null user_id is then only possible on a system document).
    CONSTRAINT kn_chunks_document_scope FOREIGN KEY (document_id, scope)
        REFERENCES kn_documents (document_id, scope) ON DELETE CASCADE,
    CONSTRAINT kn_chunks_document_user FOREIGN KEY (document_id, user_id)
        REFERENCES kn_documents (document_id, user_id) ON DELETE CASCADE,
    CONSTRAINT kn_chunks_document_model FOREIGN KEY (document_id, embedding_model)
        REFERENCES kn_documents (document_id, embedding_model) ON DELETE CASCADE,
    CONSTRAINT kn_chunks_document_ordinal UNIQUE (document_id, ordinal)
);
