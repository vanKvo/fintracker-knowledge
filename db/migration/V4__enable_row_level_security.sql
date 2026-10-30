-- KN-VS-03: Row-Level Security, enabled last (after tables, index and grants exist).
--
-- A row is visible when scope = 'system', or scope = 'user' and user_id equals the
-- app.current_user_id setting. Callers set it with set_config(..., true) inside the same
-- transaction as their query, because each RDS Data API call is otherwise its own session.
-- With no context the setting reads as NULL or '', both of which match no owner.
--
-- FORCE applies the policies to the table owner too. Superusers still bypass RLS, so the
-- Knowledge roles must never be superusers.

ALTER TABLE kn_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE kn_documents FORCE ROW LEVEL SECURITY;
ALTER TABLE kn_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE kn_chunks FORCE ROW LEVEL SECURITY;

-- Reads, and the row check for ingestion writes (WITH CHECK rejects user rows owned by
-- someone other than the current user, on insert and on update).
CREATE POLICY kn_documents_visible ON kn_documents
    FOR SELECT TO kn_ingest, kn_retrieve, kn_delete
    USING (scope = 'system'
           OR (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), '')));

CREATE POLICY kn_documents_ingest_write ON kn_documents
    FOR ALL TO kn_ingest
    USING (scope = 'system'
           OR (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), '')))
    WITH CHECK (scope = 'system'
           OR (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), '')));

-- Account deletion removes only the current user's own rows; system rows are never deletable here.
CREATE POLICY kn_documents_delete_own ON kn_documents
    FOR DELETE TO kn_delete
    USING (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), ''));

CREATE POLICY kn_chunks_visible ON kn_chunks
    FOR SELECT TO kn_ingest, kn_retrieve, kn_delete
    USING (scope = 'system'
           OR (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), '')));

CREATE POLICY kn_chunks_ingest_write ON kn_chunks
    FOR ALL TO kn_ingest
    USING (scope = 'system'
           OR (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), '')))
    WITH CHECK (scope = 'system'
           OR (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), '')));

CREATE POLICY kn_chunks_delete_own ON kn_chunks
    FOR DELETE TO kn_delete
    USING (scope = 'user' AND user_id = nullif(current_setting('app.current_user_id', true), ''));
